import torch
import torch.nn as nn
from typing import List, Optional

from config_model import ModelConfig
from model_parts import Transformer, LayerKVCache

# Optional TPU support — no-op when torch_xla is not installed.
try:
    import torch_xla.core.xla_model as xm
    _USE_XLA = True
except ImportError:
    xm = None
    _USE_XLA = False


def _xla_mark_step(t: torch.Tensor) -> None:
    """Flush the lazy XLA graph so TPU work actually executes each step."""
    if _USE_XLA and t.device.type == "xla":
        xm.mark_step()


class LLM(nn.Module):
    """
    Decoder-only LLM with standard multi-head attention, RoPE,
    dense feed-forward, and RMSNorm pre-norm.

    Supports three operating modes via `use_cache`:

    ┌─────────────────┬───────────────────────────────────────────────────┐
    │ Mode            │ How to call                                       │
    ├─────────────────┼───────────────────────────────────────────────────┤
    │ Training        │ logits = model(ids, use_cache=False)              │
    │                 │   — no cache allocated, full causal mask          │
    ├─────────────────┼───────────────────────────────────────────────────┤
    │ Prefill         │ logits = model(prompt_ids, use_cache=True)        │
    │                 │   — allocates cache on first call, fills cache    │
    │                 │     with prompt KVs                               │
    ├─────────────────┼───────────────────────────────────────────────────┤
    │ Decode          │ logits = model(next_tok, use_cache=True)          │
    │  (one token)    │   — appends one token to cache each step,         │
    │                 │     no mask needed (cache order = causality)      │
    └─────────────────┴───────────────────────────────────────────────────┘

    Call  model.reset_cache()  between independent generation requests.
    Call  model.build_cache(batch_size, device)  to pre-allocate the cache
    manually before generation (optional — auto-allocated on first use).

    Returns
    -------
        logits : (b, T, vocabulary_size)
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.embed    = nn.Embedding(cfg.vocabulary_size, cfg.d_model)
        self.drop     = nn.Dropout(cfg.dropout)
        self.layers   = nn.ModuleList(
            [Transformer(cfg) for _ in range(cfg.transformer_blocks)]
        )
        self.norm     = nn.RMSNorm(cfg.d_model)
        self.out_head = nn.Linear(cfg.d_model, cfg.vocabulary_size, bias=False)

        # Weight tying: shares the ~51M embedding matrix with the output
        # head instead of allocating it twice. Required to stay <100M
        # with the 200k o200k_base vocab.
        if cfg.tie_word_embeddings:
            self.out_head.weight = self.embed.weight

        # Cache — None until build_cache() is called or use_cache=True triggers
        # auto-allocation on the first forward pass.
        self.kv_caches: Optional[List[LayerKVCache]] = None

    # ── Cache lifecycle ────────────────────────────────────────────────────────

    def build_cache(
        self,
        batch_size: int,
        device:     torch.device,
        dtype:      torch.dtype = torch.float32,
    ) -> None:
        """Pre-allocate KV caches for all layers."""
        head_dim = self.cfg.d_model // self.cfg.num_heads
        self.kv_caches = [
            LayerKVCache(
                batch_size  = batch_size,
                num_heads   = self.cfg.num_heads,
                max_seq_len = self.cfg.max_seq_len,
                head_dim    = head_dim,
                device      = device,
                dtype       = dtype,
            )
            for _ in range(self.cfg.transformer_blocks)
        ]

    def reset_cache(self) -> None:
        """Zero all cache buffers and reset fill pointers to 0."""
        if self.kv_caches is not None:
            for cache in self.kv_caches:
                cache.reset()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: torch.Tensor,          # (b, T)
        use_cache: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : (b, T)  — token ids
        use_cache : bool
            False → training / eval without cache (full causal mask, T tokens).
            True  → generation; cache is auto-allocated on first call if needed.

        Returns
        -------
        logits : (b, T, vocabulary_size)
        """
        b, _ = input_ids.shape

        if use_cache and self.kv_caches is None:
            self.build_cache(
                batch_size = b,
                device     = input_ids.device,
                dtype      = self.embed.weight.dtype,
            )

        x = self.drop(self.embed(input_ids))   # (b, T, d_model)

        for i, layer in enumerate(self.layers):
            cache = self.kv_caches[i] if use_cache else None
            x = layer(x, kv_cache=cache)

        x      = self.norm(x)
        logits = self.out_head(x)   # (b, T, vocabulary_size)

        return logits

@torch.no_grad()  # no_grad (not inference_mode): in-place KV-cache writes are illegal on XLA inference tensors
def generate(
    model:          LLM,
    prompt_ids:     torch.Tensor,       # (b, T_prompt)
    max_new_tokens: int,
    temperature:    float = 1.0,
    top_k:          int   = 0,          # 0 = disabled
    eos_token_id:   Optional[int] = None,
) -> torch.Tensor:
    """
    Autoregressive generation using KV cache.

    1. Prefill — run the full prompt through the model to populate the cache.
    2. Decode  — feed one token at a time, appending to the cache each step.

    Returns
    -------
    generated : (b, T_prompt + max_new_tokens) — prompt + new tokens
    """
    model.eval()
    model.reset_cache()

    generated = prompt_ids.clone()         # (b, T_prompt)

    # ── Prefill ────────────────────────────────────────────────────────────
    logits = model(prompt_ids, use_cache=True)   # (b, T_prompt, V)
    _xla_mark_step(logits)
    next_logits = logits[:, -1, :]               # (b, V)

    # ── Decode loop ────────────────────────────────────────────────────────
    for _ in range(max_new_tokens):

        next_token = _sample(next_logits, temperature=temperature, top_k=top_k)  # (b, 1)
        generated  = torch.cat([generated, next_token], dim=1)

        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

        logits      = model(next_token, use_cache=True)   # (b, 1, V)
        _xla_mark_step(logits)
        next_logits = logits[:, -1, :]                    # (b, V)

    return generated


def _sample(
    logits:      torch.Tensor,   # (b, V)
    temperature: float,
    top_k:       int,
) -> torch.Tensor:               # (b, 1)
    """Temperature + optional top-k sampling. Returns token ids (b, 1)."""
    if temperature != 1.0:
        logits = logits / temperature

    if top_k > 0:
        values, _ = torch.topk(logits, top_k, dim=-1)
        threshold  = values[:, -1].unsqueeze(-1)
        logits     = logits.masked_fill(logits < threshold, float("-inf"))

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)   # (b, 1)
