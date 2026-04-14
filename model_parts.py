import json
import math
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F


class GroupedQueryAttentionWithRoPE(nn.Module):
    """
    Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE).

    Dimension legend (used throughout):
        b          = batch size
        T          = sequence length (num_tokens)
        d_in       = input feature dim
        d_out      = output feature dim  (= num_heads * head_dim)
        H          = num_heads  (query heads)
        Hkv        = num_kv_heads
        G          = num_groups = H // Hkv
        D          = head_dim   = d_out // H
    """

    def __init__(
        self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        num_kv_heads=None,
        qkv_bias=False,
        use_rope=True,
        window_size=None,
        use_attention_bias=True,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.num_heads    = num_heads                                          # H
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads  # Hkv
        assert num_heads % self.num_kv_heads == 0, \
            "num_heads must be divisible by num_kv_heads"

        self.num_groups       = num_heads // self.num_kv_heads                 # G
        self.head_dim         = d_out // num_heads                             # D
        self.d_out            = d_out
        self.use_rope         = use_rope
        self.window_size      = window_size
        self.use_attention_bias = use_attention_bias

        # Q projects to (H * D), K/V project to (Hkv * D)
        self.W_q = nn.Linear(d_in, d_out,                          bias=qkv_bias)
        self.W_k = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout  = nn.Dropout(dropout)

        # Learnable additive bias (applied after scaling, before masking)
        if use_attention_bias:
            # shape: (1, H, context_length, context_length)
            self.attention_bias = nn.Parameter(
                torch.zeros(1, self.num_heads, context_length, context_length)
            )

        if use_rope:
            # cos/sin: (context_length, D)
            cos, sin = self._precompute_rope_cos_sin(context_length, self.head_dim)
            self.register_buffer("cos_cached", cos)
            self.register_buffer("sin_cached", sin)

    # ------------------------------------------------------------------
    # RoPE helpers
    # ------------------------------------------------------------------

    def _precompute_rope_cos_sin(self, seq_len, dim, theta=10000.0):
        """
        Returns cos, sin each of shape (seq_len, dim).
        The full embedding is [freqs, freqs] so it covers all of dim.
        """
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        # inv_freq: (dim//2,)
        t = torch.arange(seq_len, device=inv_freq.device).float()
        # t: (seq_len,)
        freqs = torch.outer(t, inv_freq)       # (seq_len, dim//2)
        emb   = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        return emb.cos(), emb.sin()            # each (seq_len, dim)

    def _apply_rotary_emb(self, x, cos, sin):
        """
        Apply RoPE in-place style.

        Args:
            x:   (b, num_heads, T, D)   — query or key tensor
            cos: (1, 1, T, D)           — pre-sliced & pre-shaped by caller
            sin: (1, 1, T, D)

        Returns:
            rotated tensor, same shape as x
        """
        # Split along the LAST dimension into two halves of size D//2
        # x1, x2: (b, num_heads, T, D//2)
        x1, x2 = x.chunk(2, dim=-1)

        # cos/sin are (1, 1, T, D), but we only need the first D//2 entries
        # because emb = [freqs, freqs], both halves are identical
        # So cos[..., :D//2] and sin[..., :D//2] are what we want.
        # Alternatively, chunk them the same way:
        cos1, _ = cos.chunk(2, dim=-1)   # (1, 1, T, D//2)
        sin1, _ = sin.chunk(2, dim=-1)   # (1, 1, T, D//2)

        o1 = x1 * cos1 - x2 * sin1      # (b, H, T, D//2)
        o2 = x2 * cos1 + x1 * sin1      # (b, H, T, D//2)
        return torch.cat((o1, o2), dim=-1)  # (b, H, T, D)

    # ------------------------------------------------------------------
    # GQA helper
    # ------------------------------------------------------------------

    def _repeat_kv(self, kv, num_repeats):
        """
        Expand KV heads to match the number of query heads.

        Args:
            kv:          (b, Hkv, T, D)
            num_repeats: G  (= H // Hkv)

        Returns:
            (b, H, T, D)
        """
        if num_repeats == 1:
            return kv
        b, hkv, T, D = kv.shape
        kv = kv[:, :, None, :, :]                         # (b, Hkv, 1, T, D)
        kv = kv.expand(b, hkv, num_repeats, T, D)         # (b, Hkv, G, T, D)
        return kv.reshape(b, hkv * num_repeats, T, D)     # (b, H, T, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x):
        """
        Args:
            x: (b, T, d_in)

        Returns:
            (b, T, d_out)
        """
        b, T, d_in = x.shape

        # ── Projections ──────────────────────────────────────────────
        # Q: (b, T, d_out)  →  (b, H, T, D)
        queries = self.W_q(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)

        # K: (b, T, Hkv*D)  →  (b, Hkv, T, D)
        keys    = self.W_k(x).view(b, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # V: (b, T, Hkv*D)  →  (b, Hkv, T, D)
        values  = self.W_v(x).view(b, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── RoPE ─────────────────────────────────────────────────────
        if self.use_rope:
            # Slice to current sequence length, then unsqueeze for broadcasting
            # cos_cached: (context_length, D) → slice → (T, D) → (1, 1, T, D)
            cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)
            sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)

            # _apply_rotary_emb expects cos/sin as (1, 1, T, D) — no further unsqueeze inside
            queries = self._apply_rotary_emb(queries, cos, sin)  # (b, H,   T, D)
            keys    = self._apply_rotary_emb(keys,    cos, sin)  # (b, Hkv, T, D)

        # ── GQA: repeat KV heads ─────────────────────────────────────
        keys   = self._repeat_kv(keys,   self.num_groups)  # (b, H, T, D)
        values = self._repeat_kv(values, self.num_groups)  # (b, H, T, D)

        # ── Attention scores ─────────────────────────────────────────
        # (b, H, T, D) @ (b, H, D, T) → (b, H, T, T)
        attn = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Learnable bias (added after scaling)
        if self.use_attention_bias:
            attn = attn + self.attention_bias[:, :, :T, :T]  # (1, H, T, T) broadcast

        # Causal mask: mask out future positions with -inf
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn = attn.masked_fill(causal_mask, float("-inf"))  # (b, H, T, T)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # ── Context ──────────────────────────────────────────────────
        # (b, H, T, T) @ (b, H, T, D) → (b, H, T, D)
        context = torch.matmul(attn, values)

        # (b, H, T, D) → (b, T, H*D) = (b, T, d_out)
        context = context.transpose(1, 2).contiguous().view(b, T, self.d_out)

        return self.out_proj(context)  # (b, T, d_out)


import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

@dataclass
class MoEConfig:
    d_model:        int   = 512    # token embedding dim
    d_ff:           int   = 2048   # each expert's inner dim
    num_experts:    int   = 8      # E
    top_k:          int   = 2      # k  (active experts per token)
    dropout:        float = 0.1
    # Load-balancing loss coefficient (α in Switch Transformer paper §2.2)
    aux_loss_coef:  float = 1e-2
    # Capacity factor C: expert buffer = C * (T / E) tokens.
    # Used only during training; set None to skip capacity limiting.
    capacity_factor: Optional[float] = 1.25


# ──────────────────────────────────────────────────────────────────────
# Expert: a single FFN  (identical in structure to a standard FFN block)
# ──────────────────────────────────────────────────────────────────────

class Expert(nn.Module):
    """
    One expert FFN.

    Dimensions:
        input/output : (tokens_for_this_expert, d_model)
        hidden       : (tokens_for_this_expert, d_ff)
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1  = nn.Linear(d_model, d_ff)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (n_tokens, d_model)
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ──────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────

class TopKRouter(nn.Module):
    """
    Linear router: projects each token to E logits, picks top-k experts.

    Reference: Shazeer et al. 2017 §2, Switch Transformer §2.1

    Forward returns:
        gate_weights : (b*T, k)   softmax-normalised weights for selected experts
        expert_idx   : (b*T, k)   indices of selected experts
        router_probs : (b*T, E)   full softmax distribution (for aux loss)
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.W_r = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        # x: (N, d_model)  where N = b*T (flattened batch×seq)
        logits      = self.W_r(x)                           # (N, E)
        router_probs = F.softmax(logits, dim=-1)             # (N, E)

        # top-k selection
        gate_weights, expert_idx = torch.topk(router_probs, self.top_k, dim=-1)
        # gate_weights: (N, k) — re-normalise so selected weights sum to 1
        gate_weights = gate_weights / gate_weights.sum(dim=-1, keepdim=True)

        return gate_weights, expert_idx, router_probs


# ──────────────────────────────────────────────────────────────────────
# Load-balancing auxiliary loss
# ──────────────────────────────────────────────────────────────────────

def load_balance_loss(router_probs: torch.Tensor,
                      expert_idx:   torch.Tensor,
                      num_experts:  int) -> torch.Tensor:
    """
    Switch Transformer auxiliary loss (§2.2):

        L_aux = α · E · Σ_i  f_i · P_i

    where
        f_i = fraction of tokens routed to expert i
        P_i = mean router probability assigned to expert i

    Both are (E,) vectors; their dot product measures imbalance.
    Minimising L_aux encourages uniform expert utilisation.

    Args:
        router_probs : (N, E)  full softmax over all experts
        expert_idx   : (N, k)  selected expert indices per token
        num_experts  : E

    Returns:
        scalar loss
    """
    N = router_probs.size(0)

    # f_i: fraction of tokens that chose expert i (any of the k slots)
    # one-hot over E for every (token, slot) pair → sum over slots → mean over tokens
    one_hot = F.one_hot(expert_idx, num_classes=num_experts).float()  # (N, k, E)
    # token routed to expert i if it appears in ANY of the k slots
    token_mask = one_hot.sum(dim=1).clamp(max=1.0)                     # (N, E)  binary
    f = token_mask.mean(dim=0)                                          # (E,)

    # P_i: mean router probability for expert i across all tokens
    P = router_probs.mean(dim=0)                                        # (E,)

    return num_experts * (f * P).sum()


# ──────────────────────────────────────────────────────────────────────
# MoE FFN layer  (drop-in replacement for a standard FFN)
# ──────────────────────────────────────────────────────────────────────

class MoELayer(nn.Module):
    """
    Mixture-of-Experts FFN layer.

    Replaces a standard FFN inside a Transformer block.
    Each token is independently routed to top-k experts;
    their outputs are combined via a weighted sum.

    Data flow (N = b*T tokens):
        x (b, T, d_model)
          → flatten → (N, d_model)
          → router  → gate_weights (N, k), expert_idx (N, k)
          → per-expert FFN on dispatched tokens
          → accumulate weighted outputs
          → unflatten → (b, T, d_model)

    Capacity limiting (training):
        Each expert processes at most ⌊C · N / E⌋ tokens.
        Tokens that overflow an expert are dropped (their contribution
        is zeroed). C=1.25 is the Switch Transformer default.

    Returns:
        output   : (b, T, d_model)
        aux_loss : scalar — add α * aux_loss to your total training loss
    """

    def __init__(self, cfg: MoEConfig):
        super().__init__()
        self.cfg        = cfg
        self.router     = TopKRouter(cfg.d_model, cfg.num_experts, cfg.top_k)
        self.experts    = nn.ModuleList([
            Expert(cfg.d_model, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.num_experts)
        ])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (b, T, d_model)

        Returns:
            out      : (b, T, d_model)
            aux_loss : scalar tensor
        """
        b, T, d = x.shape
        N = b * T

        # ── Flatten ───────────────────────────────────────────────────
        x_flat = x.view(N, d)                          # (N, d_model)

        # ── Route ─────────────────────────────────────────────────────
        # gate_weights : (N, k)   re-normalised top-k probabilities
        # expert_idx   : (N, k)   which experts to call
        # router_probs : (N, E)   full distribution (for aux loss)
        gate_weights, expert_idx, router_probs = self.router(x_flat)

        # ── Aux loss ──────────────────────────────────────────────────
        aux_loss = self.cfg.aux_loss_coef * load_balance_loss(
            router_probs, expert_idx, self.cfg.num_experts
        )

        # ── Expert capacity ───────────────────────────────────────────
        # capacity: maximum tokens an expert will accept this forward pass
        if self.cfg.capacity_factor is not None and self.training:
            capacity = int(self.cfg.capacity_factor * N / self.cfg.num_experts)
            capacity = max(capacity, 1)
        else:
            capacity = N   # no limit at inference

        # ── Dispatch & combine ────────────────────────────────────────
        # Output accumulator: start at zero, add each expert's contribution
        out_flat = torch.zeros_like(x_flat)            # (N, d_model)

        for expert_id, expert in enumerate(self.experts):
            # Find all (token, slot) pairs assigned to this expert
            # expert_idx: (N, k)  — check all k slots
            token_mask, slot_idx = torch.where(expert_idx == expert_id)
            # token_mask : 1-D indices into [0..N) of tokens routed here
            # slot_idx   : which of the k slots matched (needed to read gate weight)

            if token_mask.numel() == 0:
                continue   # expert unused this batch

            # Apply capacity: truncate if over budget
            if token_mask.numel() > capacity:
                token_mask = token_mask[:capacity]
                slot_idx   = slot_idx[:capacity]

            # Gather the tokens assigned to this expert
            expert_input  = x_flat[token_mask]         # (n_e, d_model)

            # Run the expert FFN
            expert_output = expert(expert_input)        # (n_e, d_model)

            # Retrieve gate weights for this expert/slot combination
            w = gate_weights[token_mask, slot_idx]      # (n_e,)

            # Weighted accumulate back into output (scatter-add)
            out_flat.index_add_(
                0,
                token_mask,
                expert_output * w.unsqueeze(-1)         # (n_e, d_model)
            )

        # ── Unflatten ─────────────────────────────────────────────────
        out = out_flat.view(b, T, d)                    # (b, T, d_model)

        return out, aux_loss




