# model_parts.py — standard dense Transformer blocks:
# Multi-Head Attention + RoPE, dense FFN, RMSNorm pre-norm, KV cache.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from config_model import ModelConfig


# ──────────────────────────────────────────────────────────────────────────────
# KV Cache
# ──────────────────────────────────────────────────────────────────────────────

class LayerKVCache:
    def __init__(
        self,
        batch_size:  int,
        num_heads:   int,
        max_seq_len: int,
        head_dim:    int,
        device:      torch.device,
        dtype:       torch.dtype = torch.float32,
    ):
        self.max_seq_len  = max_seq_len
        self.seq_len: int = 0

        self.k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )
        self.v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype,
        )

    def update(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ):
        T_new = new_k.shape[2]
        end   = self.seq_len + T_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: position {end - 1} >= max_seq_len {self.max_seq_len}. "
                "Call reset() between requests or increase max_seq_len."
            )
        self.k_cache[:, :, self.seq_len:end] = new_k
        self.v_cache[:, :, self.seq_len:end] = new_v
        self.seq_len = end
        return self.k_cache[:, :, :end], self.v_cache[:, :, :end]

    def reset(self) -> None:
        self.k_cache.zero_()
        self.v_cache.zero_()
        self.seq_len = 0


# ──────────────────────────────────────────────────────────────────────────────
# Standard Multi-Head Attention + RoPE (KV-cache compatible)
# ──────────────────────────────────────────────────────────────────────────────

class MultiHeadAttentionWithRoPE(nn.Module):

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = cfg.num_heads
        self.head_dim  = cfg.d_model // cfg.num_heads
        self.d_out     = cfg.d_model
        self.use_rope  = cfg.use_rope

        self.W_q      = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.qkv_bias)
        self.W_k      = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.qkv_bias)
        self.W_v      = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.qkv_bias)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout  = nn.Dropout(cfg.dropout)

        if cfg.use_rope:
            cos, sin = self._precompute_rope(cfg.max_seq_len, self.head_dim)
            self.register_buffer("cos_cached", cos)   # (max_seq_len, head_dim)
            self.register_buffer("sin_cached", sin)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _precompute_rope(seq_len: int, dim: int, theta: float = 10_000.0):
        """
        Returns cos/sin tables of shape (seq_len, dim).
        inv_freq has dim//2 entries; we cat(freqs, freqs) to fill all dim slots
        so that the rotate-half trick works without any re-chunking.
        """
        assert dim % 2 == 0, "head_dim must be even for RoPE"
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (dim//2,)
        t        = torch.arange(seq_len).float()                               # (seq_len,)
        freqs    = torch.outer(t, inv_freq)                                    # (seq_len, dim//2)
        emb      = torch.cat((freqs, freqs), dim=-1)                           # (seq_len, dim)
        return emb.cos(), emb.sin()

    @staticmethod
    def _apply_rotary_emb(
        x:   torch.Tensor,   # (b, H, T, D)
        cos: torch.Tensor,   # (1, 1, T, D)
        sin: torch.Tensor,   # (1, 1, T, D)
    ) -> torch.Tensor:
        """Rotate-half RoPE. cos/sin are already (1,1,T,D)."""
        D = x.shape[-1]
        half = D // 2
        x1 = x[..., :half]    # (b, H, T, D//2)
        x2 = x[..., half:]    # (b, H, T, D//2)
        c  = cos[..., :half]  # (1, 1, T, D//2)
        s  = sin[..., :half]  # (1, 1, T, D//2)
        return torch.cat([x1 * c - x2 * s,
                          x2 * c + x1 * s], dim=-1)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        x:        torch.Tensor,              # (b, T, d_model)
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:

        b, T, _ = x.shape

        q = self.W_q(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(b, T, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            offset = kv_cache.seq_len if kv_cache is not None else 0

            max_pos = self.cos_cached.shape[0]
            if offset + T > max_pos:
                raise ValueError(
                    f"RoPE table overflow: offset ({offset}) + T ({T}) = {offset + T} "
                    f"> max_seq_len ({max_pos}). Increase context_length in ModelConfig."
                )

            cos = self.cos_cached[offset:offset + T]            # (T, D)
            sin = self.sin_cached[offset:offset + T]            # (T, D)
            cos = cos.unsqueeze(0).unsqueeze(0)                  # (1, 1, T, D)
            sin = sin.unsqueeze(0).unsqueeze(0)                  # (1, 1, T, D)

            q = self._apply_rotary_emb(q, cos, sin)
            k = self._apply_rotary_emb(k, cos, sin)

        if kv_cache is not None:
            k_full, v_full = kv_cache.update(k, v)
        else:
            k_full, v_full = k, v

        # (b, H, T, D) x (b, H, D, S) -> (b, H, T, S)
        attn = torch.matmul(q, k_full.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if kv_cache is None:
            causal = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
            )
            attn = attn.masked_fill(causal, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        ctx = torch.matmul(attn, v_full)                         # (b, H, T, D)
        ctx = ctx.transpose(1, 2).contiguous().view(b, T, self.d_out)
        return self.out_proj(ctx)


# ──────────────────────────────────────────────────────────────────────────────
# Standard dense feed-forward
# ──────────────────────────────────────────────────────────────────────────────

class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1  = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2  = nn.Linear(cfg.d_ff,    cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


# ──────────────────────────────────────────────────────────────────────────────
# Transformer block (pre-norm with RMSNorm)
# ──────────────────────────────────────────────────────────────────────────────

class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn  = MultiHeadAttentionWithRoPE(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn   = FeedForward(cfg)

    def forward(
        self,
        x:        torch.Tensor,
        kv_cache: Optional[LayerKVCache] = None,
    ) -> torch.Tensor:

        x = x + self.attn(self.norm1(x), kv_cache=kv_cache)
        x = x + self.ffn(self.norm2(x))
        return x
