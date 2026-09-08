from dataclasses import dataclass

try:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    _VOCAB_SIZE = enc.n_vocab
except Exception:
    _VOCAB_SIZE = 200_019   # o200k_base vocab size fallback


@dataclass
class ModelConfig:
    # Shared (~57M params total with weight tying, see below)
    d_model:            int   = 256
    dropout:            float = 0.1
    context_length:     int   = 512
    vocabulary_size:    int   = _VOCAB_SIZE

    # Standard multi-head attention (head_dim = 256 // 8 = 32)
    num_heads:          int   = 8
    transformer_blocks: int   = 8
    qkv_bias:           bool  = False
    use_rope:           bool  = True

    # Standard dense FFN (4x expansion)
    d_ff:               int   = 1024

    # Tie input embedding and output head (saves ~51M params
    # with the 200k o200k_base vocab — required to stay <100M)
    tie_word_embeddings: bool = True

    @property
    def max_seq_len(self) -> int:
        """Alias used by model_parts.py."""
        return self.context_length
