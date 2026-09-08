"""
prepare_data.py
───────────────
Streams FineWeb-Edu from HuggingFace, tokenises with tiktoken (o200k_base),
and writes binary shards to  data/train/  and  data/val/.

FineWeb-Edu is ~1.3T tokens total, so the dataset is consumed in
streaming mode and capped via --max_train_tokens / --max_val_tokens.
Docs are randomly assigned to train/val with --val_ratio (seeded).

Each shard is a flat uint32 numpy array saved as  shard_NNNN.npy.
A tiny metadata file  data/{split}/meta.json  records total token count
and number of shards so the DataLoader never has to glob at runtime.

Usage
-----
    python prepare_data.py                                        # 100M train / 10M val from sample-10BT
    python prepare_data.py --max_train_tokens 1000000000          # 1B train tokens
    python prepare_data.py --subset sample-100BT --seed 0         # larger pool
    python prepare_data.py --shard_size 5e7                       # 50M tokens per shard
    python prepare_data.py --val_ratio 0.01                       # 1% of docs -> val

Output layout
-------------
    data/
        train/
            shard_0000.npy
            shard_0001.npy
            ...
            meta.json          {"num_shards": N, "total_tokens": T}
        val/
            shard_0000.npy
            meta.json
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm.auto import tqdm  # notebook widgets in Jupyter, console bar otherwise

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_DATASET          = "HuggingFaceFW/fineweb-edu"
DEFAULT_SUBSET           = "sample-10BT"  # 10B-token pool; use sample-100BT for >1B-token runs
DEFAULT_SHARD_SIZE       = int(1e8)       # 100M tokens per shard (~400 MB uint32)
DEFAULT_MAX_TRAIN_TOKENS = int(1e8)       # 100M train tokens
DEFAULT_MAX_VAL_TOKENS   = int(1e7)       # 10M val tokens
DEFAULT_VAL_RATIO        = 0.01           # ~1% of docs -> val
DEFAULT_SEED             = 42
DATA_ROOT                = Path("data")
EOS_TOKEN                = "<|endoftext|>"


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tokenise FineWeb-Edu into binary shards.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HF dataset repo (default HuggingFaceFW/fineweb-edu)")
    p.add_argument("--subset", default=DEFAULT_SUBSET,
                   help="dataset config: sample-10BT / sample-100BT / sample-350BT / CC-MAIN-YYYY-WW")
    p.add_argument("--split", default="train",
                   help="dataset split (fineweb-edu only has train)")
    p.add_argument("--shard_size", type=float, default=DEFAULT_SHARD_SIZE,
                   help="tokens per shard (default 1e8)")
    p.add_argument("--max_train_tokens", type=float, default=DEFAULT_MAX_TRAIN_TOKENS,
                   help="cap on train tokens (default 1e8)")
    p.add_argument("--max_val_tokens", type=float, default=DEFAULT_MAX_VAL_TOKENS,
                   help="cap on val tokens (default 1e7)")
    p.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO,
                   help="fraction of docs assigned to val (default 0.01)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="seed for train/val doc assignment")
    return p.parse_args()


# ── shard accumulator ─────────────────────────────────────────────────────────
class ShardWriter:
    """Accumulates token ids and flushes fixed-size uint32 shards to disk."""

    def __init__(self, split_dir: Path, shard_size: int):
        self.split_dir = split_dir
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.buf = np.empty(shard_size, dtype=np.uint32)
        self.ptr = 0
        self.shard_idx = 0
        self.total_tokens = 0

    def add(self, ids: np.ndarray) -> None:
        offset = 0
        while offset < len(ids):
            space = self.shard_size - self.ptr
            chunk = ids[offset:offset + space]
            self.buf[self.ptr:self.ptr + len(chunk)] = chunk
            self.ptr += len(chunk)
            offset += len(chunk)
            if self.ptr == self.shard_size:
                self._flush()

    def _flush(self) -> None:
        out = self.split_dir / f"shard_{self.shard_idx:04d}.npy"
        np.save(out, self.buf[:self.ptr])
        self.total_tokens += self.ptr
        self.shard_idx += 1
        self.ptr = 0

    def close(self) -> dict:
        if self.ptr > 0:
            self._flush()
        meta = {"num_shards": self.shard_idx, "total_tokens": self.total_tokens}
        (self.split_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        return meta


def main():
    args = get_args()
    shard_size = int(args.shard_size)
    max_train = int(args.max_train_tokens)
    max_val = int(args.max_val_tokens)
    rng = random.Random(args.seed)

    print(f"Streaming {args.dataset} (config={args.subset}, split={args.split}) …")
    ds = load_dataset(args.dataset, name=args.subset, split=args.split, streaming=True)

    enc = tiktoken.get_encoding("o200k_base")
    eos_id = enc.encode_single_token(EOS_TOKEN)

    train_w = ShardWriter(DATA_ROOT / "train", shard_size)
    val_w = ShardWriter(DATA_ROOT / "val", shard_size)

    # Val docs that arrive after the val cap is hit overflow into train so no
    # tokens are wasted; training stops once both caps are reached.
    # The stream length is unknown, so progress/ETA is tracked against the
    # known token caps (max_train + max_val) instead of doc count.
    tok_total = max_train + max_val
    pbar = tqdm(total=tok_total, desc="tokenising fineweb-edu",
                unit="tok", unit_scale=True)
    n_docs = 0
    for example in ds:
        text = example.get("text", "")
        if not text:
            continue
        n_docs += 1

        ids = enc.encode_ordinary(text)
        ids.append(eos_id)
        arr = np.array(ids, dtype=np.uint32)

        added = 0
        to_val = rng.random() < args.val_ratio
        if to_val and val_w.total_tokens + val_w.ptr < max_val:
            # may straddle the cap — trim to exactly max_val
            remaining = max_val - (val_w.total_tokens + val_w.ptr)
            chunk = arr[:remaining] if len(arr) > remaining else arr
            val_w.add(chunk)
            added = len(chunk)
        elif train_w.total_tokens + train_w.ptr < max_train:
            remaining = max_train - (train_w.total_tokens + train_w.ptr)
            chunk = arr[:remaining] if len(arr) > remaining else arr
            train_w.add(chunk)
            added = len(chunk)
        elif val_w.total_tokens + val_w.ptr < max_val:
            # train full but val still short: overflow val-assigned docs here
            remaining = max_val - (val_w.total_tokens + val_w.ptr)
            chunk = arr[:remaining] if len(arr) > remaining else arr
            val_w.add(chunk)
            added = len(chunk)

        if added:
            pbar.update(added)

        if n_docs % 200 == 0:
            pbar.set_postfix({
                "docs": f"{n_docs:,}",
                "train_M": f"{(train_w.total_tokens + train_w.ptr) / 1e6:.1f}",
                "val_M": f"{(val_w.total_tokens + val_w.ptr) / 1e6:.2f}",
            })

        if (train_w.total_tokens + train_w.ptr >= max_train
                and val_w.total_tokens + val_w.ptr >= max_val):
            break

    pbar.close()
    train_meta = train_w.close()
    val_meta = val_w.close()

    print(f"\n✓  Done! ({n_docs:,} docs streamed)")
    print(f"   Train : {train_meta['total_tokens']:,} tokens  |  {train_meta['num_shards']} shards")
    print(f"   Val   : {val_meta['total_tokens']:,} tokens  |  {val_meta['num_shards']} shards")
    print(f"   Saved to  {DATA_ROOT.resolve()}/")


if __name__ == "__main__":
    main()
