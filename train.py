"""
train.py
────────
Trains the LLM (MHA + RoPE, dense FFN) on pre-tokenised FineWeb-Edu shards
produced by  prepare_data.py.

Quick start
-----------
    python prepare_data.py          # once
    python train.py
    python train.py --batch_size 8 --max_steps 20000 --compile

Graphs
------
Every time a checkpoint is saved, five PNG plots are written to:

    artifacts/graphs/
        step_XXXXXXX_loss.png           train + val loss curves
        step_XXXXXXX_perplexity.png     train + val perplexity curves
        step_XXXXXXX_lr.png             learning-rate schedule
        step_XXXXXXX_tokens_per_sec.png throughput over time
        step_XXXXXXX_tokens_seen.png    cumulative tokens seen
        latest_loss.png                 always overwritten (watch live)
        latest_perplexity.png
        latest_tokens_seen.png

Raw numbers are persisted as:
    artifacts/graphs/metrics.json
so you can re-plot without re-training.

Checkpoint layout
-----------------
    checkpoints/
        step_0001000.pt  ...  latest.pt
"""

import argparse, json, math, os, shutil, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm  # notebook widgets in Jupyter, console bar otherwise

# Optional TPU support — guarded so CUDA/CPU runs are unaffected when
# torch_xla is not installed.
try:
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
    USE_XLA = True
except ImportError:
    xm = None
    MpDeviceLoader = None
    USE_XLA = False

from config_model import ModelConfig
from model import LLM


def resolve_device(requested: str) -> torch.device:
    """Pick cpu/cuda/xla. 'auto' prefers CUDA, then TPU (if usable), then CPU."""
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if requested == "xla" or requested == "auto":
        if USE_XLA:
            try:
                return xm.xla_device()
            except Exception as e:
                if requested == "xla":
                    raise RuntimeError(
                        "TPU requested but no XLA device found. "
                        "On a TPU VM, ensure torch_xla matches your torch version.") from e
                print(f"  (no XLA device: {e} — falling back)")
    if requested not in ("auto", "cpu", "cuda"):
        return torch.device(requested)  # e.g. "mps"
    return torch.device("cpu")

def get_args():
    # Start from config defaults
    cfg = ModelConfig()

    p = argparse.ArgumentParser()

    # ── Training-only args (not in ModelConfig) ───────────────────────────────
    p.add_argument("--data_dir",       default="data")
    p.add_argument("--batch_size",     type=int,   default=2)
    p.add_argument("--grad_accum",     type=int,   default=8)
    p.add_argument("--max_steps",      type=int,   default=50_000)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--min_lr",         type=float, default=3e-5)
    p.add_argument("--warmup_steps",   type=int,   default=1_000)
    p.add_argument("--grad_clip",      type=float, default=1.0)
    p.add_argument("--weight_decay",   type=float, default=0.1)
    p.add_argument("--val_interval",   type=int,   default=100)
    p.add_argument("--val_steps",      type=int,   default=50)
    p.add_argument("--save_interval",  type=int,   default=1000)
    p.add_argument("--ckpt_dir",       default="checkpoints")
    p.add_argument("--graph_dir",      default="artifacts/graphs")
    p.add_argument("--log_interval",   type=int,   default=10)
    p.add_argument("--compile",        action="store_true")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--device",         default="auto",
                   help="auto (CUDA > TPU > CPU) / cpu / cuda / xla. "
                        "Needs torch_xla installed for xla.")

    # ── ModelConfig fields — defaults pulled live from the dataclass ──────────
    p.add_argument("--d_model",            type=int,   default=cfg.d_model)
    p.add_argument("--context_length",     type=int,   default=cfg.context_length)
    p.add_argument("--transformer_blocks", type=int,   default=cfg.transformer_blocks)
    p.add_argument("--num_heads",          type=int,   default=cfg.num_heads)
    p.add_argument("--d_ff",               type=int,   default=cfg.d_ff)
    p.add_argument("--dropout",            type=float, default=cfg.dropout)

    args = p.parse_args()
    return args

@dataclass
class MetricsTracker:
    """
    Accumulates every recorded data-point in plain Python lists.

    train_steps / train_losses / train_ppl   — every log_interval
    val_steps   / val_losses   / val_ppl     — every val_interval
    lr_steps    / lrs                        — every log_interval
    tps_steps   / tokens_per_sec            — every log_interval
    tokens_seen                              — cumulative tokens, every log_interval
    """
    train_steps:    List[int]   = field(default_factory=list)
    train_losses:   List[float] = field(default_factory=list)
    train_ppl:      List[float] = field(default_factory=list)
    val_steps:      List[int]   = field(default_factory=list)
    val_losses:     List[float] = field(default_factory=list)
    val_ppl:        List[float] = field(default_factory=list)
    lr_steps:       List[int]   = field(default_factory=list)
    lrs:            List[float] = field(default_factory=list)
    tps_steps:      List[int]   = field(default_factory=list)
    tokens_per_sec: List[float] = field(default_factory=list)
    tokens_seen:    List[int]   = field(default_factory=list)   # ← NEW

    def record_train(self, step: int, loss: float, lr: float, tps: float, tokens: int):
        self.train_steps.append(step)
        self.train_losses.append(loss)
        self.train_ppl.append(math.exp(min(loss, 20)))
        self.lr_steps.append(step)
        self.lrs.append(lr)
        self.tps_steps.append(step)
        self.tokens_per_sec.append(tps)
        self.tokens_seen.append(tokens)                         # ← NEW

    def record_val(self, step: int, loss: float):
        self.val_steps.append(step)
        self.val_losses.append(loss)
        self.val_ppl.append(math.exp(min(loss, 20)))

    def save_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load_json(cls, path: Path) -> "MetricsTracker":
        data = json.loads(path.read_text())
        obj  = cls()
        for k, v in data.items():
            setattr(obj, k, v)
        return obj

def plot_graphs(metrics: MetricsTracker, graph_dir: Path, step: int):
    """
    Renders 5 dark-themed PNG charts and saves them as:
        step_XXXXXXX_<name>.png   (permanent, one per checkpoint)
        latest_<name>.png         (always overwritten, easy to watch live)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    graph_dir.mkdir(parents=True, exist_ok=True)

    TRAIN = "#4C9BE8"
    VAL   = "#E8744C"
    LR_C  = "#6BBF59"
    TPS_C = "#B05CE8"
    TOK_C = "#E8C84C"   # ← NEW colour for tokens-seen chart
    BG    = "#0F1117"
    GRID  = "#2A2D3A"
    TEXT  = "#E0E0E0"

    def style(ax, title, xlabel, ylabel):
        ax.set_facecolor(BG)
        ax.set_title(title,   color=TEXT, fontsize=13, pad=10)
        ax.set_xlabel(xlabel, color=TEXT, fontsize=10)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=10)
        ax.tick_params(colors=TEXT)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID)
        ax.grid(True, color=GRID, linewidth=0.6, linestyle="--")
        ax.legend(facecolor="#1C1F2E", edgecolor=GRID,
                  labelcolor=TEXT, fontsize=9)

    def save(fig, stem):
        numbered = graph_dir / f"step_{step:07d}_{stem}.png"
        latest   = graph_dir / f"latest_{stem}.png"
        fig.savefig(numbered, dpi=130, bbox_inches="tight",
                    facecolor=BG, edgecolor="none")
        shutil.copy(numbered, latest)
        plt.close(fig)

    # 1. Loss
    fig, ax = plt.subplots(figsize=(9, 4)); fig.patch.set_facecolor(BG)
    if metrics.train_steps:
        ax.plot(metrics.train_steps, metrics.train_losses,
                color=TRAIN, lw=1.2, label="Train loss", alpha=0.85)
    if metrics.val_steps:
        ax.plot(metrics.val_steps, metrics.val_losses,
                color=VAL, lw=2.0, marker="o", ms=4, label="Val loss", zorder=5)
    style(ax, f"Loss  (step {step:,})", "Step", "Loss")
    save(fig, "loss")

    # 2. Perplexity
    fig, ax = plt.subplots(figsize=(9, 4)); fig.patch.set_facecolor(BG)
    if metrics.train_steps:
        ax.plot(metrics.train_steps, metrics.train_ppl,
                color=TRAIN, lw=1.2, label="Train PPL", alpha=0.85)
    if metrics.val_steps:
        ax.plot(metrics.val_steps, metrics.val_ppl,
                color=VAL, lw=2.0, marker="o", ms=4, label="Val PPL", zorder=5)
    style(ax, f"Perplexity  (step {step:,})", "Step", "Perplexity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    save(fig, "perplexity")

    # 3. LR
    fig, ax = plt.subplots(figsize=(9, 3)); fig.patch.set_facecolor(BG)
    if metrics.lr_steps:
        ax.plot(metrics.lr_steps, metrics.lrs, color=LR_C, lw=1.4, label="LR")
    style(ax, f"Learning Rate  (step {step:,})", "Step", "LR")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    save(fig, "lr")

    # 4. Throughput
    fig, ax = plt.subplots(figsize=(9, 3)); fig.patch.set_facecolor(BG)
    if metrics.tps_steps:
        ax.plot(metrics.tps_steps, metrics.tokens_per_sec,
                color=TPS_C, lw=1.2, label="Tokens/sec", alpha=0.85)
    style(ax, f"Throughput  (step {step:,})", "Step", "Tokens / sec")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))
    save(fig, "tokens_per_sec")

    # 5. Tokens seen  ← NEW
    fig, ax = plt.subplots(figsize=(9, 3)); fig.patch.set_facecolor(BG)
    if metrics.train_steps and metrics.tokens_seen:
        ax.plot(metrics.train_steps, [t / 1e6 for t in metrics.tokens_seen],
                color=TOK_C, lw=1.4, label="Tokens seen")
    style(ax, f"Total Tokens Seen  (step {step:,})", "Step", "Tokens (M)")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    save(fig, "tokens_seen")

    print(f"  📊 Graphs → {graph_dir}/step_{step:07d}_*.png  +  latest_*.png")

# ── Dataset ───────────────────────────────────────────────────────────────────
class ShardedDataset(Dataset):
    def __init__(self, split_dir: Path, context_length: int):
        if not (split_dir / "meta.json").exists():
            raise FileNotFoundError(
                f"{split_dir}/meta.json missing — run prepare_data.py first.")
        self.context_length = context_length
        self._index: List = []
        for sp in sorted(split_dir.glob("shard_*.npy")):
            arr = np.load(sp, mmap_mode="r")
            for start in range(0, len(arr) - context_length, context_length):
                self._index.append((sp, start))
        if not self._index:
            raise FileNotFoundError(f"No windows found in {split_dir}")

    def __len__(self): return len(self._index)

    def __getitem__(self, idx):
        sp, start = self._index[idx]
        arr   = np.load(sp, mmap_mode="r")
        chunk = arr[start : start + self.context_length + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])

# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)

# ── Checkpoint helpers ────────────────────────────────────────────────────────
def save_checkpoint(step, model, optimizer, val_loss, cfg,
                    ckpt_dir, metrics, graph_dir, device):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    saver = xm.save if device.type == "xla" else torch.save
    saver({
        "step": step, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "val_loss": val_loss,
        "cfg": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
    }, path)
    shutil.copy(path, ckpt_dir / "latest.pt")
    print(f"  ✓  Checkpoint → {path}")

    # Persist metrics JSON
    metrics.save_json(graph_dir / "metrics.json")

    # Save all five graphs
    plot_graphs(metrics, graph_dir, step)


def load_latest(model, optimizer, ckpt_dir, device, graph_dir):
    metrics_json = graph_dir / "metrics.json"
    metrics = MetricsTracker.load_json(metrics_json) if metrics_json.exists() \
              else MetricsTracker()
    latest = ckpt_dir / "latest.pt"
    if not latest.exists():
        return 0, float("inf"), metrics
    state = torch.load(latest, map_location=device)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    step, val = state["step"], state.get("val_loss", float("inf"))
    print(f"  ✓  Resumed from step {step:,}  (val_loss={val:.4f})")
    return step, val, metrics

# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def estimate_val_loss(model, val_loader, val_steps, device, ctx):
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in tqdm(range(val_steps), desc="validating", unit="batch", leave=False):
        try: x, y = next(it)
        except StopIteration: break
        x, y = x.to(device), y.to(device)
        with ctx:
            logits = model(x, use_cache=False)
        losses.append(nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")

def main():
    args      = get_args()
    device    = resolve_device(args.device)
    is_xla    = device.type == "xla"
    graph_dir = Path(args.graph_dir)
    ckpt_dir  = Path(args.ckpt_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    ptdtype = {"float32": torch.float32,
               "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]
    # AMP where supported (CUDA + XLA/bfloat16). GradScaler is CUDA-float16-only;
    # on XLA / CPU / float32 the scaler stays None and plain backward is used.
    use_amp = args.dtype != "float32" and device.type in ("cuda", "xla")
    ctx     = (torch.autocast(device_type=device.type, dtype=ptdtype) if use_amp
               else torch.amp.autocast(device_type="cpu", enabled=False))
    scaler  = (torch.amp.GradScaler(enabled=True)
               if (args.dtype == "float16" and device.type == "cuda") else None)

    cfg = ModelConfig(
        context_length=args.context_length, d_model=args.d_model,
        transformer_blocks=args.transformer_blocks, num_heads=args.num_heads,
        d_ff=args.d_ff, dropout=args.dropout,
    )
    model = LLM(cfg).to(device)
    if args.compile:
        if is_xla:
            print("  (torch.compile skipped — XLA compiles graphs natively)")
        else:
            print("torch.compile …"); model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model  : {n_params:.1f} M params | device={device} | dtype={args.dtype}")
    print(f"Graphs : {graph_dir.resolve()}")

    decay    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], 
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    data_root    = Path(args.data_dir)
    train_ds     = ShardedDataset(data_root / "train", args.context_length)
    val_ds       = ShardedDataset(data_root / "val",   args.context_length)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    if is_xla:
        # Streams host batches to the TPU; required for good TPU utilisation.
        train_loader = MpDeviceLoader(train_loader, device)
        val_loader   = MpDeviceLoader(val_loader, device)
    print(f"Train: {len(train_ds):,} windows  |  Val: {len(val_ds):,} windows")

    start_step, best_val, metrics = load_latest(model, optimizer, ckpt_dir, device, graph_dir)

    # Restore total tokens seen from the last checkpoint (handles resume correctly)
    total_tokens: int = metrics.tokens_seen[-1] if metrics.tokens_seen else 0  # ← NEW
    print(f"Tokens seen so far: {total_tokens:,}")                             # ← NEW

    model.train()
    train_iter   = iter(train_loader)
    t0           = time.time()
    running_loss = 0.0

    # Tokens processed per optimiser step (grad_accum micro-batches × batch × seq)
    tokens_per_step: int = args.grad_accum * args.batch_size * args.context_length  # ← NEW

    pbar = tqdm(range(start_step, args.max_steps),
                initial=start_step, total=args.max_steps,
                desc="training", unit="step")
    for step in pbar:

        lr = get_lr(step, args)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with ctx:
                logits = model(x, use_cache=False)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)) / args.grad_accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            accum_loss += loss.item()

        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        if is_xla:
            # TPU step: xm.optimizer_step replaces scaler.step/optimizer.step,
            # mark_step flushes the lazy XLA graph for execution.
            xm.optimizer_step(optimizer)
            xm.mark_step()
        elif scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        running_loss += accum_loss

        total_tokens += tokens_per_step   # ← NEW: increment after every optimiser step
        pbar.set_postfix({"loss": f"{accum_loss:.4f}", "lr": f"{lr:.1e}",
                          "tokens": f"{total_tokens/1e6:.1f}M"})

        # ── Log + record train metrics ────────────────────────────────────────
        if (step + 1) % args.log_interval == 0:
            elapsed  = time.time() - t0
            tps      = (args.log_interval * tokens_per_step / elapsed)  # ← uses same constant
            avg_loss = running_loss / args.log_interval
            ppl      = math.exp(min(avg_loss, 20))

            pbar.write(f"step {step+1:>7,}/{args.max_steps:,}  "
                       f"loss={avg_loss:.4f}  ppl={ppl:.1f}  "
                       f"lr={lr:.2e}  tok/s={tps:,.0f}  "
                       f"tokens={total_tokens/1e6:.2f}M  "   # ← NEW in log line
                       f"{elapsed:.1f}s")

            metrics.record_train(step + 1, avg_loss, lr, tps, total_tokens)  # ← passes tokens
            running_loss = 0.0
            t0 = time.time()

        # ── Validate + record val metrics ─────────────────────────────────────
        if (step + 1) % args.val_interval == 0:
            val_loss = estimate_val_loss(model, val_loader, args.val_steps, device, ctx)
            val_ppl  = math.exp(min(val_loss, 20))
            tag      = " ★ new best" if val_loss < best_val else ""
            pbar.write(f"  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}{tag}")

            metrics.record_val(step + 1, val_loss)

            if val_loss < best_val:
                best_val = val_loss

        # ── Checkpoint + save graphs ──────────────────────────────────────────
        if (step + 1) % args.save_interval == 0:
            val_loss = estimate_val_loss(model, val_loader, args.val_steps, device, ctx)
            # don't double-record if val was already taken this step
            if not metrics.val_steps or metrics.val_steps[-1] != step + 1:
                metrics.record_val(step + 1, val_loss)

            save_checkpoint(step + 1, model, optimizer, val_loss,
                            cfg, ckpt_dir, metrics, graph_dir, device)

    # ── Final ─────────────────────────────────────────────────────────────────
    val_loss = estimate_val_loss(model, val_loader, args.val_steps, device, ctx)
    if not metrics.val_steps or metrics.val_steps[-1] != args.max_steps:
        metrics.record_val(args.max_steps, val_loss)
    save_checkpoint(args.max_steps, model, optimizer, val_loss,
                    cfg, ckpt_dir, metrics, graph_dir, device)
    print(f"\n✓  Done.  val_loss={val_loss:.4f}  "
          f"val_ppl={math.exp(min(val_loss,20)):.1f}  "
          f"total_tokens={total_tokens/1e6:.2f}M")   # ← NEW in final summary
    print(f"   All graphs in {graph_dir.resolve()}/")


if __name__ == "__main__":
    main()
