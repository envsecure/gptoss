"""
train.py
────────
Trains the SLM (simple decoder-only transformer) on pre-tokenised TinyStories shards
produced by  prepare_data.py.  Runs on CPU, GPU or TPU.

TPU ready
---------
The script auto-detects PyTorch/XLA.  On a TPU (Colab / Kaggle / TPU VM):
  - `xmp.spawn` starts one worker per TPU core (all cores trained at once)
  - data is sharded across cores (each core sees distinct batches, no overlap)
  - gradient clip + step go through `xm.optimizer.*` (XLA-safe)
  - checkpoints / metrics / plots are written only by worker 0 via `xm.save`
  - bfloat16 is used through `torch.autocast(device_type="xla")` (native on TPU)

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
from torch.utils.data import Dataset, DataLoader, Sampler

from model import Modelcfg, SLM


# ── TPU / PyTorch-XLA detection ───────────────────────────────────────────────

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    _XLA_AVAILABLE = True
except ImportError:                     # torch_xla not installed → CPU / GPU only
    xm = pl = xmp = None
    _XLA_AVAILABLE = False


def is_tpu_runtime() -> bool:
    """True when running on a TPU with torch_xla available."""
    if not _XLA_AVAILABLE:
        return False
    try:
        return xm.device_type().lower() == "tpu"
    except Exception:
        return os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


# ── CLI ───────────────────────────────────────────────────────────────────────

def get_args():
    # Start from config defaults
    cfg = Modelcfg()

    p = argparse.ArgumentParser()

    # ── Training-only args (not in Modelcfg) ──────────────────────────────────
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
    p.add_argument("--graph_dir",      default="artifacts/graph")
    p.add_argument("--log_interval",   type=int,   default=10)
    p.add_argument("--compile",        action="store_true")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--tpu_cores",      type=int,   default=1,
                   help="TPU: spawn this many workers (multi-core). 1 = single-process "
                        "on xla:0 (default, works everywhere). 8 = all cores of a "
                        "TPU v2-8/v3-8/v4-8 TPU VM)")

    # ── Modelcfg fields — defaults pulled live from the config class ──────────
    p.add_argument("--d_model",       type=int, default=cfg.d_model)
    p.add_argument("--n_t_layers",    type=int, default=cfg.n_t_layers)
    p.add_argument("--n_heads",       type=int, default=cfg.n_heads)
    p.add_argument("--context_size",  type=int, default=cfg.context_size)

    args = p.parse_args()
    return args


# ── Metrics tracker ───────────────────────────────────────────────────────────

@dataclass
class MetricsTracker:
    """
    Accumulates every recorded data-point in plain Python lists.

    train_steps / train_losses / train_ppl   — every log_interval
    val_steps   / val_losses   / val_ppl     — every val_interval
    lr_steps    / lrs                        — every log_interval
    tps_steps   / tokens_per_sec             — every log_interval
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
    tokens_seen:    List[int]   = field(default_factory=list)

    def record_train(self, step: int, loss: float, lr: float, tps: float, tokens: int):
        self.train_steps.append(step)
        self.train_losses.append(loss)
        self.train_ppl.append(math.exp(min(loss, 20)))
        self.lr_steps.append(step)
        self.lrs.append(lr)
        self.tps_steps.append(step)
        self.tokens_per_sec.append(tps)
        self.tokens_seen.append(tokens)

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


# ── Plotting ──────────────────────────────────────────────────────────────────

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
    TOK_C = "#E8C84C"
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
    style(ax, f"Perplexity  (step {step})", "Step", "Perplexity")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    save(fig, "perplexity")

    # 3. LR
    fig, ax = plt.subplots(figsize=(9, 3)); fig.patch.set_facecolor(BG)
    if metrics.lr_steps:
        ax.plot(metrics.lr_steps, metrics.lrs, color=LR_C, lw=1.4, label="LR")
    style(ax, f"Learning Rate  (step {step})", "Step", "LR")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1e"))
    save(fig, "lr")

    # 4. Throughput
    fig, ax = plt.subplots(figsize=(9, 3)); fig.patch.set_facecolor(BG)
    if metrics.tps_steps:
        ax.plot(metrics.tps_steps, metrics.tokens_per_sec,
                color=TPS_C, lw=1.2, label="Tokens/sec", alpha=0.85)
    style(ax, f"Throughput  (step {step})", "Step", "Tokens / sec")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else str(int(x))))
    save(fig, "tokens_per_sec")

    # 5. Tokens seen
    fig, ax = plt.subplots(figsize=(9, 3)); fig.patch.set_facecolor(BG)
    if metrics.train_steps and metrics.tokens_seen:
        ax.plot(metrics.train_steps, [t / 1e6 for t in metrics.tokens_seen],
                color=TOK_C, lw=1.4, label="Tokens seen")
    style(ax, f"Total Tokens Seen  (step {step})", "Step", "Tokens (M)")
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


class ShardedSampler(Sampler):
    """Non-overlapping shuffled subset of a dataset per worker (TPU multi-core)."""
    def __init__(self, n: int, rank: int, world: int, seed: int = 42):
        self.n, self.rank, self.world, self.seed = n, rank, world, seed

    def __iter__(self):
        g = torch.Generator(); g.manual_seed(self.seed)
        perm = torch.randperm(self.n, generator=g).tolist()
        return iter(perm[self.rank::self.world])

    def __len__(self):
        return (self.n + self.world - 1) // self.world


def make_loader(ds, args, device, rank: int, world: int, is_tpu: bool, drop_last: bool):
    """Build a DataLoader, sharded by rank when world > 1 (so TPU cores see disjoint data)."""
    kwargs = {"batch_size": args.batch_size, "drop_last": drop_last, "shuffle": True}
    if world > 1:
        kwargs = {"batch_size": args.batch_size, "drop_last": drop_last, "shuffle": False}
        kwargs["sampler"] = ShardedSampler(len(ds), rank, world, seed=args.seed)
    if is_tpu or world > 1:
        kwargs["num_workers"] = 0
    else:
        kwargs["num_workers"] = 4
        kwargs["pin_memory"]  = device.type == "cuda"
        kwargs["persistent_workers"] = True
    loader = DataLoader(ds, **kwargs)
    if world > 1:
        loader = pl.MpDeviceLoader(loader, device)
    return loader


# ── LR schedule ───────────────────────────────────────────────────────────────

def get_lr(step, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (args.lr - args.min_lr)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(step, model, optimizer, val_loss, cfg,
                    ckpt_dir, metrics, graph_dir,
                    is_tpu: bool = False, master: bool = True):
    if is_tpu:
        xm.rendezvous(f"spt_{step}")          # all workers reach the barrier first
    if not master:
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:07d}.pt"
    state = {
        "step": step, "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "val_loss": val_loss,
        "cfg": {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
    }
    if is_tpu:
        xm.save(state, str(path))          # moves tensors to CPU before writing
    else:
        torch.save(state, path)
    shutil.copy(path, ckpt_dir / "latest.pt")
    print(f"  ✓  Checkpoint → {path}")

    metrics.save_json(graph_dir / "metrics.json")
    plot_graphs(metrics, graph_dir, step)


def load_latest(model, optimizer, ckpt_dir, device, graph_dir, is_tpu=False, master=True):
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
    if master:
        print(f"  ✓  Resumed from step {step:,}  (val_loss={val:.4f})")
    return step, val, metrics


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def estimate_val_loss(model, val_loader, val_steps, device, ctx):
    model.eval()
    losses = []
    it = iter(val_loader)
    for _ in range(val_steps):
        try: x, y = next(it)
        except StopIteration: break
        x, y = x.to(device), y.to(device)
        with ctx:
            logits = model(x)
        losses.append(nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ── Distributed training loop (works on CPU / GPU / TPU) ─────────────────────

def train_worker(rank: int, args):
    """
    Backbone of training.  When `--tpu_cores > 1` this is spawned once per core via
    `xmp.spawn`; otherwise it runs once (rank 0) on CPU, GPU or single-core TPU.
    """
    is_tpu = is_tpu_runtime()
    master = rank == 0
    world  = args.tpu_cores if is_tpu else 1

    device = (xm.xla_device() if is_tpu
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    graph_dir = Path(args.graph_dir)
    ckpt_dir  = Path(args.ckpt_dir)

    torch.manual_seed(args.seed)
    ptdtype = {"float32": torch.float32,
               "float16": torch.float16,
               "bfloat16": torch.bfloat16}[args.dtype]
    use_amp = args.dtype != "float32"
    ctx     = torch.autocast(device_type=("xla" if is_tpu else device.type),
                             dtype=ptdtype, enabled=use_amp)
    scaler  = torch.amp.GradScaler("cuda", enabled=(not is_tpu and args.dtype == "float16"))

    cfg = Modelcfg()
    cfg.n_t_layers   = args.n_t_layers
    cfg.n_heads      = args.n_heads
    cfg.d_model      = args.d_model
    cfg.context_size = args.context_size
    model = SLM(cfg).to(device)

    if args.compile:
        if is_tpu:
            print("  torch.compile is not enabled on TPU (use --compile on GPU/CPU)")
        else:
            print("torch.compile …"); model = torch.compile(model)

    # Decay only the 2-D weight matrices; biass/norm stay undecayed.
    decay    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], 
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    data_root = Path(args.data_dir)
    train_ds  = ShardedDataset(data_root / "train", args.context_size)
    val_ds    = ShardedDataset(data_root / "val",   args.context_size)
    train_loader = make_loader(train_ds, args, device, rank, world, is_tpu, drop_last=True)
    val_loader   = make_loader(val_ds,   args, device, rank, world, is_tpu, drop_last=False)

    if master:
        graph_dir.mkdir(parents=True, exist_ok=True)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model  : {n_params:.1f} M params | device={device} "
              f"| workers={world} rank={rank} | dtype={args.dtype}")
        print(f"Graphs : {graph_dir.resolve()}")

    start_step, best_val, metrics = load_latest(
        model, optimizer, ckpt_dir, device, graph_dir, is_tpu, master)

    total_tokens: int = metrics.tokens_seen[-1] if metrics.tokens_seen else 0
    if master:
        print(f"Tokens seen so far: {total_tokens:,}")

    model.train()
    train_iter   = iter(train_loader)
    t0           = time.time()
    running_loss = 0.0

    # Global tokens per optimiser step = per-core × world size (adds all TPU cores)
    per_core_tokens = args.grad_accum * args.batch_size * args.context_size
    tokens_per_step: int = per_core_tokens * world

    for step in range(start_step, args.max_steps):
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
                logits = model(x)
                ce     = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss   = ce / args.grad_accum
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # ── Optimiser step (XLA-safe on TPU) ────────────────────────────────
        if is_tpu:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            xm.optimizer_step(optimizer, barrier=True)   # marks the step + syncs replicas
        else:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        running_loss += accum_loss
        total_tokens += tokens_per_step

        # ── Log + record train metrics (master only) ────────────────────────
        if (step + 1) % args.log_interval == 0 and master:
            elapsed  = time.time() - t0
            tps      = (args.log_interval * tokens_per_step / elapsed)
            avg_loss = running_loss / args.log_interval
            ppl      = math.exp(min(avg_loss, 20))

            print(f"step {step+1:>7,}/{args.max_steps:,}  "
                  f"loss={avg_loss:.4f}  ppl={ppl:.1f}  "
                  f"lr={lr:.2e}  tok/s={tps:,.0f}  "
                  f"tokens={total_tokens/1e6:.2f}M  "
                  f"{elapsed:.1f}s")

            metrics.record_train(step + 1, avg_loss, lr, tps, total_tokens)
            running_loss = 0.0
            t0 = time.time()

        # ── Validate + record val metrics ───────────────────────────────────
        if (step + 1) % args.val_interval == 0:
            val_loss = estimate_val_loss(model, val_loader, args.val_steps, device, ctx)
            if master:
                val_ppl = math.exp(min(val_loss, 20))
                tag     = " ★ new best" if val_loss < best_val else ""
                print(f"  val_loss={val_loss:.4f}  val_ppl={val_ppl:.1f}{tag}")

            if master:
                metrics.record_val(step + 1, val_loss)
            if val_loss < best_val:
                best_val = val_loss

        # ── Checkpoint + graphs (master only) ───────────────────────────────
        if (step + 1) % args.save_interval == 0:
            val_loss = estimate_val_loss(model, val_loader, args.val_steps, device, ctx)
            if master and (not metrics.val_steps or metrics.val_steps[-1] != step + 1):
                metrics.record_val(step + 1, val_loss)
            save_checkpoint(step + 1, model, optimizer, val_loss, cfg,
                            ckpt_dir, metrics, graph_dir, is_tpu, master)

    # ── Final ───────────────────────────────────────────────────────────────
    val_loss = estimate_val_loss(model, val_loader, args.val_steps, device, ctx)
    if master:
        if not metrics.val_steps or metrics.val_steps[-1] != args.max_steps:
            metrics.record_val(args.max_steps, val_loss)
        print(f"\n✓  Done.  val_loss={val_loss:.4f}  "
              f"val_ppl={math.exp(min(val_loss,20)):.1f}  "
              f"total_tokens={total_tokens/1e6:.2f}M")
        print(f"   All graphs in {graph_dir.resolve()}/")
    save_checkpoint(args.max_steps, model, optimizer, val_loss, cfg,
                    ckpt_dir, metrics, graph_dir, is_tpu, master)
    if is_tpu:
        xm.rendezvous("done")


def main():
    args = get_args()
    if is_tpu_runtime() and args.tpu_cores > 1:
        xmp.spawn(train_worker, args=(args,), nprocs=args.tpu_cores)
    else:
        train_worker(0, args)


if __name__ == "__main__":
    main()