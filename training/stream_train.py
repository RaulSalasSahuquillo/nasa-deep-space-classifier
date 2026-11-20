#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==============================================================================
# NASA Deep Space Image Classification & Neural Training Pipeline
#
# File:         training/stream_train.py
# Description:  Fine-tunes the ResNet-34 AstroClass model on up to 20,000 images
#               per class that are STREAMED from remote archives (SDSS, DECaLS,
#               NASA Images API, ESA/Hubble, Wikimedia Commons). Images live only
#               in RAM as 256x256 JPEG bytes: nothing is written to disk.
#
# Usage:        python training/stream_train.py --target 20000
# ==============================================================================

from __future__ import annotations

import argparse
import io
import json
import math
import os
import queue
import random
import signal
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchvision.models import resnet34

sys.path.insert(0, str(Path(__file__).resolve().parent))
from streaming import CLASS_NAMES  # noqa: E402
from streaming.sources import build_recipe  # noqa: E402
from streaming.store import ImageStore  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parent
DEFAULT_INIT = TRAINING_DIR / "trained_net.pth"
DEFAULT_OUT = TRAINING_DIR / "trained_net_stream.pth"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Relative sampling weight of each source group inside a class (also scaled by sqrt(group size)).
GROUP_WEIGHTS = {"nasa": 1.6, "esa": 1.6, "commons": 1.4, "decals": 1.0, "sdss": 0.9, "sdss_field": 0.7, "sdss_faint": 0.9}


# utils
class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def build_model(init: Path | None, device) -> nn.Module:
    net = resnet34(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))
    if init is not None and init.exists():
        sd = torch.load(init, map_location="cpu", weights_only=True)
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        net.load_state_dict(sd)
    return net.to(device).to(memory_format=torch.channels_last)


def save_atomic(state, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def decode(b: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))


def to_tensor_batch(arrs) -> torch.Tensor:
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()


# GPU augment
def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    k = 5
    ax = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(ax**2) / (2 * sigma**2))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def augment(xu8: torch.Tensor, out: int = 224) -> torch.Tensor:
    """Astro-aware augmentation on GPU: free rotation, flips, zoom/crop, stretch (gamma), noise, blur."""
    dev = xu8.device
    B = xu8.shape[0]
    x = xu8.float() / 255.0
    # geometry
    ang = torch.randint(0, 4, (B,), device=dev).float() * (math.pi / 2) + (torch.rand(B, device=dev) - 0.5) * 2 * math.radians(20)
    zoom = 1.0 + torch.rand(B, device=dev) * 0.55
    asp = torch.exp((torch.rand(B, device=dev) - 0.5) * 0.2)
    flip = (torch.rand(B, device=dev) < 0.5).float() * 2 - 1
    sx, sy = flip / zoom * asp, 1.0 / zoom / asp
    tx = (torch.rand(B, device=dev) - 0.5) * 2 * (1 - 1 / zoom) * 0.9
    ty = (torch.rand(B, device=dev) - 0.5) * 2 * (1 - 1 / zoom) * 0.9
    c, s = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([c * sx, -s * sy, tx], 1), torch.stack([s * sx, c * sy, ty], 1)], 1)
    grid = F.affine_grid(theta, (B, 3, out, out), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    # --- photometry (sources use very different stretches)
    gamma = torch.exp((torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.6)
    x = x.clamp_min(1e-4) ** gamma
    x = x * (1 + (torch.rand(B, 1, 1, 1, device=dev) - 0.5) * 0.5) * (1 + (torch.rand(B, 3, 1, 1, device=dev) - 0.5) * 0.12)
    m = x.mean((1, 2, 3), keepdim=True)
    x = (x - m) * (0.8 + torch.rand(B, 1, 1, 1, device=dev) * 0.45) + m
    # --- noise / blur on random subsets
    nm = (torch.rand(B, 1, 1, 1, device=dev) < 0.25).float()
    x = x + torch.randn_like(x) * 0.02 * nm
    bm = torch.rand(B, device=dev) < 0.2
    if bm.any():
        x[bm] = _gauss_blur(x[bm], float(random.uniform(0.7, 1.6)))
    x = x.clamp(0, 1)
    return ((x - MEAN.to(dev)) / STD.to(dev)).contiguous(memory_format=torch.channels_last)


def prep_eval(xu8: torch.Tensor, out: int = 224) -> torch.Tensor:
    x = xu8.float() / 255.0
    x = F.interpolate(x, size=(out, out), mode="bilinear", antialias=True, align_corners=False)
    return ((x - MEAN.to(x.device)) / STD.to(x.device)).contiguous(memory_format=torch.channels_last)


# batches
class BatchFeeder(threading.Thread):
    """Background thread: sample from the live store, decode with a thread pool, queue CPU tensors."""

    def __init__(self, store: ImageStore, batch: int, depth: int = 6, workers: int = 10, seed: int = 0):
        super().__init__(daemon=True)
        self.store, self.batch = store, batch
        self.q: queue.Queue = queue.Queue(maxsize=depth)
        self.pool = ThreadPoolExecutor(workers)
        self.rng = random.Random(seed)
        self.halt = threading.Event()
        self.error = None

    def run(self):
        while not self.halt.is_set():
            try:
                bs, ys = self.store.sample_batch(self.batch, GROUP_WEIGHTS, self.rng)
                if not bs:
                    time.sleep(0.5)
                    continue
                x = to_tensor_batch(list(self.pool.map(decode, bs))).pin_memory()
                y = torch.tensor(ys, dtype=torch.long)
            except Exception as e:  # noqa: BLE001  - a bad sample must not kill the run
                self.error = repr(e)
                time.sleep(0.2)
                continue
            while not self.halt.is_set():
                try:
                    self.q.put((x, y), timeout=0.5)
                    break
                except queue.Full:
                    continue

    def get(self, interrupted: threading.Event, timeout_s: float = 300.0):
        """Blocking get that notices SIGINT and a starved/dead feeder instead of hanging forever."""
        waited = 0.0
        while not interrupted.is_set():
            try:
                return self.q.get(timeout=2.0)
            except queue.Empty:
                waited += 2.0
                if not self.is_alive() or waited > timeout_s:
                    raise RuntimeError(f"batch feeder stalled (alive={self.is_alive()}, last error={self.error})")
        return None


# evaluation
@torch.no_grad()
def predict(model: nn.Module, items, device, tta: bool = False, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    pool = ThreadPoolExecutor(10)
    for i in range(0, len(items), bs):
        chunk = items[i : i + bs]
        x = to_tensor_batch(list(pool.map(decode, [c[0] for c in chunk]))).to(device, non_blocking=True)
        x = prep_eval(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if tta:
                p = 0
                for k in range(4):
                    xr = torch.rot90(x, k, (2, 3))
                    p = p + F.softmax(model(xr).float(), 1) + F.softmax(model(torch.flip(xr, (3,))).float(), 1)
                p = p / 8
            else:
                p = F.softmax(model(x).float(), 1)
        out.append(p.cpu().numpy())
    pool.shutdown()
    return np.concatenate(out) if out else np.zeros((0, len(CLASS_NAMES)))


def metrics_from(probs: np.ndarray, items) -> dict:
    from sklearn.metrics import confusion_matrix, f1_score

    y = np.array([it[1] for it in items])
    pred = probs.argmax(1)
    n = len(CLASS_NAMES)
    cm = confusion_matrix(y, pred, labels=list(range(n)))
    per_class = {}
    for c in range(n):
        tp = cm[c, c]
        per_class[CLASS_NAMES[c]] = {
            "support": int(cm[c].sum()),
            "recall": float(tp / max(1, cm[c].sum())),
            "precision": float(tp / max(1, cm[:, c].sum())),
        }
    by_group = defaultdict(lambda: [0, 0])
    for p, it in zip(pred, items):
        by_group[f"{CLASS_NAMES[it[1]]}/{it[2]}"][0] += int(p == it[1])
        by_group[f"{CLASS_NAMES[it[1]]}/{it[2]}"][1] += 1
    curated = [i for i, it in enumerate(items) if it[2] in ("nasa", "esa", "commons")]
    cur_acc = float(np.mean([pred[i] == y[i] for i in curated])) if curated else None
    grp_rec = [a / t for (a, t) in by_group.values() if t >= 5]
    nn = len(y)
    p = float((pred == y).mean())
    z = 1.96
    wil = (p + z * z / (2 * nn) - z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn))) / (1 + z * z / nn), (
        p + z * z / (2 * nn) + z * math.sqrt(p * (1 - p) / nn + z * z / (4 * nn * nn))) / (1 + z * z / nn)
    return {
        "n": int(len(y)),
        "accuracy_ci95": [float(wil[0]), float(wil[1])],
        "curated_accuracy": cur_acc,
        "curated_n": len(curated),
        "source_balanced_recall": float(np.mean(grp_rec)) if grp_rec else None,
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean([v["recall"] for v in per_class.values() if v["support"] > 0])),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(n)), zero_division=0)),
        "per_class": per_class,
        "per_group_acc": {k: {"acc": a / t, "n": t} for k, (a, t) in sorted(by_group.items())},
        "confusion": cm.tolist(),
    }


# plots
def save_plots(run_dir: Path, history: list, final: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [h["step"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6), dpi=130)
    ax[0].plot(steps, [h["loss"] for h in history], color="#1f77b4")
    ax[0].set_title("Training loss (smoothed)")
    ax[0].set_xlabel("step")
    ax[0].grid(alpha=0.3)
    ev = [h for h in history if "val_acc" in h]
    ax[1].plot([h["step"] for h in ev], [100 * h["val_acc"] for h in ev], "o-", label="val accuracy", color="#2ca02c")
    ax[1].plot([h["step"] for h in ev], [100 * h["val_bacc"] for h in ev], "s-", label="val balanced acc.", color="#d62728")
    ax[1].set_title("Held-out validation (streamed)")
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("%")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    ax[2].plot(steps, [h["lr"] for h in history], color="#9467bd")
    ax[2].set_title("Learning rate")
    ax[2].set_xlabel("step")
    ax[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "learning_curves_stream.png")
    plt.close()

    cm = np.array(final["confusion"])
    cmn = cm / np.maximum(1, cm.sum(1, keepdims=True))
    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}\n{100 * cmn[i, j]:.1f}%", ha="center", va="center", color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — streamed held-out set (EMA + TTA)")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix_stream.png")
    plt.close()


# main
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", type=int, default=20000, help="images per class (upper bound per source recipe)")
    p.add_argument("--init", type=Path, default=DEFAULT_INIT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--wd", type=float, default=0.02)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--decay-steps", type=int, default=6000, help="cosine decay length once streaming has finished")
    p.add_argument("--min-start", type=int, default=600, help="train images per class before the first step")
    p.add_argument("--start-timeout-min", type=float, default=12)
    p.add_argument("--max-fetch-min", type=float, default=150)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--val-fraction", type=float, default=0.03)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--only", nargs="*", help="restrict to these source names (debug)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = TRAINING_DIR / "stream_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log = Logger(run_dir / "log.txt")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    log(f"run dir: {run_dir}")
    log(f"args: {vars(args)}")

    # streaming
    store = ImageStore(val_fraction=args.val_fraction, seed=args.seed)
    stop_fetch = threading.Event()
    sources = [s for s in build_recipe(args.target) if not args.only or s.name in args.only]
    fetch_threads = []
    for s in sources:
        t = threading.Thread(target=s.run, args=(store, stop_fetch), name=s.name, daemon=True)
        t.start()
        fetch_threads.append(t)
    t_fetch0 = time.time()
    log(f"streaming started: {len(sources)} sources -> RAM only (no disk writes)")

    def fetch_done() -> bool:
        return all(not t.is_alive() for t in fetch_threads)

    def status() -> str:
        tot = store.class_totals()
        return (
            " ".join(f"{k}={v}" for k, v in tot.items())
            + f" | ok={store.stats['ok']} dupImg={store.stats['dup_img']} conflict={store.stats['conflict']} bad={store.stats['bad']}"
            + f" | RAM={store.memory_mb():.0f}MB"
        )

    # wait for a minimal pool in every class
    t0 = time.time()
    while True:
        tc = store.train_counts()
        ready = all(tc.get(c, 0) >= args.min_start for c in range(len(CLASS_NAMES)))
        if ready or fetch_done() or (time.time() - t0) > args.start_timeout_min * 60:
            break
        log("warming up: " + status())
        time.sleep(20)
    log("start training with: " + status())

    # model
    model = build_model(args.init, device)
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9985), use_buffers=True)
    head, body = list(model.fc.parameters()), [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 2}], weight_decay=args.wd)
    base_lrs = [g["lr"] for g in opt.param_groups]
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    feeder = BatchFeeder(store, args.batch, seed=args.seed)
    feeder.start()

    interrupted = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: interrupted.set())
    signal.signal(signal.SIGTERM, lambda *_: interrupted.set())

    history, step, decay_start, ema_loss, ema_acc = [], 0, None, None, None
    seen = 0
    t_train0 = time.time()
    last_status = 0.0
    best = {"score": -1.0, "step": 0}

    def lr_scale() -> float:
        w = min(1.0, (step + 1) / args.warmup)
        if decay_start is None:
            return w
        prog = min(1.0, (step - decay_start) / args.decay_steps)
        return w * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))

    def evaluate(tag: str, tta=False) -> dict:
        items = store.val_items()
        probs = predict(ema.module, items, device, tta=tta)
        m = metrics_from(probs, items)
        log(f"EVAL[{tag}] step={step} n={m['n']} acc={100 * m['accuracy']:.2f}% bacc={100 * m['balanced_accuracy']:.2f}% f1={100 * m['macro_f1']:.2f}%"
            + " | " + " ".join(f"{k}:{100 * v['recall']:.1f}" for k, v in m["per_class"].items()))
        return m

    log("training loop started")
    while not interrupted.is_set():
        if decay_start is None and (fetch_done() or (time.time() - t_fetch0) > args.max_fetch_min * 60):
            stop_fetch.set()
            decay_start = step
            log(f"streaming finished -> cosine decay for {args.decay_steps} steps | " + status())
            log("source summary: " + json.dumps({s.name: {"attempted": s.attempted, "error": s.error} for s in sources}))
        if decay_start is not None and step - decay_start >= args.decay_steps:
            break

        got = feeder.get(interrupted)
        if got is None:
            break
        x, y = got
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        s = lr_scale()
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * s
        model.train()
        xa = augment(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xa)
            loss = crit(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        ema.update_parameters(model)
        step += 1
        seen += len(y)
        l = float(loss.detach())
        a = float((logits.argmax(1) == y).float().mean())
        ema_loss = l if ema_loss is None else 0.98 * ema_loss + 0.02 * l
        ema_acc = a if ema_acc is None else 0.98 * ema_acc + 0.02 * a

        if step % 50 == 0:
            history.append({"step": step, "loss": ema_loss, "train_acc": ema_acc, "lr": opt.param_groups[0]["lr"]})
        if step % 100 == 0 and time.time() - last_status > 25:
            last_status = time.time()
            log(f"step {step:6d} loss={ema_loss:.4f} acc={100 * ema_acc:.1f}% lr={opt.param_groups[0]['lr']:.2e} "
                f"{seen / (time.time() - t_train0):.0f} img/s {'[streaming]' if decay_start is None else '[decay]'} | {status()}")
        if step % args.eval_every == 0:
            m = evaluate("periodic")
            history.append({"step": step, "loss": ema_loss, "train_acc": ema_acc, "lr": opt.param_groups[0]["lr"],
                            "val_acc": m["accuracy"], "val_bacc": m["balanced_accuracy"], "val_n": m["n"]})
            save_atomic(ema.module.state_dict(), run_dir / "last_ema.pth")  # crash insurance
            if decay_start is not None and m["balanced_accuracy"] > best["score"]:
                best = {"score": m["balanced_accuracy"], "step": step}
                save_atomic(ema.module.state_dict(), run_dir / "best_ema.pth")

    feeder.halt.set()
    stop_fetch.set()
    save_atomic(ema.module.state_dict(), run_dir / "final_ema.pth")
    save_atomic(ema.module.state_dict(), args.out)
    log(f"weights saved -> {args.out} (evaluation follows)")

    # final eval
    log("final evaluation (EMA weights, 8-fold TTA)")
    items = store.val_items()
    final_probs = predict(ema.module, items, device, tta=True)
    final = metrics_from(final_probs, items)
    plain = metrics_from(predict(ema.module, items, device, tta=False), items)
    base_model = build_model(args.init, device)
    base = metrics_from(predict(base_model, items, device, tta=False), items)
    fmt = lambda v: "n/a" if v is None else f"{100 * v:.2f}%"  # noqa: E731
    log(f"FINAL new model : acc={fmt(final['accuracy'])} (95% CI {fmt(final['accuracy_ci95'][0])}-{fmt(final['accuracy_ci95'][1])}) bacc={fmt(final['balanced_accuracy'])} "
        f"f1={fmt(final['macro_f1'])} | curated-only acc={fmt(final['curated_accuracy'])} (n={final['curated_n']}) | source-balanced recall={fmt(final['source_balanced_recall'])} | no-TTA acc {fmt(plain['accuracy'])}")
    log(f"BASELINE (old)  : acc={fmt(base['accuracy'])} bacc={fmt(base['balanced_accuracy'])} f1={fmt(base['macro_f1'])} | curated-only acc={fmt(base['curated_accuracy'])} | "
        f"source-balanced recall={fmt(base['source_balanced_recall'])} on the same held-out streamed images")

    summary = {
        "class_totals": store.class_totals(),
        "groups": {f"{CLASS_NAMES[c]}/{g}": n for (c, g), n in sorted(store.counts.items())},
        "stats": dict(store.stats),
        "esa_by_class": next((s.n_by_class for s in sources if s.name == "esa_hubble"), {}),
        "source_errors": {s.name: s.error for s in sources if s.error},
        "val_items": len(items),
        "steps": step,
        "images_seen_in_training": seen,
        "train_minutes": (time.time() - t_train0) / 60,
        "args": {k: str(v) for k, v in vars(args).items()},
    }
    (run_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "metrics.json").write_text(json.dumps({"new": final, "new_no_tta": plain, "baseline": base, "history": history}, indent=2))
    save_plots(run_dir, history, final)
    log("dataset summary: " + json.dumps(summary["class_totals"]) + " | groups: " + json.dumps(summary["groups"]))
    log("done.")
    sys.stdout.flush()
    os._exit(0)  # daemon fetch threads may still be inside slow network calls


if __name__ == "__main__":
    main()
