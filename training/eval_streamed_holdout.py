#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate one or more ResNet-34 weight files on the held-out streamed split.

The split is defined by md5(seed:uid) (see training/streaming/store.py), so every image selected here is
one that stream_train.py never used for training. Only that ~5% subset is downloaded (streamed into RAM,
nothing is written to disk). ESA/Hubble is skipped because its shards are huge.

Usage: python training/eval_streamed_holdout.py --weights trained_net.pth.backup trained_net_stream.pth
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stream_train import build_model, metrics_from, predict  # noqa: E402
from streaming import CLASS_NAMES  # noqa: E402
from streaming.sources import build_recipe  # noqa: E402
from streaming.store import ImageStore  # noqa: E402

TRAINING_DIR = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", type=Path, required=True)
    ap.add_argument("--target", type=int, default=20000, help="recipe size (controls how many candidates are crawled)")
    ap.add_argument("--timeout-min", type=float, default=25)
    ap.add_argument("--out", type=Path, default=None, help="optional JSON file with the metrics")
    args = ap.parse_args()

    store = ImageStore(val_fraction=0.05, seed=0)  # identical split to training
    store.only_val = True
    stop = threading.Event()
    sources = [s for s in build_recipe(args.target) if s.name != "esa_hubble"]
    for s in sources:
        if s.name == "decals_galaxies":
            s.quota = 300
    threads = [threading.Thread(target=s.run, args=(store, stop), daemon=True) for s in sources]
    for t in threads:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in threads) and time.time() - t0 < args.timeout_min * 60:
        time.sleep(15)
        print(f"[{time.time() - t0:5.0f}s] held-out images so far: {store.class_totals()}", flush=True)
    stop.set()

    items = store.val_items()
    print("held-out set:", store.class_totals(), "n =", len(items), flush=True)
    dev = torch.device("cuda")
    results = {}
    for w in args.weights:
        m = metrics_from(predict(build_model(w, dev), items, dev), items)
        results[str(w)] = m
        print(f"\n== {w.name}: acc={100 * m['accuracy']:.2f}% bacc={100 * m['balanced_accuracy']:.2f}% f1={100 * m['macro_f1']:.2f}% "
              f"curated-only={m['curated_accuracy'] and round(100 * m['curated_accuracy'], 2)}% (n={m['curated_n']}) "
              f"source-balanced recall={100 * m['source_balanced_recall']:.2f}%")
        print("   recall:", {k: round(100 * v['recall'], 1) for k, v in m['per_class'].items()}, "support:", {k: v['support'] for k, v in m['per_class'].items()})
        print("   per-group:", {g: f"{100 * v['acc']:.0f}% (n={v['n']})" for g, v in m['per_group_acc'].items()})
    if args.out:
        args.out.write_text(json.dumps({"n": len(items), "class_totals": store.class_totals(), "results": results}, indent=2))
    sys.stdout.flush()
    import os

    os._exit(0)


if __name__ == "__main__":
    main()
