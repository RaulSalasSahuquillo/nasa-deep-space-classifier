#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two ResNet-34 AstroClass weight files on the project's own local images.

The local folders (images/, network_sorting/) were labelled by the *previous* model, so they are
NOT ground truth: this tool only shows where the new weights disagree with the old ones so the
disagreements can be inspected by eye.

Usage: python training/audit_local_images.py [--old trained_net.pth] [--new trained_net_stream.pth]
"""

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet34

ROOT = Path(__file__).resolve().parent.parent
CLASS_NAMES = ["star", "galaxy", "quasar", "nebula", "planet"]
TF = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def load(path: Path, device) -> nn.Module:
    net = resnet34(weights=None)
    net.fc = nn.Linear(net.fc.in_features, len(CLASS_NAMES))
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, default=ROOT / "training" / "trained_net.pth")
    ap.add_argument("--new", type=Path, default=ROOT / "training" / "trained_net_stream.pth")
    ap.add_argument("--dirs", nargs="*", default=["images/nasaDownloads", "network_sorting"])
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old, new = load(args.old, dev), load(args.new, dev)

    files = []
    for d in args.dirs:
        files += [p for p in (ROOT / d).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    rows = []
    for p in sorted(files):
        try:
            x = TF(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
        except Exception:
            continue
        po, pn = torch.softmax(old(x), 1)[0], torch.softmax(new(x), 1)[0]
        folder = p.parent.name if p.parent.name in CLASS_NAMES else ""
        rows.append((str(p.relative_to(ROOT)), folder, CLASS_NAMES[int(po.argmax())], float(po.max()), CLASS_NAMES[int(pn.argmax())], float(pn.max())))
    n = len(rows)
    agree = sum(r[2] == r[4] for r in rows)
    print(f"{n} images | old/new agree on {agree} ({100 * agree / max(1, n):.1f}%)")
    dis = [r for r in rows if r[2] != r[4]]
    print(f"{len(dis)} disagreements")
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "folder_label", "old_pred", "old_conf", "new_pred", "new_conf"])
            w.writerows(rows)
    for r in dis[:80]:
        print(f"{r[0]:<95} folder={r[1]:<7} old={r[2]:<7}({r[3]:.2f}) new={r[4]:<7}({r[5]:.2f})")


if __name__ == "__main__":
    main()
