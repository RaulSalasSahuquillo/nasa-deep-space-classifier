"""In-memory image store with de-duplication, quality gates and a fixed val split.

Every accepted image is stored as a 256x256 JPEG (~15-40 KB). 100k images take
roughly 1-3 GB of RAM and never touch the disk.

De-duplication
--------------
* every source: 64-bit dHash, canonical over the 8 flips/rotations, Hamming distance <= 6;
* curated sources (nasa / esa / commons), where the same press picture is republished with
  other crops, stretches and resolutions: additionally a 16x16 low-frequency signature
  (mean-removed, unit norm) with cosine similarity >= 0.93.
A duplicate is dropped, so a picture and its twin can never end up on opposite sides of the
train/val split.
"""

from __future__ import annotations

import hashlib
import io
import random
import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import CLASS_NAMES

STORE_SIZE = 256
Image.MAX_IMAGE_PIXELS = 400_000_000  # big press images are fine, we only downscale them

CURATED_GROUPS = ("nasa", "esa", "commons")
# Mean-luminance ceiling per group. Astronomical frames are dark; charts/diagrams/landscapes are not.
# NASA/ESA are curated press images (bright Moon/Mars/infrared frames are legitimate), Commons is not.
BRIGHT_LIMIT = {"commons": 135.0, "nasa": 175.0, "esa": 175.0}
HASH_BITS = 6
SIG_COS = 0.93


def _dhash_canonical(gray9: np.ndarray) -> int:
    """dHash of a 9x9 grayscale thumbnail, minimised over the 8 dihedral variants."""
    best = None
    g = gray9
    for flip in (False, True):
        h = g[:, ::-1] if flip else g
        for k in range(4):
            r = np.rot90(h, k)
            bits = (r[:8, 1:] > r[:8, :-1]).flatten()
            val = 0
            for b in bits:
                val = (val << 1) | int(b)
            best = val if best is None else min(best, val)
    return best


def prepare(raw: bytes, max_mean: Optional[float] = None) -> Tuple[Optional[Tuple[bytes, int, np.ndarray]], str]:
    """Decode raw bytes -> ((jpeg256, canonical dhash, 16x16 signature), "ok") or (None, reason)."""
    try:
        im = Image.open(io.BytesIO(raw))
        if im.format == "JPEG":
            im.draft("RGB", (STORE_SIZE * 2, STORE_SIZE * 2))  # fast DCT-domain downscale
        w, h = im.size
        if min(w, h) < 48 or max(w, h) / min(w, h) > 2.6:
            return None, "shape"
        if w * h > 120_000_000 and im.format != "JPEG":
            return None, "huge"  # giant lossless frame that would blow up RAM when decoded
        im = im.convert("RGB")
    except Exception:
        return None, "decode"
    thumb = np.asarray(im.resize((32, 32), Image.BILINEAR), dtype=np.float32)
    # Reject blank / near-constant frames (outside survey footprint, error placeholders).
    if thumb.std() < 3.0 or thumb.max() < 30:
        return None, "blank"
    if max_mean is not None and float(thumb.mean()) > max_mean:
        return None, "bright"
    gray = im.convert("L")
    g9 = np.asarray(gray.resize((9, 9), Image.BILINEAR), dtype=np.float32)
    g16 = np.asarray(gray.resize((16, 16), Image.BILINEAR), dtype=np.float32).flatten()
    g16 = g16 - g16.mean()
    n = np.linalg.norm(g16)
    sig = (g16 / n).astype(np.float32) if n > 1e-3 else np.zeros(256, np.float32)
    im = im.resize((STORE_SIZE, STORE_SIZE), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return (buf.getvalue(), _dhash_canonical(g9), sig), "ok"


class ImageStore:
    """Thread-safe store. Items are addressed by integer index."""

    def __init__(self, val_fraction: float = 0.05, seed: int = 0):
        self.lock = threading.Lock()
        self.val_fraction = val_fraction
        self.seed = seed
        self.only_val = False  # evaluation mode: sources skip downloading anything that is not held-out
        self.data: List[bytes] = []
        self.label: List[int] = []
        self.group: List[str] = []
        self.is_val: List[bool] = []
        self.by_cg: Dict[Tuple[int, str], List[int]] = defaultdict(list)  # train indices
        self.val_idx: List[int] = []
        self._uids = set()
        self._hashes = np.zeros(1024, dtype=np.uint64)
        self._n_hash = 0
        self._hash_item = np.zeros(1024, dtype=np.int64)
        self._sigs = np.zeros((1024, 256), dtype=np.float32)  # curated items only
        self._sig_item: List[int] = []
        self.stats = defaultdict(int)  # status -> count
        self.counts = defaultdict(int)  # (class_idx, group) -> accepted (train+val)

    # ingest
    def _is_val(self, uid: str) -> bool:
        h = int(hashlib.md5(f"{self.seed}:{uid}".encode()).hexdigest()[:8], 16)
        return (h % 10_000) < int(self.val_fraction * 10_000)

    def wants(self, uid: str) -> bool:
        """True when the image should be downloaded at all (always, unless only_val is set)."""
        return (not self.only_val) or self._is_val(uid)

    def add_raw(self, cls: int, group: str, uid: str, raw: bytes) -> str:
        """Prepare + insert. Returns a status (ok / dup_uid / dup_img / conflict / bad)."""
        with self.lock:
            if uid in self._uids:
                self.stats["dup_uid"] += 1
                return "dup_uid"
        prep, why = prepare(raw, BRIGHT_LIMIT.get(group))
        if prep is None:
            with self.lock:
                self.stats["bad"] += 1
                self.stats[f"bad_{why}_{CLASS_NAMES[cls]}"] += 1
            return "bad"
        jpeg, dh, sig = prep
        curated = group in CURATED_GROUPS
        with self.lock:
            if uid in self._uids:
                self.stats["dup_uid"] += 1
                return "dup_uid"
            self._uids.add(uid)
            hit = None
            if self._n_hash:
                hs = self._hashes[: self._n_hash]
                x = np.bitwise_xor(hs, np.uint64(dh))
                near = np.nonzero(np.bitwise_count(x) <= HASH_BITS)[0]
                if near.size:
                    hit = int(self._hash_item[near[0]])
            if hit is None and curated and self._sig_item and float(sig @ sig) > 0.5:
                sims = self._sigs[: len(self._sig_item)] @ sig
                j = int(np.argmax(sims))
                if sims[j] >= SIG_COS:
                    hit = self._sig_item[j]
            if hit is not None:
                key = "dup_img" if self.label[hit] == cls else "conflict"
                self.stats[key] += 1
                return key
            idx = len(self.data)
            self.data.append(jpeg)
            self.label.append(cls)
            self.group.append(group)
            if self._n_hash == len(self._hashes):
                self._hashes = np.concatenate([self._hashes, np.zeros(len(self._hashes), np.uint64)])
                self._hash_item = np.concatenate([self._hash_item, np.zeros(len(self._hash_item), np.int64)])
            self._hashes[self._n_hash] = np.uint64(dh)
            self._hash_item[self._n_hash] = idx
            self._n_hash += 1
            if curated and float(sig @ sig) > 0.5:
                if len(self._sig_item) == len(self._sigs):
                    self._sigs = np.concatenate([self._sigs, np.zeros_like(self._sigs)])
                self._sigs[len(self._sig_item)] = sig
                self._sig_item.append(idx)
            val = self._is_val(uid)
            self.is_val.append(val)
            if val:
                self.val_idx.append(idx)
            else:
                self.by_cg[(cls, group)].append(idx)
            self.counts[(cls, group)] += 1
            self.stats["ok"] += 1
            return "ok"

    # queries
    def class_totals(self) -> Dict[str, int]:
        with self.lock:
            out = {n: 0 for n in CLASS_NAMES}
            for (c, _g), n in self.counts.items():
                out[CLASS_NAMES[c]] += n
            return out

    def train_counts(self) -> Dict[int, int]:
        with self.lock:
            out: Dict[int, int] = defaultdict(int)
            for (c, _g), idxs in self.by_cg.items():
                out[c] += len(idxs)
            return dict(out)

    def sample_batch(self, n: int, group_weights: Dict[str, float], rng: random.Random) -> Tuple[List[bytes], List[int]]:
        """Class-balanced, group-weighted sampling with replacement from the *current* train pool."""
        with self.lock:
            per_class: Dict[int, List[Tuple[str, List[int]]]] = defaultdict(list)
            for (c, g), idxs in self.by_cg.items():
                if idxs:
                    per_class[c].append((g, idxs))
            classes = sorted(per_class)
            if not classes:
                return [], []
            out_b, out_y = [], []
            for _ in range(n):
                c = rng.choice(classes)
                groups = per_class[c]
                # sqrt(size) keeps tiny curated groups represented without memorising them
                w = [group_weights.get(g, 1.0) * (len(ix) ** 0.5) for g, ix in groups]
                g, idxs = rng.choices(groups, weights=w, k=1)[0]
                i = idxs[rng.randrange(len(idxs))]
                out_b.append(self.data[i])
                out_y.append(c)
            return out_b, out_y

    def val_items(self) -> List[Tuple[bytes, int, str]]:
        with self.lock:
            return [(self.data[i], self.label[i], self.group[i]) for i in self.val_idx]

    def memory_mb(self) -> float:
        with self.lock:
            return sum(len(b) for b in self.data) / 1e6
