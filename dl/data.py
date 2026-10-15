"""Data for the learned router.

train/val : bench/data/dev.jsonl   (the same 30 % split used to tune the rules; val = 15 % of it, grouped)
test      : bench/data/test.jsonl  (never used for training or model selection)

Groups: an item and its translations / romanizations share a group id, so they never
straddle train and val (no translation leakage).
"""
from __future__ import annotations

import json
import math
import random

import numpy as np
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kavach.router import router_input  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "bench" / "data"


def group_id(item_id: str) -> str:
    return re.sub(r"-(en|hi|ta|te|bn|mr|gu|hiL|hiLn)$", "", item_id)


def to_text(r: dict) -> str:
    turns = r.get("turns")
    if turns:
        return router_input(turns[-1], turns[:-1])
    return router_input(r["text"])


def load_split(name: str) -> list[dict]:
    rows = [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8")]
    for r in rows:
        r["x"] = to_text(r)
        r["group"] = group_id(r["id"])
    return rows


def train_val(val_frac: float = 0.15, seed: int = 0, exclude_source: str | None = None):
    dev = load_split("dev")
    if exclude_source:
        dev = [r for r in dev if r["source"] != exclude_source]
    groups = sorted({r["group"] for r in dev})
    rng = random.Random(seed)
    rng.shuffle(groups)
    val_g = set(groups[: int(len(groups) * val_frac)])
    return [r for r in dev if r["group"] not in val_g], [r for r in dev if r["group"] in val_g]


def threshold_for_recall(y, p, target):
    """Largest threshold whose recall on harmful items is >= target (fewest judge calls for that recall)."""
    pos = np.sort(np.asarray(p)[np.asarray(y) == 1])
    if len(pos) == 0:
        return 0.5
    k = int(math.floor((1 - target) * len(pos)))
    return float(pos[min(k, len(pos) - 1)])



if __name__ == "__main__":
    tr, va = train_val()
    te = load_split("test")
    for n, d in (("train", tr), ("val", va), ("test", te)):
        print(n, len(d), "harmful", sum(r["label"] for r in d))
