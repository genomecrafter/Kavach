"""Fine-tune a multilingual transformer as KAVACH's learned router.

    python dl/train_router.py --model xlm-roberta-base --out dl/runs/xlmr
    python dl/train_router.py --model google/muril-base-cased --out dl/runs/muril
    python dl/train_router.py --model distilbert-base-multilingual-cased --out dl/runs/mbert
    python dl/train_router.py --model xlm-roberta-base --exclude_source india-safety --out dl/runs/xlmr_loso   # generalisation

Outputs in --out: the model (fp16), router_config.json (operating threshold chosen on validation),
history.json (per-epoch loss / val metrics), test_preds.jsonl (P(harmful) for every test prompt).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import os
for _k, _v in (("USE_TF", "0"), ("USE_TORCH", "1"), ("USE_FLAX", "0"), ("TRANSFORMERS_NO_TF", "1")):
    os.environ.setdefault(_k, _v)   # PyTorch only: never let transformers import a (possibly broken) TensorFlow

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from data import load_split, threshold_for_recall, train_val


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def predict(model, tok, texts, max_len, bs, device):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            with torch.autocast(device_type=device.split(":")[0], dtype=torch.float16, enabled=device.startswith("cuda")):
                logits = model(**enc).logits
            out += torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xlm-roberta-base")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--exclude_source", default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--from_scratch_config", default=None, help="(testing only) random-init model from a config dir")
    a = ap.parse_args()
    set_seed(a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va = train_val(exclude_source=a.exclude_source)
    te = load_split("test")
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  device {device}  model {a.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.from_scratch_config or a.model)
    if a.from_scratch_config:
        from transformers import AutoConfig
        model = AutoModelForSequenceClassification.from_config(AutoConfig.from_pretrained(a.from_scratch_config, num_labels=2))
    else:
        model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2)
    model.to(device)

    def collate(batch):
        enc = tok([r["x"] for r in batch], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([r["label"] for r in batch])
        return enc

    dl = DataLoader(tr, batch_size=a.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * len(dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    # class balance: weight positives so both classes contribute equally
    pos = sum(r["label"] for r in tr); w = torch.tensor([len(tr) / (2 * (len(tr) - pos)), len(tr) / (2 * pos)]).float().to(device)
    lossf = torch.nn.CrossEntropyLoss(weight=w)

    hist, best, best_state = [], -1.0, None
    yv = [r["label"] for r in va]
    for ep in range(a.epochs):
        model.train(); t0, tot, n = time.time(), 0.0, 0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = lossf(model(**batch).logits.float(), labels)
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(labels); n += len(labels)
        pv = predict(model, tok, [r["x"] for r in va], a.max_len, 64, device)
        auc, ap_ = roc_auc_score(yv, pv), average_precision_score(yv, pv)
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_auroc": auc, "val_auprc": ap_, "sec": time.time() - t0})
        print(json.dumps(hist[-1]), flush=True)
        if auc > best:
            best = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    pv = predict(model, tok, [r["x"] for r in va], a.max_len, 64, device)
    thr = threshold_for_recall(yv, pv, a.target_recall)
    pt = predict(model, tok, [r["x"] for r in te], a.max_len, 64, device)

    with open(out / "test_preds.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(te, pt):
            f.write(json.dumps({"id": r["id"], "p": round(p, 5)}) + "\n")
    with open(out / "val_preds.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(va, pv):
            f.write(json.dumps({"id": r["id"], "p": round(p, 5)}) + "\n")
    (out / "history.json").write_text(json.dumps(hist, indent=1), encoding="utf-8")
    model.half().save_pretrained(out); tok.save_pretrained(out)
    yt = [r["label"] for r in te]
    cfg = {"base_model": a.model, "threshold": thr, "target_recall": a.target_recall, "max_len": a.max_len,
           "exclude_source": a.exclude_source, "best_val_auroc": best,
           "test_auroc": roc_auc_score(yt, pt), "test_auprc": average_precision_score(yt, pt),
           "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    (out / "router_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("FINAL", json.dumps(cfg))


if __name__ == "__main__":
    main()
