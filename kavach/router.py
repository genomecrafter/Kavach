"""Learned router (deep-learning component).

A fine-tuned multilingual transformer (XLM-R / MuRIL / mBERT) that estimates
P(harmful | prompt, recent turns). In cascade mode it decides which prompts are
worth an LLM-judge call, replacing or complementing the hand-written lexical
router. Like the rules, it never refuses on its own; it only routes.

    from kavach.router import LearnedRouter
    guard = Kavach(llm=..., judge_llm=..., router=LearnedRouter("router_model"))

Needs:  pip install torch transformers
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# KAVACH's transformer router uses PyTorch only. If TensorFlow/Flax happen to be installed (often an old or
# protobuf-incompatible TensorFlow), `transformers` may try to import them and crash. Disable them up front;
# this must run before `transformers` is imported anywhere.
for _k, _v in (("USE_TF", "0"), ("USE_TORCH", "1"), ("USE_FLAX", "0"), ("TRANSFORMERS_NO_TF", "1")):
    os.environ.setdefault(_k, _v)

SEP = " \n[TURN] "


def _load_or_refit_sklearn(p: Path, cfg: dict):
    """Pickled scikit-learn models break across library versions. Load the saved model only if it was
    written by the installed scikit-learn version and still works; otherwise re-fit it here from the
    project's dev split (about 10-20 s, deterministic) and save it for next time."""
    import joblib
    import sklearn
    if cfg.get("sklearn_version") == sklearn.__version__ and (p / "model.joblib").exists():
        try:
            m = joblib.load(p / "model.joblib")
            m.predict_proba(["hello"])
            return m
        except Exception:
            pass
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "dl"))
    from data import train_val  # noqa: E402
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    print(f"[kavach] re-fitting the TF-IDF router for scikit-learn {sklearn.__version__} (one time, ~20 s)...", flush=True)
    tr, _ = train_val(exclude_source=cfg.get("exclude_source"))
    m = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                      max_features=300000),
                      LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"))
    m.fit([r["x"] for r in tr], [r["label"] for r in tr])
    try:
        joblib.dump(m, p / "model.joblib")
        cfg["sklearn_version"] = sklearn.__version__
        (p / "router_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    except OSError:
        pass
    return m


def router_input(text: str, history: list[str] | None = None, max_turns: int = 3) -> str:
    """Router sees the last few user turns, so multi-turn escalation is visible to it."""
    turns = (history or [])[-(max_turns - 1):] + [text] if max_turns > 1 else [text]
    return SEP.join(t[:1500] for t in turns)


class LearnedRouter:
    def __init__(self, path: str, device: str | None = None, threshold: float | None = None, max_len: int = 256):
        p = Path(path)
        cfg = json.loads((p / "router_config.json").read_text(encoding="utf-8")) if (p / "router_config.json").exists() else {}
        self.threshold = threshold if threshold is not None else float(cfg.get("threshold", 0.5))
        self.max_len = int(cfg.get("max_len", max_len))
        self.name = cfg.get("base_model", p.name)
        self._sk = None
        if (p / "model.joblib").exists() or cfg.get("kind") == "sklearn":   # classical baseline router (TF-IDF + LR)
            self._sk, self.device = _load_or_refit_sklearn(p, cfg), "cpu"
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(p)
        self.model = AutoModelForSequenceClassification.from_pretrained(p).float().to(self.device).eval()
        self._torch = torch
        self.score(["warm-up"])          # first forward pass is slow on CPU; do it at start-up, not mid-demo

    def score(self, texts: list[str], batch_size: int = 32) -> list[float]:
        if self._sk is not None:
            return [float(x) for x in self._sk.predict_proba(texts)[:, 1]]
        out = []
        with self._torch.no_grad():
            for i in range(0, len(texts), batch_size):
                enc = self.tok(texts[i:i + batch_size], truncation=True, max_length=self.max_len,
                               padding=True, return_tensors="pt").to(self.device)
                logits = self.model(**enc).logits
                out += self._torch.softmax(logits, -1)[:, 1].float().cpu().tolist()
        return out

    def __call__(self, text: str, history: list[str] | None = None) -> float:
        return self.score([router_input(text, history)])[0]
