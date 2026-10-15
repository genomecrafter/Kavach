"""Classical-ML baseline router: character n-gram TF-IDF + logistic regression (same splits as the transformers).

    python dl/tfidf_baseline.py --out dl/runs/tfidf
"""
import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline

from data import load_split, threshold_for_recall, train_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dl/runs/tfidf")
    ap.add_argument("--target_recall", type=float, default=0.95)
    ap.add_argument("--exclude_source", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr, va = train_val(exclude_source=a.exclude_source)
    te = load_split("test")
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                                        max_features=300000),
                        LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced"))
    clf.fit([r["x"] for r in tr], [r["label"] for r in tr])
    pv = clf.predict_proba([r["x"] for r in va])[:, 1]
    pt = clf.predict_proba([r["x"] for r in te])[:, 1]
    yv, yt = [r["label"] for r in va], [r["label"] for r in te]
    thr = threshold_for_recall(yv, pv, a.target_recall)
    with open(out / "test_preds.jsonl", "w", encoding="utf-8") as f:
        for r, p in zip(te, pt):
            f.write(json.dumps({"id": r["id"], "p": round(float(p), 5)}) + "\n")
    cfg = {"base_model": "tfidf-char-lr", "threshold": thr, "target_recall": a.target_recall,
           "exclude_source": a.exclude_source, "best_val_auroc": roc_auc_score(yv, pv), "kind": "sklearn",
           "test_auroc": roc_auc_score(yt, pt), "test_auprc": average_precision_score(yt, pt), "n_params_M": 0.3}
    import joblib
    import sklearn
    cfg["sklearn_version"] = sklearn.__version__
    joblib.dump(clf, out / "model.joblib")
    (out / "router_config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print("FINAL", json.dumps(cfg))


if __name__ == "__main__":
    main()
