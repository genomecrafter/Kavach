"""Compare learned routers with the rule-based router on the held-out test split.

    python dl/analyze.py --runs dl/runs --out dl/report

Writes: dl/report/router_results.json, dl/report/*.tex (paper tables), dl/report/fig_*.png/pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench")); sys.path.insert(0, str(ROOT / "dl"))
from data import load_split  # noqa: E402

NAMES = {"tfidf-char-lr": "TF-IDF + LR", "xlm-roberta-base": "XLM-R base", "google/muril-base-cased": "MuRIL base",
         "distilbert-base-multilingual-cased": "DistilmBERT", "bert-base-multilingual-cased": "mBERT base",
         "ai4bharat/IndicBERTv2-MLM-only": "IndicBERT v2"}
# reference palette, categorical slots in fixed order (validated light-mode steps)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# colour follows the model, never its position in a list
MODEL_COLOR = {"XLM-R base": SERIES[0], "MuRIL base": SERIES[1], "DistilmBERT": SERIES[2], "TF-IDF + LR": SERIES[3],
               "mBERT base": SERIES[4], "IndicBERT v2": SERIES[5]}


def mcolor(name, i=0):
    return MODEL_COLOR.get(name, SERIES[i % len(SERIES)])


def lang_group(lang):
    if lang == "en":
        return "en"
    if lang == "hi":
        return "hi"
    if lang.startswith("hi-Latn"):
        return "hi-Latn"
    return "other Indic"


def rules_routing(test, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import evaluate as E
    run = E.make_kavach_runner("full")
    out = {r["id"]: bool(run(r["text"], turns=r.get("turns"), sid=r["id"])["routed"]) for r in test}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


def rates(rows, routed):
    h = [r for r in rows if r["label"] == 1]; b = [r for r in rows if r["label"] == 0]
    rp = 100 * sum(routed[r["id"]] for r in h) / max(len(h), 1)
    rn = 100 * sum(routed[r["id"]] for r in b) / max(len(b), 1)
    return round(rp, 1), round(rn, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="dl/runs")
    ap.add_argument("--out", default="dl/report")
    a = ap.parse_args()
    runs_dir, out = ROOT / a.runs, ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    test = load_split("test")
    by_id = {r["id"]: r for r in test}
    y = np.array([r["label"] for r in test])
    rules = rules_routing(test, out / "rules_routing.json")
    rules_p, rules_n = rates(test, rules)

    runs = []
    for d in sorted(p for p in runs_dir.iterdir() if (p / "router_config.json").exists() and (p / "test_preds.jsonl").exists()):
        cfg = json.loads((d / "router_config.json").read_text(encoding="utf-8"))
        preds = {json.loads(l)["id"]: json.loads(l)["p"] for l in open(d / "test_preds.jsonl", encoding="utf-8")}
        p = np.array([preds[r["id"]] for r in test])
        hist = json.loads((d / "history.json").read_text(encoding="utf-8")) if (d / "history.json").exists() else []
        runs.append({"dir": d.name, "cfg": cfg, "p": p, "hist": hist})

    results = {"rules": {"route_pos": rules_p, "route_neg": rules_n}, "runs": []}
    for R in runs:
        cfg, p = R["cfg"], R["p"]
        thr = cfg["threshold"]
        learned = {r["id"]: bool(pi >= thr) for r, pi in zip(test, p)}
        union = {k: learned[k] or rules[k] for k in learned}
        # matched budget: threshold giving the same benign-escalation rate as the rules
        pb = np.sort(p[y == 0])[::-1]
        k = int(round(rules_n / 100 * len(pb)))
        t_m = pb[min(k, len(pb) - 1)] if k > 0 else 1.01
        matched = {r["id"]: bool(pi > t_m) for r, pi in zip(test, p)}
        entry = {"run": R["dir"], "model": NAMES.get(cfg["base_model"], cfg["base_model"]),
                 "exclude_source": cfg.get("exclude_source"), "params_M": cfg.get("n_params_M"),
                 "auroc": round(100 * roc_auc_score(y, p), 1), "threshold": round(thr, 4),
                 "learned": rates(test, learned), "union": rates(test, union), "matched_budget": rates(test, matched),
                 "per_lang": {}, "multiturn": None}
        for g in ("en", "hi", "hi-Latn", "other Indic"):
            rows = [r for r in test if lang_group(r["lang"]) == g]
            entry["per_lang"][g] = {"learned": rates(rows, learned), "rules": rates(rows, rules)}
        mt = [r for r in test if r["source"] == "india-multiturn"]
        entry["multiturn"] = {"learned": rates(mt, learned)[0], "rules": rates(mt, rules)[0]}
        if cfg.get("exclude_source"):
            src = cfg["exclude_source"]
            rows = [r for r in test if r["source"] == src]
            idx = {r["id"]: i for i, r in enumerate(test)}
            # AUROC: held-out source's harmful prompts vs ALL benign test prompts (the source itself has no benign items)
            auc_rows = rows + [r for r in test if r["label"] == 0 and r["source"] != src]
            ys = np.array([r["label"] for r in auc_rows]); ps = np.array([p[idx[r["id"]]] for r in auc_rows])
            entry["heldout_source"] = {"source": src, "recall_learned": rates(rows, learned)[0],
                                       "recall_rules": rates(rows, rules)[0],
                                       "auroc": round(100 * roc_auc_score(ys, ps), 1) if 0 < ys.sum() < len(ys) else None}
        results["runs"].append(entry)
    (out / "router_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    # ---------- console summary ----------
    print(f"rules router: Route+ {rules_p}  Route- {rules_n}")
    for e in results["runs"]:
        tag = f"{e['model']}" + (f" (w/o {e['exclude_source']})" if e["exclude_source"] else "")
        print(f"{tag:34s} AUROC {e['auroc']:5.1f} | learned R+ {e['learned'][0]:5.1f} R- {e['learned'][1]:5.1f} | "
              f"matched-budget R+ {e['matched_budget'][0]:5.1f} | union R+ {e['union'][0]:5.1f} R- {e['union'][1]:5.1f}"
              + (f" | held-out {e['heldout_source']}" if e.get("heldout_source") else ""))

    # ---------- LaTeX tables ----------
    main_runs = [e for e in results["runs"] if not e["exclude_source"]]
    # aggregate seeds: one row per model, mean +- std when a model was trained with several seeds
    by_model = {}
    for e in main_runs:
        by_model.setdefault(e["model"], []).append(e)

    def ms(vals):
        v = np.array(vals, dtype=float)
        return f"{v.mean():.1f}" + (f"$\\pm${v.std(ddof=1):.1f}" if len(v) > 1 else "")
    agg = {}
    rows = [f"Rule-based router (\\textsc{{Kavach}}-H) & -- & -- & {rules_p} & {rules_n} & {rules_p} \\\\"]
    for m, es in by_model.items():
        agg[m] = {"n_seeds": len(es), "auroc": [e["auroc"] for e in es], "route_pos": [e["learned"][0] for e in es],
                  "route_neg": [e["learned"][1] for e in es], "matched": [e["matched_budget"][0] for e in es]}
        rows.append(f"{m}{f' ({len(es)} seeds)' if len(es) > 1 else ''} & {es[0]['params_M']} & {ms(agg[m]['auroc'])} & "
                    f"{ms(agg[m]['route_pos'])} & {ms(agg[m]['route_neg'])} & {ms(agg[m]['matched'])} \\\\")
    results["aggregate"] = agg
    (out / "router_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    main_runs = [es[0] for es in by_model.values()]   # one representative run per model for figures
    (out / "router_table.tex").write_text(
        "\\begin{table*}[t]\\centering\\small\n\\caption{Learned routers on the held-out test split (4{,}648 prompts). "
        "Route$^{+}$: harmful prompts sent to the judge; Route$^{-}$: benign prompts sent (cost). The operating threshold "
        "is chosen on validation for 95\\% recall. Last column: Route$^{+}$ when each router is allowed exactly the rule "
        "router's judge budget. Mean$\\pm$std over seeds where several were trained.}\\label{tab:router}\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Router & Params (M) & AUROC & Route$^{+}$ & Route$^{-}$ & Route$^{+}$ @ equal budget \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table*}\n", encoding="utf-8")
    loso = [e for e in results["runs"] if e.get("heldout_source")]
    if loso:
        lr = [f"{e['model']} & {e['heldout_source']['auroc']} & {e['heldout_source']['recall_learned']} & "
              f"{e['heldout_source']['recall_rules']} \\\\" for e in loso]
        (out / "loso_table.tex").write_text(
            "\\begin{table}[t]\\centering\\footnotesize\n\\caption{Generalisation to an unseen source: routers trained "
            "\\emph{without} the India-safety data, tested on it (5 languages + romanized Hindi).}\\label{tab:loso}\n"
            "\\begin{tabular}{lrrr}\n\\toprule\nRouter & AUROC & Route$^{+}$ learned & Route$^{+}$ rules \\\\\n\\midrule\n"
            + "\n".join(lr) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n", encoding="utf-8")

    # ---------- figures ----------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    seen = set()
    fig_runs = []
    for R in runs:
        nm = NAMES.get(R["cfg"]["base_model"], R["cfg"]["base_model"])
        if not R["cfg"].get("exclude_source") and nm not in seen:
            seen.add(nm); fig_runs.append(R)
    for i, R in enumerate(fig_runs):
        fpr, tpr, _ = roc_curve(y, R["p"])
        ax.plot(100 * fpr, 100 * tpr, color=mcolor(NAMES.get(R['cfg']['base_model'], R['cfg']['base_model']), i), lw=1.6,
                label=f"{NAMES.get(R['cfg']['base_model'], R['cfg']['base_model'])} ({100 * roc_auc_score(y, R['p']):.1f})")
    ax.plot([rules_n], [rules_p], marker="D", ms=6, color=INK, ls="none", label="Rule router")
    ax.set_xlabel("Benign prompts sent to judge (%)"); ax.set_ylabel("Harmful prompts sent to judge (%)")
    ax.set_xlim(0, 100); ax.set_ylim(0, 101); ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_router_roc.pdf"); fig.savefig(out / "fig_router_roc.png", dpi=200)

    if main_runs:
        groups = ["en", "hi", "hi-Latn", "other Indic"]
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
        series = [("Rule router", [main_runs[0]["per_lang"][g]["rules"][0] for g in groups], INK2)]
        for i, e in enumerate(main_runs):
            series.append((e["model"], [e["per_lang"][g]["learned"][0] for g in groups], mcolor(e["model"], i)))
        wbar = 0.8 / len(series)
        for j, (name, vals, col) in enumerate(series):
            ax.bar(np.arange(len(groups)) + (j - (len(series) - 1) / 2) * wbar, vals, wbar * 0.92, color=col, label=name)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups); ax.set_ylim(0, 105)
        ax.set_ylabel("Harmful routed to judge (%)")
        ax.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
        fig.tight_layout(); fig.savefig(out / "fig_router_lang.pdf"); fig.savefig(out / "fig_router_lang.png", dpi=200)

    hist_runs = [R for R in fig_runs if R["hist"]]
    if hist_runs:
        fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.2))
        for i, R in enumerate(hist_runs):
            ep = [h["epoch"] for h in R["hist"]]
            nm = NAMES.get(R["cfg"]["base_model"], R["cfg"]["base_model"])
            axs[0].plot(ep, [h["train_loss"] for h in R["hist"]], color=mcolor(nm, i), lw=1.6, marker="o", ms=3, label=nm)
            axs[1].plot(ep, [100 * h["val_auroc"] for h in R["hist"]], color=mcolor(nm, i), lw=1.6, marker="o", ms=3, label=nm)
        axs[0].set_title("Training loss", fontsize=8); axs[1].set_title("Validation AUROC (%)", fontsize=8)
        for ax in axs:
            ax.set_xlabel("Epoch"); ax.set_xticks(ep)
        axs[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(out / "fig_training.pdf"); fig.savefig(out / "fig_training.png", dpi=200)
    # keep the paper folder self-contained (Overleaf): copy tables and vector figures next to main.tex
    import shutil
    (ROOT / "paper" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "paper" / "figures").mkdir(parents=True, exist_ok=True)
    for f in ("router_table.tex", "loso_table.tex"):
        if (out / f).exists():
            shutil.copy(out / f, ROOT / "paper" / "tables" / f)
    for f in ("fig_router_roc.pdf", "fig_router_lang.pdf", "fig_training.pdf"):
        if (out / f).exists():
            shutil.copy(out / f, ROOT / "paper" / "figures" / f)
    print("report ->", out)


if __name__ == "__main__":
    main()
