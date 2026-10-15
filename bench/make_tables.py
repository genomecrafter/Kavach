"""Turns bench/results/*.json into LaTeX tables in paper/tables/ (the paper \\input's them)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES, OUT = ROOT / "bench" / "results", ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {"keyword": "Keyword blocklist", "tactic-only": "Tactic-only detector", "harm-only": "Harm-only filter",
         "no-decoupling": "Binary (harm OR tactic)", "kavach-heuristic": "\\textsc{Kavach}-H (ours)",
         "no-translit": "\\quad $-$ transliteration", "no-phonetic": "\\quad $-$ phonetic keys",
         "no-protective": "\\quad $-$ protective context", "naive-judge": "Naive LLM judge (yes/no)",
         "kavach-judge": "\\textsc{Kavach}-J (IDP judge, ours)", "kavach-cascade": "\\textsc{Kavach}-C (cascade, ours)",
         "llamaguard": "Llama Guard 4",
         "kavach-cascade-learned": "\\textsc{Kavach}-C, learned router (ours)",
         "kavach-cascade-union": "\\textsc{Kavach}-C, rules $\\cup$ learned (ours)"}


def load(name):
    p = RES / f"{name}.json"
    return {r["system"]: r for r in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}


def cell(r, grp, m):
    v = r["table"].get(grp, {}).get(m)
    return f"{v[0]:.1f}" if v else "--"


def table(rows, cols, caption, label, note=""):
    spec = "l" + "r" * len(cols)
    head = " & ".join(["System"] + [c[0] for c in cols]) + r" \\"
    body = []
    for sysname, r in rows:
        body.append(" & ".join([NAMES.get(sysname, sysname)] + [f(r) for _, f in cols]) + r" \\")
    return (f"\\begin{{table}}[t]\n\\centering\\footnotesize\n\\caption{{{caption}}}\\label{{{label}}}\n"
            f"\\setlength{{\\tabcolsep}}{{3.5pt}}\n\\resizebox{{\\columnwidth}}{{!}}{{\\begin{{tabular}}{{{spec}}}\n\\toprule\n{head}\n\\midrule\n"
            + "\n".join(body) + f"\n\\bottomrule\n\\end{{tabular}}}}\n{note}\\end{{table}}\n")


def wide(t):   # two-column table for the wide LLM results
    return (t.replace("{table}", "{table*}").replace("\\resizebox{\\columnwidth}{!}{", "{")
             .replace("\\footnotesize", "\\small"))


off = load("test_offline")
if off:
    sel = ["keyword", "tactic-only", "harm-only", "no-decoupling", "kavach-heuristic"]
    cols = [("DSR$\\uparrow$", lambda r: cell(r, "all", "DSR")),
            ("FRR$\\downarrow$", lambda r: cell(r, "all", "FRR")),
            ("FRR wild$\\downarrow$", lambda r: cell(r, "src:wild-regular", "FRR")),
            ("FRR XS$\\downarrow$", lambda r: cell(r, "src:xstest-safe", "FRR")),
            ("Route$^+$", lambda r: cell(r, "all", "ROUTE+") if r["system"] == "kavach-heuristic" else "--"),
            ("Route$^-$", lambda r: cell(r, "all", "ROUTE-") if r["system"] == "kavach-heuristic" else "--")]
    (OUT / "offline.tex").write_text(table([(s, off[s]) for s in sel if s in off], cols,
        "Model-free stage on the held-out test split (\\%). DSR: harmful prompts refused; FRR: benign prompts refused; "
        "wild = in-the-wild regular prompts, XS = XSTest safe prompts; Route$^{+/-}$: harmful/benign prompts refused "
        "or escalated to the judge.", "tab:offline"))
    sel = ["kavach-heuristic", "no-translit", "no-phonetic", "no-protective"]
    cols = [(n, (lambda g: lambda r: cell(r, g, "ROUTE+"))(g)) for n, g in
            [("en", "lang:en"), ("hi", "lang:hi"), ("hi-Latn", "lang:hi-Latn"), ("hi-Latn$^\\ast$", "lang:hi-Latn-noisy")]]
    cols += [("multi-turn", lambda r: cell(r, "src:india-multiturn", "ROUTE+")), ("FRR", lambda r: cell(r, "all", "FRR"))]
    (OUT / "ablation.tex").write_text(table([(s, off[s]) for s in sel if s in off], cols,
        "Ablation of the canonicalization stage: harmful-prompt routing recall (Route$^+$, \\%) by language. "
        "hi-Latn$^\\ast$ = romanized Hindi with phonetic spelling noise.", "tab:ablation"))

llm = {**load("test_llm"), **load("test_llamaguard")}
ROUTER_LABELS = {"tfidf": "TF-IDF", "xlmr": "XLM-R", "muril": "MuRIL", "mbert": "DistilmBERT"}
for f in sorted(RES.glob("test_llm_router_*.json")):
    for r in json.loads(f.read_text(encoding="utf-8")):
        llm[r["system"]] = r
        lab = ROUTER_LABELS.get(r.get("router_label"), r.get("router_label"))
        kind = "learned router" if r["system"].startswith("cascade-learned") else "rules $\\cup$ learned"
        NAMES[r["system"]] = f"\\textsc{{Kavach}}-C, {kind} ({lab})"
if llm:
    sel = [s for s in ["keyword", "kavach-heuristic", "llamaguard", "naive-judge", "kavach-judge", "kavach-cascade"] if s in llm]
    sel += sorted(s for s in llm if s.startswith("cascade-learned-")) + sorted(s for s in llm if s.startswith("cascade-union-"))
    groups = [("en", "lang:en"), ("hi", "lang:hi"), ("hi-Latn$^\\ast$", "lang:hi-Latn-noisy"), ("ta", "lang:ta"),
              ("bn", "lang:bn")]
    cols = [("DSR", lambda r: cell(r, "all", "DSR")), ("FRR", lambda r: cell(r, "all", "FRR"))]
    cols += [(f"DSR {n}", (lambda g: lambda r: cell(r, g, "DSR"))(g)) for n, g in groups]
    labels = {}
    for l in open(ROOT / "bench" / "data" / "test.jsonl", encoding="utf-8"):
        d = json.loads(l)
        labels[d["id"]] = (d["label"], d["lang"])

    def romanized(r):   # hi-Latn + hi-Latn* pooled (both are small groups)
        h = [i for i in r["items"] if labels[i["id"]][0] == 1 and labels[i["id"]][1].startswith("hi-Latn")]
        return f"{100 * sum(i['refuse'] for i in h) / len(h):.1f}" if h else "--"

    def intervened(r):  # harmful prompts refused OR answered only under GUARD/REDIRECT
        h = [i for i in r["items"] if labels[i["id"]][0] == 1]
        if not h or "actions" not in h[0]:
            return "--"
        k = sum(i["refuse"] or i["actions"][-1] in ("GUARD", "REDIRECT") for i in h)
        return f"{100 * k / len(h):.1f}"

    cols = [("DSR", lambda r: cell(r, "all", "DSR")), ("Int.", intervened), ("FRR", lambda r: cell(r, "all", "FRR"))]
    cols += [(f"DSR {n}", (lambda g: lambda r: cell(r, g, "DSR"))(g)) for n, g in
             [("en", "lang:en"), ("hi", "lang:hi")]]
    cols += [("DSR hi-Latn", romanized)]
    cols += [(f"DSR {n}", (lambda g: lambda r: cell(r, g, "DSR"))(g)) for n, g in [("ta", "lang:ta"), ("bn", "lang:bn")]]
    cols += [("MT", lambda r: cell(r, "src:india-multiturn", "DSR")), ("LLM calls", lambda r: f"{r['judge_call_rate']:.0f}")]
    model = next(iter(llm.values())).get("model", "LLM")
    nh = llm[sel[0]]["table"]["all"]["DSR"][1]
    nb = llm[sel[0]]["table"]["all"]["FRR"][1]
    (OUT / "llm.tex").write_text(wide(table([(s, llm[s]) for s in sel], cols,
        f"End-to-end results with \\texttt{{{model}}} as judge (\\%, stratified test sample: {nh} harmful, {nb} benign). "
        "DSR: harmful refused or blocked. Int.: harmful refused, blocked, or answered only under a hardened prompt "
        "(\\textsc{Guard}) or as a safe rewrite (\\textsc{Redirect}). hi-Latn pools plain and noisy romanized Hindi. "
        "MT: multi-turn jailbreak conversations defended. LLM calls: share of prompts that invoked the LLM.", "tab:llm")),
        encoding="utf-8")
else:
    (OUT / "llm.tex").write_text(
        "\\begin{table}[t]\\centering\\footnotesize\\caption{End-to-end results with an LLM judge.}\\label{tab:llm}\n"
        "\\fbox{\\parbox{0.9\\columnwidth}{\\centering PENDING: run \\texttt{bench/run\\_llm.py} with a Groq or Gemini key, "
        "then recompile.}}\\end{table}\n", encoding="utf-8")
print("tables ->", OUT, sorted(p.name for p in OUT.iterdir()))
