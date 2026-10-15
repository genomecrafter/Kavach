"""One-command LLM experiments (works on Windows, macOS and Linux).

    Put your key in a file named .env next to pyproject.toml:   GEMINI_API_KEY=your-key
    (or: PowerShell  $env:GEMINI_API_KEY="..."   |   bash  export GEMINI_API_KEY=...)
    python bench/run_llm.py

Every API response is cached in bench/results/llm_cache.sqlite. If the free daily quota runs out,
just run the same command again later: finished calls are reused and the run continues where it stopped.
"""
import argparse
import os
import secrets
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
RES.mkdir(parents=True, exist_ok=True)

# API key from a .env file next to pyproject.toml (lines like GEMINI_API_KEY=...), if present
env_file = ROOT / ".env"
if env_file.exists():
    raw = env_file.read_bytes()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    for line in text.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Stable judge-fence secret + response cache, set BEFORE kavach is imported.
# (nonce_key.txt is the released key that reproduces the cached Gemini run; .nonce_key is the older name)
key_file = next((RES / n for n in ("nonce_key.txt", ".nonce_key") if (RES / n).exists()), RES / "nonce_key.txt")
if not key_file.exists():
    key_file.write_text(secrets.token_hex(16), encoding="utf-8")
os.environ.setdefault("KAVACH_NONCE_KEY", key_file.read_text(encoding="utf-8").strip())
os.environ.setdefault("KAVACH_CACHE", str(RES / "llm_cache.sqlite"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import evaluate as E  # noqa: E402
from kavach.judge import IntentJudge  # noqa: E402
from kavach.llm import OpenAICompatLLM, QuotaExhausted  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini", choices=["gemini", "groq", "openai", "ollama"])
    ap.add_argument("--model", default=None, help="e.g. gemini-3.5-flash-lite (default: auto-pick)")
    ap.add_argument("--n", type=int, default=6, help="prompts per (source, language) group")
    ap.add_argument("--rpm", type=float, default=None, help="requests/minute cap (default: 10 gemini, 28 groq)")
    ap.add_argument("--systems", default="kavach-judge,kavach-cascade,naive-judge")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()

    rpm = a.rpm or {"gemini": 10, "groq": 28}.get(a.provider, 30)
    os.environ["KAVACH_RPM"] = str(rpm)
    if a.list_models:
        url, env, _ = __import__("kavach.llm", fromlist=["PRESETS"]).PRESETS[a.provider]
        print("\n".join(OpenAICompatLLM.list_models(url, os.environ[env])))
        return

    llm = OpenAICompatLLM.from_preset(a.provider, a.model)
    print(f"provider={a.provider}  model={llm.model}  rpm cap={rpm}  cache={os.environ['KAVACH_CACHE']}")

    # 1. smoke test: one judge call must parse
    v = IntentJudge(llm).judge("Phishing se apne parivaar ko kaise bachaye?")
    if v.error:
        print("Judge smoke test FAILED:", v.error, "\nRaw output:", v.raw[:500])
        print("Try another model:  python bench/run_llm.py --list-models   then  --model <id>")
        sys.exit(1)
    print(f"smoke test ok -> severity={v.harm_severity}, language={v.user_language}, core='{v.core_request[:80]}'")

    data = E.load("test", a.n)
    turns = sum(len(r.get("turns") or [1]) for r in data)
    systems = a.systems.split(",")
    est = turns * sum(s in ("kavach-judge", "naive-judge") for s in systems)
    print(f"{len(data)} test prompts ({turns} turns). Estimated new API calls: <= {est} "
          f"(~{est / rpm:.0f} min at {rpm} rpm; cascade re-uses the judge's cached calls).")

    builders = {"kavach-judge": lambda: E.make_kavach_runner("full", judge_llm=llm, mode="judge"),
                "kavach-cascade": lambda: E.make_kavach_runner("full", judge_llm=llm, mode="cascade"),
                "naive-judge": lambda: E.make_naive_judge(llm),
                "kavach-heuristic": lambda: E.make_kavach_runner("full"),
                "keyword": lambda: E.run_keyword}
    results, t0 = [], time.time()
    try:
        for s in ["keyword", "kavach-heuristic"] + systems:
            print(f"running {s} ...", flush=True)
            results.append(E.evaluate(builders[s](), data, s))
    except QuotaExhausted as e:
        print(f"\nFree-tier quota reached after {llm.calls} API calls ({e}).\n"
              "Nothing is lost: run the SAME command again later (or tomorrow) and it resumes from the cache.")
        sys.exit(2)

    import json
    for r in results:
        r["model"] = llm.model
    (RES / "test_llm.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    groups = ["all"] + sorted({g for r in results for g in r["table"] if g != "all"})
    E.print_table(results, groups)
    print(f"\n{llm.calls} API calls in {(time.time() - t0) / 60:.1f} min. Saved bench/results/test_llm.json")
    import make_tables  # noqa: F401  (writes paper/tables/*.tex on import)
    print("Paper tables updated in paper/tables/. Recompile paper/main.tex (e.g. on Overleaf).")


if __name__ == "__main__":
    main()
