"""End-to-end cascade with the LEARNED router, re-using the Gemini answers cached by bench/run_llm.py.

    python dl/cascade_eval.py --router dl/runs/xlmr            (same 186-prompt sample as run_llm.py)

Judge calls in cascade mode are byte-identical to those of the always-judge run, so they are served
from bench/results/llm_cache.sqlite: normally 0 new API calls, and no API key is needed.
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
env_file = ROOT / ".env"
if env_file.exists():
    raw = env_file.read_bytes()
    txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    for line in txt.splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
key_file = RES / ".nonce_key"
if not key_file.exists() and (RES / "nonce_key.txt").exists():   # some upload tools drop dot-files
    key_file = RES / "nonce_key.txt"
if not key_file.exists():
    sys.exit("bench/results/.nonce_key not found: copy the bench/results folder from the machine that ran run_llm.py")
os.environ.setdefault("KAVACH_NONCE_KEY", key_file.read_text(encoding="utf-8").strip())
os.environ.setdefault("KAVACH_CACHE", str(RES / "llm_cache.sqlite"))
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "bench"))

import evaluate as E  # noqa: E402
from kavach.llm import OpenAICompatLLM  # noqa: E402
from kavach.router import LearnedRouter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--router", required=True, help="folder of a trained router (dl/runs/<name>)")
    ap.add_argument("--model", default=None, help="judge model used by run_llm.py (default: read from test_llm.json)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--threshold", type=float, default=None, help="override the router's validation threshold")
    ap.add_argument("--label", default=None, help="short name for this router in tables (default: folder name)")
    a = ap.parse_args()
    prev = json.loads((RES / "test_llm.json").read_text(encoding="utf-8"))
    model = a.model or prev[0].get("model")
    os.environ.setdefault("GEMINI_API_KEY", "cache-only")
    llm = OpenAICompatLLM.from_preset("gemini", model)
    router = LearnedRouter(a.router, threshold=a.threshold)
    print(f"judge={model}  router={router.name} (threshold {router.threshold:.3f})  device={router.device}")
    data = E.load("test", a.n)
    label = a.label or Path(a.router).name
    systems = {
        f"cascade-learned-{label}": E.make_kavach_runner("full", judge_llm=llm, mode="cascade", router=router, router_mode="learned"),
        f"cascade-union-{label}": E.make_kavach_runner("full", judge_llm=llm, mode="cascade", router=router, router_mode="union"),
    }
    results = []
    for name, run in systems.items():
        print("running", name, flush=True)
        results.append(E.evaluate(run, data, name))
    for r in results:
        r["model"] = model
        r["router"] = router.name
    E.print_table(results, ["all", "lang:en", "lang:hi", "lang:hi-Latn", "lang:hi-Latn-noisy", "src:india-safety",
                            "src:india-multiturn"])
    print(f"new API calls: {llm.calls} (0 means everything came from the cache)")
    for r in results:
        r["router_label"] = label
    (RES / f"test_llm_router_{label}.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved bench/results/test_llm_router_{label}.json")


if __name__ == "__main__":
    main()
