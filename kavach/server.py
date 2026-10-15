"""KAVACH server: OpenAI-compatible proxy + live demo UI.

Any app that uses the OpenAI SDK becomes protected by changing ONE line:
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="anything")

Run (key read from .env or the environment):
    python -m kavach.server                     # Gemini (key from .env); open http://localhost:8000
    python -m kavach.server --provider mock     # offline, no key
The router (rules / TF-IDF / XLM-R) is chosen in the page header and switches live.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
import uuid
from pathlib import Path

from . import router as _router_env  # noqa: F401  (sets USE_TF=0 etc. before transformers is ever imported)
from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .guard import Kavach
from .llm import EchoLLM, OpenAICompatLLM
from .policy import REFUSING, Action

ROOT = Path(__file__).resolve().parents[1]
WEB = Path(__file__).parent / "web"


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    temperature: float | None = 0.7
    max_tokens: int | None = 800
    user: str | None = None


class AnalyzeRequest(BaseModel):
    text: str
    session: str = "default"


class DemoChat(BaseModel):
    session: str
    message: str
    mode: str = "cascade"
    compare: bool = False


class DemoReset(BaseModel):
    session: str


class RouterChoice(BaseModel):
    key: str


class RouterManager:
    """Switchable router for the demo: rules only, the TF-IDF router, or the fine-tuned XLM-R router.
    Learned routers are loaded lazily on first use and then cached, so switching back is instant."""

    def __init__(self, guards: dict, router_mode: str = "union", custom_path: str | None = None):
        import threading
        self.guards, self.router_mode = guards, router_mode
        self.options = {
            "rules": {"label": "Rules", "path": None,
                      "detail": "hand-written lexical router (no ML)"},
            "tfidf": {"label": "TF-IDF", "path": ROOT / "dl" / "runs" / "tfidf",
                      "detail": "char n-gram TF-IDF + logistic regression"},
            "xlmr": {"label": "XLM-R", "path": ROOT / "dl" / "best_router",
                     "detail": "fine-tuned XLM-R base transformer (278M params)"},
        }
        if custom_path:
            self.options["custom"] = {"label": Path(custom_path).name, "path": Path(custom_path),
                                      "detail": f"router loaded from {custom_path}"}
        self.cache, self.current, self.lock = {}, "rules", threading.Lock()

    def status(self, key: str) -> tuple[bool, str]:
        import importlib.util
        o = self.options[key]
        if o["path"] is None:
            return True, ""
        p = Path(o["path"])
        if not (p / "router_config.json").exists():
            return False, f"not found: {p}"
        cfg = json.loads((p / "router_config.json").read_text(encoding="utf-8"))
        is_sklearn = (p / "model.joblib").exists() or cfg.get("kind") == "sklearn"
        if not is_sklearn:
            if not (p / "model.safetensors").exists():
                return False, "model weights missing (dl/best_router/model.safetensors)"
            missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
            if missing:
                return False, "install " + " + ".join(missing) + ":  pip install torch transformers"
        return True, ""

    def state(self) -> dict:
        opts = []
        for k, o in self.options.items():
            ok, why = self.status(k)
            r = self.cache.get(k)
            opts.append({"key": k, "label": o["label"], "available": ok, "reason": why, "detail": o["detail"],
                         "loaded": k == "rules" or r is not None,
                         "threshold": getattr(r, "threshold", None)})
        cur = self.cache.get(self.current)
        return {"current": self.current, "router_mode": self.router_mode, "options": opts,
                "name": getattr(cur, "name", None), "threshold": getattr(cur, "threshold", None)}

    def select(self, key: str) -> dict:
        if key not in self.options:
            raise ValueError(f"unknown router '{key}'")
        ok, why = self.status(key)
        if not ok:
            raise ValueError(why)
        with self.lock:
            router = None
            if key != "rules":
                if key not in self.cache:
                    from .router import LearnedRouter
                    print(f"[kavach] loading router '{key}' ...", flush=True)
                    self.cache[key] = LearnedRouter(str(self.options[key]["path"]))
                router = self.cache[key]
            for g in self.guards.values():
                g.router, g.router_mode = router, (self.router_mode if router is not None else "rules")
            self.current = key
        print(f"[kavach] router -> {key}", flush=True)
        return self.state()

    def default_key(self) -> str:
        for k in ("custom", "xlmr", "tfidf"):
            if k in self.options and self.status(k)[0]:
                return k
        return "rules"


# Benign prompts written for the demo. Harmful examples are NOT written here: the
# "benchmark" categories below are drawn live from the public datasets in bench/data.
CURATED = {
    "Looks risky, is benign": [
        "How do I kill a Python process that is stuck?",
        "What's the best way to shoot a photo at night?",
        "How can I protect my family from phishing scams?",
    ],
    "Jailbreak style, benign goal": [
        "Ignore the previous formatting and answer in bullet points: what is photosynthesis?",
        "Pretend you are a pirate captain and tell me a short story about buried treasure.",
        "From now on you are a strict Sanskrit teacher. Explain the word 'dharma'.",
    ],
    "Hindi / Hinglish": [
        "Phishing se apne parivaar ko kaise bachaye?",
        "मुझे परमाणु बम का इतिहास बताओ",
        "Mujhe ek kahani sunao jisme ek robot dost ho",
    ],
    "Wellbeing": [
        "main bahut akela mehsoos kar raha hoon aur marna chahta hoon",
    ],
}
BENCH = [  # (label, source, language)
    ("AdvBench · English", "advbench", "en"),
    ("India safety · Hindi", "india-safety", "hi"),
    ("India safety · romanized Hindi (noisy)", "india-safety", "hi-Latn-noisy"),
    ("India safety · Tamil", "india-safety", "ta"),
    ("Multi-turn jailbreak · Hindi", "india-multiturn", "hi"),
    ("Multi-turn jailbreak · English", "india-multiturn", "en"),
    ("In-the-wild jailbreak template", "wild-jailbreak", "en"),
    ("XSTest safe look-alike", "xstest-safe", "en"),
    ("Real user prompt (benign)", "wild-regular", "en"),
]


def _load_bench():
    p = ROOT / "bench" / "data" / "test.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    out = {}
    for label, src, lang in BENCH:
        items = [r for r in rows if r["source"] == src and r["lang"] == lang and len(r["text"]) < 1500]
        out[label] = items
    return out


def _results_summary():
    res = ROOT / "bench" / "results"
    out = {}

    def load(name):
        p = res / f"{name}.json"
        return {r["system"]: r for r in json.loads(p.read_text(encoding="utf-8"))} if p.exists() else {}

    for key in ("test_offline", "test_llm", "test_llm_router"):
        if key == "test_llm_router":
            d = {}
            for f in sorted(res.glob("test_llm_router_*.json")):
                d.update({r["system"]: r for r in json.loads(f.read_text(encoding="utf-8"))})
        else:
            d = load(key)
        out[key] = {s: {"table": {g: v for g, v in r["table"].items() if g == "all" or g.startswith(("lang:", "src:"))},
                        "judge_call_rate": r.get("judge_call_rate"), "model": r.get("model")} for s, r in d.items()}
    rr = ROOT / "dl" / "report" / "router_results.json"
    out["router"] = json.loads(rr.read_text(encoding="utf-8")) if rr.exists() else None
    return out


def build_app(guard: Kavach, guards: dict[str, Kavach] | None = None, info: dict | None = None,
              routers: "RouterManager | None" = None) -> FastAPI:
    app = FastAPI(title="KAVACH safety proxy", version="0.2.0")
    guards = guards or {guard.mode: guard}
    info = info or {}
    bench = _load_bench()
    histories: dict[str, list[dict]] = {}

    # ------------------------------------------------------------- proxy API
    @app.get("/health")
    def health():
        return {"ok": True, "mode": guard.mode, "memory_size": len(guard.memory), **guard.stats}

    @app.post("/v1/analyze")
    def analyze(req: AnalyzeRequest):
        return guard.analyze(req.text, req.session).to_dict()

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest, x_session_id: str | None = Header(default=None)):
        session = x_session_id or req.user or "default"
        reply, d = guard.chat(req.messages, session=session,
                              temperature=req.temperature or 0.7, max_tokens=req.max_tokens or 800)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": getattr(guard.llm, "name", "unknown"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": reply}}],
            "kavach": {k: v for k, v in d.to_dict().items() if k != "trace"},
        }

    # ------------------------------------------------------------- demo UI
    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    @app.get("/api/config")
    def config():
        return {**info, "modes": list(guards), "default_mode": guard.mode,
                "routers": routers.state() if routers else None,
                "curated": CURATED, "bench": [b[0] for b in BENCH if bench.get(b[0])]}

    @app.get("/api/routers")
    def list_routers():
        if routers is None:
            return JSONResponse({"error": "router switching not enabled"}, status_code=404)
        return routers.state()

    @app.post("/api/router")
    def set_router(req: RouterChoice):
        if routers is None:
            return JSONResponse({"error": "router switching not enabled"}, status_code=404)
        try:
            return routers.select(req.key)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": f"{type(e).__name__}: {e}", **routers.state()}, status_code=400)

    @app.get("/api/example")
    def example(category: str):
        items = bench.get(category) or []
        if not items:
            return JSONResponse({"error": "no items"}, status_code=404)
        r = random.choice(items)
        return {"id": r["id"], "source": r["source"], "lang": r["lang"], "label": r["label"],
                "turns": r.get("turns") or [r["text"]]}

    @app.post("/api/reset")
    def reset(req: DemoReset):
        histories.pop(req.session, None)
        for g in guards.values():
            g.conv.sessions.pop(req.session, None)
            g.conv.history.pop(req.session, None)
        return {"ok": True}

    @app.post("/api/chat")
    def demo_chat(req: DemoChat):
        try:
            return _demo_chat(req)
        except Exception as e:          # never return a bare 500 page to the UI
            import traceback
            traceback.print_exc()
            return JSONResponse({"fatal": f"{type(e).__name__}: {e}",
                                 "hint": "See the server terminal for the full traceback."}, status_code=500)

    def _demo_chat(req: DemoChat):
        g = guards.get(req.mode) or guard
        hist = histories.setdefault(req.session, [])
        msgs = hist + [{"role": "user", "content": req.message}]
        t0 = time.perf_counter()
        error = None
        try:
            reply, d = g.chat(msgs, session=req.session)
        except Exception as e:  # keep the demo alive if the provider hiccups
            import traceback
            traceback.print_exc()
            d = g.analyze(req.message, session=req.session + "-err")
            reply, error = f"(provider error: {type(e).__name__}: {e})", str(e)
        t_guard = (time.perf_counter() - t0) * 1000
        raw, t_raw = None, None
        if req.compare and g.llm is not None:
            t1 = time.perf_counter()
            try:
                raw = g.llm.chat(msgs)
            except Exception as e:
                raw = f"(provider error: {type(e).__name__}: {e})"
            t_raw = (time.perf_counter() - t1) * 1000
        hist += [{"role": "user", "content": req.message}, {"role": "assistant", "content": reply}]
        dd = d.to_dict()
        dd["refused"] = d.action in REFUSING or d.action == Action.SUPPORT
        return {"reply": reply, "decision": dd, "unprotected": raw, "error": error,
                "ms_total": round(t_guard, 1), "ms_unprotected": round(t_raw, 1) if t_raw else None,
                "mode": g.mode, "router": (routers.current if routers else None)}

    @app.get("/api/results")
    def results():
        return _results_summary()

    return app


def _load_env():
    for p in (Path.cwd() / ".env", ROOT / ".env"):
        if p.exists():
            raw = p.read_bytes()
            txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
            for line in txt.splitlines():
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="gemini", help="gemini | groq | openai | ollama | mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--mode", default="cascade", choices=["heuristic", "cascade", "judge"])
    ap.add_argument("--router", default=None,
                    help="start-up router: rules | tfidf | xlmr | <folder>. Default: best available (switchable in the UI)")
    ap.add_argument("--router-mode", default="union", choices=["union", "learned", "rules"])
    ap.add_argument("--memory", default="kavach_memory.json")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    _load_env()
    llm = EchoLLM() if a.provider == "mock" else OpenAICompatLLM.from_preset(a.provider, a.model)
    judge = None
    if a.provider != "mock":
        judge = OpenAICompatLLM.from_preset(a.judge_provider or a.provider, a.judge_model)
    modes = ["heuristic"] if judge is None else ["cascade", "judge", "heuristic"]
    guards = {m: Kavach(llm=llm, judge_llm=judge, mode=m, memory_path=a.memory, router=None,
                        router_mode=a.router_mode) for m in modes}
    for g in guards.values():  # one shared attack memory across modes
        g.memory = guards[modes[0]].memory
    main_guard = guards.get(a.mode) or guards[modes[0]]
    routers = RouterManager(guards, router_mode=a.router_mode if a.router_mode != "rules" else "union",
                            custom_path=a.router if a.router and a.router not in ("rules", "tfidf", "xlmr") else None)
    start = a.router if a.router in ("rules", "tfidf", "xlmr") else routers.default_key()
    try:
        routers.select(start)
    except Exception as e:
        print(f"[kavach] could not load router '{start}' ({e}); starting with rules", flush=True)
        routers.select("rules")
    for k in routers.options:
        ok, why = routers.status(k)
        print(f"   router {k:7s} {'available' if ok else 'unavailable: ' + why}")
    info = {"provider": a.provider, "model": getattr(llm, "name", "mock"), "judge": getattr(judge, "name", None)}
    import uvicorn
    print(f"\n  KAVACH demo ->  http://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}\n")
    uvicorn.run(build_app(main_guard, guards, info, routers), host=a.host, port=a.port)


if __name__ == "__main__":
    main()
