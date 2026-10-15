"""Model-agnostic LLM backends. Anything with `chat(messages, **kw) -> str` works."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from typing import Protocol

import requests

PRESETS = {
    # provider: (base_url, env var for key, default model; None = auto-pick from the provider's model list)
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.1-8b-instant"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY", None),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", "llama3.1"),
}
# Preference order when auto-picking a Gemini model: cheap, fast, highest free-tier quota first.
GEMINI_PREFERENCE = ["flash-lite", "flash"]


class QuotaExhausted(RuntimeError):
    """Raised when the provider keeps answering 429 (e.g. the free daily quota is used up)."""


class LLM(Protocol):
    name: str

    def chat(self, messages: list[dict], **kw) -> str: ...


class _Cache:
    """Tiny on-disk response cache so interrupted / quota-limited runs resume for free."""

    def __init__(self, path: str):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS c (k TEXT PRIMARY KEY, v TEXT)")
        self.lock = threading.Lock()

    def get(self, k):
        with self.lock:
            row = self.db.execute("SELECT v FROM c WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def put(self, k, v):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO c VALUES (?,?)", (k, v))
            self.db.commit()


class OpenAICompatLLM:
    """Works with Groq, Gemini (OpenAI-compat endpoint), OpenAI, Ollama, OpenRouter..."""

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: int = 90, rpm: float | None = None,
                 send_max_tokens: bool = True, cache_path: str | None = None):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout
        self.name = f"{model}"
        self.send_max_tokens = send_max_tokens
        rpm = rpm or float(os.getenv("KAVACH_RPM", "0") or 0)   # client-side throttle for free tiers
        self._gap, self._last = (60.0 / rpm if rpm else 0.0), 0.0
        cache_path = cache_path or os.getenv("KAVACH_CACHE")
        self.cache = _Cache(cache_path) if cache_path else None
        self.calls = 0          # real API calls made (cache hits excluded)

    @classmethod
    def from_preset(cls, provider: str, model: str | None = None, **kw) -> "OpenAICompatLLM":
        url, env, default = PRESETS[provider]
        key = os.getenv(env) or ("ollama" if provider == "ollama" else None)
        if not key:
            raise RuntimeError(f"Set the {env} environment variable for provider '{provider}'.")
        model = model or default or cls.auto_pick(url, key)
        # Gemini "thinking" models spend hidden tokens; a small max_tokens can yield empty answers.
        kw.setdefault("send_max_tokens", provider != "gemini")
        return cls(url, key, model, **kw)

    @staticmethod
    def list_models(base_url: str, api_key: str) -> list[str]:
        r = requests.get(f"{base_url.rstrip('/')}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=30)
        r.raise_for_status()
        return [m["id"].removeprefix("models/") for m in r.json().get("data", [])]

    @classmethod
    def auto_pick(cls, base_url: str, api_key: str) -> str:
        ids = cls.list_models(base_url, api_key)
        chat = [m for m in ids if m.startswith("gemini") and not any(
            x in m for x in ("image", "tts", "embedding", "live", "audio", "transcribe", "translate", "vision", "robotics"))]
        for pref in GEMINI_PREFERENCE:
            cands = [m for m in chat if pref in m and "preview" not in m and "exp" not in m] or \
                    [m for m in chat if pref in m]
            if pref == "flash":
                cands = [m for m in cands if "lite" not in m]
            if cands:
                return sorted(cands, key=_version_key, reverse=True)[0]
        if chat:
            return chat[0]
        raise RuntimeError(f"No Gemini chat model found. Available: {ids[:20]}")

    def chat(self, messages, temperature: float = 0.0, max_tokens: int = 800, json_mode: bool = False, **kw) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature}
        if self.send_max_tokens:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        key = None
        if self.cache:
            key = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        r, waited_429 = None, 0
        for attempt in range(7):
            wait = self._gap - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.api_key}"})
            self.calls += 1
            if r.status_code == 429 or r.status_code >= 500:  # free-tier rate limits / transient errors
                if r.status_code == 429:
                    waited_429 += 1
                time.sleep(min(2 ** attempt * 3, 90))
                continue
            if r.status_code == 400 and "response_format" in body:  # provider without JSON mode
                body.pop("response_format", None)
                continue
            r.raise_for_status()
            choice = (r.json().get("choices") or [{}])[0]
            msg = choice.get("message") or {}          # empty when the provider's own safety filter blocks
            out = msg.get("content") or ""
            if self.cache:
                self.cache.put(key, out)
            return out
        if waited_429:
            raise QuotaExhausted(f"{self.model}: still rate-limited after retries ({r.text[:200]})")
        r.raise_for_status()
        return ""


def _version_key(model_id: str):
    import re
    nums = re.findall(r"\d+(?:\.\d+)?", model_id)
    return [float(n) for n in nums[:2]] or [0.0]


class EchoLLM:
    """Offline stand-in for the protected model (demos/tests without an API key)."""
    name = "echo-mock"

    def chat(self, messages, **kw) -> str:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        last = last.split("\n\n(Reminder from the safety layer")[0]
        return f"[mock model answer to: {last[:120]}]"
