"""Adaptive components.

ConversationRisk - decayed risk accumulator per session. Catches gradual
  multi-turn escalation where each turn alone looks harmless.
AttackMemory - "adaptive immunity": canonical forms of prompts that the judge
  CONFIRMED as jailbreaks are stored; future prompts that are near-duplicates
  (character n-gram cosine, robust to Hinglish spelling and paraphrase noise)
  get their risk raised without any retraining. Starts empty: no dataset needed.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

_VEC = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=2 ** 18,
                         alternate_sign=False, norm="l2")


class ConversationRisk:
    def __init__(self, decay: float = 0.6):
        self.decay = decay
        self.sessions: dict[str, float] = {}
        self.history: dict[str, list[str]] = {}

    def update(self, session: str, turn_risk: float, text: str) -> float:
        prev = self.sessions.get(session, 0.0)
        cum = prev * self.decay + turn_risk
        self.sessions[session] = cum
        self.history.setdefault(session, []).append(text[:500])
        return cum

    def get(self, session: str) -> float:
        return self.sessions.get(session, 0.0)

    def turns(self, session: str) -> list[str]:
        return self.history.get(session, [])


class AttackMemory:
    def __init__(self, path: str | None = None, threshold: float = 0.55, max_items: int = 5000):
        self.path = Path(path) if path else None
        self.threshold, self.max_items = threshold, max_items
        self.items: list[str] = []
        self._mat = None
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self.items = json.loads(self.path.read_text(encoding="utf-8"))
            self._rebuild()

    def _rebuild(self):
        self._mat = _VEC.transform(self.items) if self.items else None

    def add(self, canonical_text: str):
        with self._lock:
            self.items.append(canonical_text[:2000])
            self.items = self.items[-self.max_items:]
            self._rebuild()
            if self.path:
                self.path.write_text(json.dumps(self.items, ensure_ascii=False), encoding="utf-8")

    def similarity(self, canonical_text: str) -> float:
        if self._mat is None:
            return 0.0
        v = _VEC.transform([canonical_text[:2000]])
        return float(np.max((self._mat @ v.T).toarray()))

    def __len__(self):
        return len(self.items)
