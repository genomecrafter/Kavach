"""Stage 2a - Evasion-tactic detection (the *how* of a prompt).

KAVACH's core idea is Tactic/Intent decoupling: a jailbreak = (underlying goal)
+ (evasion tactic). This module scores ONLY the tactic axis and is deliberately
blind to topic, so a role-play story about dragons scores high on tactic but
low on harm, and is not refused.

Patterns run on the canonical (romanized, decoded) view, so Hindi written in
Devanagari and Hinglish written in Latin script hit the same rules.
Patterns are data: extend `kavach/resources/tactics.json` without code changes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .normalize import Canonical

_RES = Path(__file__).parent / "resources" / "tactics.json"


@dataclass
class TacticHit:
    tactic: str
    weight: float
    evidence: str


def _load():
    spec = json.loads(_RES.read_text(encoding="utf-8"))
    return {k: (v["weight"], [re.compile(p, re.I) for p in v["patterns"]]) for k, v in spec.items()}


_TACTICS = _load()

# Obfuscation signals from the canonicalizer are tactics too.
_SIGNAL_WEIGHTS = {
    "base64_payload": 0.55, "leetspeak": 0.35, "char_spacing": 0.35,
    "homoglyphs": 0.45, "zero_width_chars": 0.45, "script_mixing": 0.10,
}


def detect_tactics(c: Canonical) -> tuple[float, list[TacticHit]]:
    text = c.analysis_text
    hits: list[TacticHit] = []
    for name, (w, pats) in _TACTICS.items():
        for p in pats:
            m = p.search(text)
            if m:
                hits.append(TacticHit(name, w, m.group(0)[:80]))
                break
    for s in c.signals:
        if s in _SIGNAL_WEIGHTS:
            hits.append(TacticHit(f"obfuscation:{s}", _SIGNAL_WEIGHTS[s], s))
    # noisy-OR fusion: independent weak signals accumulate
    p_none = 1.0
    for h in hits:
        p_none *= (1 - h.weight)
    return round(1 - p_none, 4), hits
