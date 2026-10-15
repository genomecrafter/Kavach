"""Stage 2b - Harm-intent pre-scoring (the *what* of a prompt).

Cheap, model-free, language-normalized scoring used to (a) decide fast cases
and (b) gate when the LLM intent judge must be called. It matches on the
canonical romanized view plus Hinglish phonetic keys, so spelling variants
("kaise / kese / kaisey") and Devanagari input are covered by one lexicon.

The score combines:
  topic severity     - which harm categories are mentioned
  operational intent - is the user asking HOW to do it (uplift)
  protective context - prevention / history / awareness framing (dampens)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import Canonical, phonetic_key

_SPEC = json.loads((Path(__file__).parent / "resources" / "harm_lexicon.json").read_text(encoding="utf-8"))


def _term_matcher(term: str):
    words = term.split()
    if len(words) == 1 and term.isalpha() and len(phonetic_key(term)) >= 4:
        return ("key", phonetic_key(term))
    return ("re", re.compile(r"\b" + r"\s+".join(map(re.escape, words)) + r"\b"))


_CATS = {k: (v["severity"], [(t, _term_matcher(t)) for t in v["terms"]]) for k, v in _SPEC["categories"].items()}
_OPS = [re.compile(r"\b" + re.escape(m)) for m in _SPEC["operational_markers"]]
_CUES_RE = [re.compile(r"\b" + re.escape(t)) for t in _SPEC["routing_cues"]["en"] + _SPEC["routing_cues"]["hi"]]
_CUE_KEYS = sorted({phonetic_key(t) for t in _SPEC["routing_cues"]["hi"] if len(phonetic_key(t)) >= 4})
_PROT = [re.compile(r"\b" + re.escape(m)) for m in _SPEC["protective_markers"]]


@dataclass
class HarmAssessment:
    score: float
    categories: dict[str, int] = field(default_factory=dict)
    operational: list[str] = field(default_factory=list)
    protective: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    cues: list[str] = field(default_factory=list)


def assess_harm(c: Canonical) -> HarmAssessment:
    text, keys = c.analysis_text, c.keys
    cats, terms = {}, []
    for cat, (sev, matchers) in _CATS.items():
        for term, (kind, m) in matchers:
            if (kind == "key" and m in keys) or (kind == "re" and m.search(text)):
                cats[cat] = sev
                terms.append(term)
                break
    ops = [p.pattern[2:].replace("\\", "") for p in _OPS if p.search(text)]
    prot = [p.pattern[2:].replace("\\", "") for p in _PROT if p.search(text)]

    cues = [p.pattern[2:].replace("\\", "") for p in _CUES_RE if p.search(text)]
    if c.lang_hint != "en":   # Hindi phonetic keys collide with English words, so use them only on Hindi text
        cues += [k for k in _CUE_KEYS if any(w.startswith(k) for w in keys)]
    cues = sorted(set(cues))
    if not cats:
        # routing-only signal: high enough to escalate to the judge, never enough to refuse
        score = min(0.20 + 0.03 * (len(cues) - 1), 0.28) if cues else 0.0
        return HarmAssessment(score, {}, ops, prot, [], cues)
    sev = max(cats.values())
    base = {1: 0.25, 2: 0.40, 3: 0.55}[sev]
    if ops:
        base += 0.30
    if prot:
        base -= 0.30 if not ops else 0.15
    base += 0.05 * (len(cats) - 1)
    base = max(base, 0.20)          # a harm topic is always worth a judge look, even in protective framing
    return HarmAssessment(round(min(max(base, 0.0), 1.0), 4), cats, ops, prot, terms, cues)
