"""Stage 1 - Canonicalization.

Produces several *views* of a prompt so later stages analyse what the text
means, not how it is spelled:
  - unicode / homoglyph / zero-width cleanup
  - Devanagari -> Roman transliteration (Hindi and Hinglish share one space)
  - Hinglish phonetic keys (kaise / kese / kaisey -> one key)
  - decoding of embedded encodings (base64) so they are visible to the judge
Each transformation that changes the text is recorded as an obfuscation signal.
"""
from __future__ import annotations

import base64
import re
import unicodedata
from dataclasses import dataclass, field

ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍⁠﻿­"), None)

# Common Cyrillic/Greek look-alikes -> Latin
HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "Α": "A", "Β": "B", "Ε": "E", "Ο": "O",
    "Ρ": "P", "Τ": "T", "Χ": "X", "ο": "o", "ν": "v",
})

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# ---------------- Devanagari transliteration (lightweight, ITRANS-like) -------
_CONS = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh",
    "ज": "j", "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh",
    "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p",
    "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l",
    "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h", "क़": "q", "ख़": "kh",
    "ग़": "g", "ज़": "z", "ड़": "d", "ढ़": "rh", "फ़": "f",
}
_VOW = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ऋ": "ri",
        "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au", "ऑ": "o", "ऍ": "e"}
_MATRA = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "ृ": "ri",
          "े": "e", "ै": "ai", "ो": "o", "ौ": "au"}
_MISC = {"ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o", "ं": "n", "ँ": "n", "ः": "h", "।": ".", "॥": "."}
_VIRAMA = "्"
_NUKTA = "़"


def transliterate_devanagari(text: str) -> str:
    out, chars, i = [], list(unicodedata.normalize("NFC", text)), 0
    while i < len(chars):
        c = chars[i]
        if i + 1 < len(chars) and chars[i + 1] == _NUKTA and (c + _NUKTA) in _CONS:
            c, i = c + _NUKTA, i + 1
        if c in _CONS:
            out.append(_CONS[c])
            nxt = chars[i + 1] if i + 1 < len(chars) else ""
            if nxt in _MATRA:
                out.append(_MATRA[nxt]); i += 1
            elif nxt == _VIRAMA:
                i += 1
            elif nxt and (nxt in _CONS or nxt in _MISC or "ऀ" <= nxt <= "ॿ"):
                out.append("a")
            else:  # word-final schwa deletion (Hindi)
                pass
        elif c in _VOW:
            out.append(_VOW[c])
        elif c in _MATRA:
            out.append(_MATRA[c])
        elif c in _MISC:
            out.append(_MISC[c])
        else:
            out.append(c)
        i += 1
    return "".join(out)


def phonetic_key(word: str) -> str:
    """Spelling-variant-robust key for romanized Hindi/Hinglish words."""
    w = word.lower()
    for a, b in (("chh", "c"), ("ch", "c"), ("kh", "k"), ("gh", "g"), ("bh", "b"),
                 ("ph", "f"), ("th", "t"), ("dh", "d"), ("sh", "s"), ("w", "v"),
                 ("z", "j"), ("q", "k"), ("ck", "k")):
        w = w.replace(a, b)
    w = re.sub(r"(.)\1+", r"\1", w)
    if not w:
        return w
    return w[0] + re.sub(r"[aeiouy]", "", w[1:])


def script_profile(text: str) -> dict[str, int]:
    prof = {"latin": 0, "devanagari": 0, "other": 0}
    for ch in text:
        if not ch.isalpha():
            continue
        if "ऀ" <= ch <= "ॿ":
            prof["devanagari"] += 1
        elif ch.isascii():
            prof["latin"] += 1
        else:
            prof["other"] += 1
    return prof


_B64 = re.compile(r"(?:[A-Za-z0-9+/]{4}){5,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")


def _decode_b64_segments(text: str) -> list[str]:
    found = []
    for m in _B64.finditer(text):
        try:
            dec = base64.b64decode(m.group(0), validate=True).decode("utf-8")
        except Exception:
            continue
        if sum(ch.isprintable() for ch in dec) / max(len(dec), 1) > 0.9:
            found.append(dec)
    return found


@dataclass
class Canonical:
    original: str
    clean: str                         # unicode-cleaned original
    roman: str                         # all scripts romanized + lowercased
    decoded: list[str] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)
    scripts: dict[str, int] = field(default_factory=dict)
    lang_hint: str = "en"

    @property
    def analysis_text(self) -> str:
        """Single string that every downstream detector scans."""
        parts = [self.roman] + [d.lower() for d in self.decoded]
        return "\n".join(parts)

    @property
    def keys(self) -> set[str]:
        return {phonetic_key(t) for t in re.findall(r"[a-z]+", self.analysis_text) if len(t) > 2}


def canonicalize(text: str) -> Canonical:
    signals: list[str] = []
    t = unicodedata.normalize("NFKC", text)
    if t.translate(ZERO_WIDTH) != t:
        signals.append("zero_width_chars")
        t = t.translate(ZERO_WIDTH)
    if t.translate(HOMOGLYPHS) != t and any(c.isascii() and c.isalpha() for c in t):
        signals.append("homoglyphs")
        t = t.translate(HOMOGLYPHS)
    clean = t

    scripts = script_profile(clean)
    total = sum(scripts.values()) or 1
    if scripts["devanagari"] / total > 0.5:
        lang = "hi"
    elif scripts["devanagari"] > 0:
        lang = "hi-en-mixed"
    else:
        lang = "en"
    if 0.15 < scripts["devanagari"] / total < 0.85:
        signals.append("script_mixing")

    roman = transliterate_devanagari(clean).lower()

    # letter-spacing  "w o r d" -> "word"
    despaced = re.sub(r"\b(?:\w[ .\-_]){3,}\w\b", lambda m: re.sub(r"[ .\-_]", "", m.group(0)), roman)
    if despaced != roman:
        signals.append("char_spacing")
        roman = roman + "\n" + despaced

    # leetspeak only counts when digits are glued inside words
    if re.search(r"[a-z][0-9@$][a-z]", roman):
        signals.append("leetspeak")
        roman = roman + "\n" + roman.translate(LEET)

    # Hindi transliteration/Hinglish spelling: long vowels -> short (banaane -> banane)
    collapsed = re.sub(r"aa", "a", re.sub(r"ee", "i", re.sub(r"oo", "u", roman)))
    if collapsed != roman:
        roman = roman + "\n" + collapsed

    decoded = _decode_b64_segments(clean)
    if decoded:
        signals.append("base64_payload")
        decoded = [transliterate_devanagari(d) for d in decoded]

    hing = re.findall(r"\b(kaise|kaisey|kese|kaun|kya|karo|karna|batao|bataiye|mujhe|mujhko|tum|tumhe|aap|hai|hain|hoon|hu|nahi|nahin|kar|mera|meri|chahta|chahti|kyun|kyunki|yeh|woh|accha|bhai|abhi|kab|kahan)\b", roman)
    if lang == "en" and len(hing) >= 2:
        lang = "hinglish"

    return Canonical(text, clean, roman, decoded, signals, scripts, lang)
