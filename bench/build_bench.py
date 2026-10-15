"""Builds the KAVACH evaluation suite (bench/data/*.jsonl) from public datasets only.

Sources (downloaded by bench/download.sh):
  AdvBench harmful_behaviors (Zou et al. 2023)          harmful, en
  XSTest (Röttger et al. 2024)                          safe look-alikes (over-refusal) + unsafe contrasts, en
  MultiJail (Deng et al. 2024)                          harmful, en + bn
  inspect-india-evals safety.csv (Apache-2.0)           harmful, en/hi/ta/te/bn, India-specific
  inspect-india-evals jailbreak_multilingual.csv        multi-turn jailbreaks, 7 Indian languages
  inspect-india-evals dpi_dataset.csv                   India DPI prompts, answer vs refuse, en
  inspect-india-evals mmlu_translated.csv               benign questions, en/hi/mr/ta/te/bn/gu
  jailbreak_llms (Shen et al. CCS'24)                   in-the-wild jailbreak templates + regular prompts

Derived (programmatic, no new content is written):
  hi-Latn   : Hindi items romanized with kavach's transliterator   (both harmful AND benign)
  hi-Latn-noisy : + seeded phonetic spelling noise, mimicking real Hinglish spelling variation
"""
import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

from kavach.normalize import transliterate_devanagari

RNG = random.Random(13)


def phonetic_noise(s: str, p: float = 0.35) -> str:
    rules = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("sh", "s"), ("v", "w"), ("ai", "e"),
             ("kh", "k"), ("ph", "f"), ("z", "j"), ("e ", "ey "), ("i ", "ee ")]
    out = s
    for a, b in rules:
        if a in out and RNG.random() < p:
            out = out.replace(a, b)
    return out


def rows(raw: Path):
    R = []
    add = lambda **k: R.append(k)

    adv = pd.read_csv(raw / "harmful_behaviors.csv")
    for i, g in enumerate(adv.goal):
        add(id=f"adv-{i}", source="advbench", text=g, label=1, lang="en", kind="direct")

    xs = pd.read_csv(raw / "xstest_prompts.csv")
    for _, r in xs.iterrows():
        add(id=f"xs-{r.id}", source=f"xstest-{r.label}", text=r.prompt, label=int(r.label == "unsafe"),
            lang="en", kind="direct", meta=r.type)

    mj = pd.read_csv(raw / "MultiJail.csv")
    for _, r in mj.iterrows():
        for lang in ("en", "bn"):
            add(id=f"mj-{r.id}-{lang}", source="multijail", text=str(r[lang]), label=1, lang=lang, kind="direct")

    ie = raw / "india_evals"
    sf = pd.read_csv(ie / "safeguards/datasets/safety.csv")
    for _, r in sf.iterrows():
        for col, lang in (("original_prompt", "en"), ("Hindi", "hi"), ("Tamil", "ta"), ("Telugu", "te"), ("Bengali", "bn")):
            add(id=f"ind-{r.id}-{lang}", source="india-safety", text=str(r[col]), label=1, lang=lang, kind="direct",
                meta=r.category)
        roman = transliterate_devanagari(str(r.Hindi))
        add(id=f"ind-{r.id}-hiL", source="india-safety", text=roman, label=1, lang="hi-Latn", kind="direct", meta=r.category)
        add(id=f"ind-{r.id}-hiLn", source="india-safety", text=phonetic_noise(roman), label=1, lang="hi-Latn-noisy",
            kind="direct", meta=r.category)

    dpi = pd.read_csv(ie / "dpi_safety/datasets/dpi_dataset.csv")
    for _, r in dpi.iterrows():
        add(id=f"dpi-{r.Id}", source="india-dpi", text=r.Prompt, label=int(r["Expected Action"] == "Refuse"),
            lang="en", kind="direct", meta=r.Category)

    mm = pd.read_csv(ie / "multilingual/datasets/mmlu_translated.csv")
    for _, r in mm.iterrows():
        add(id=f"mmlu-{r.id}-{r.language}", source="mmlu-benign", text=str(r.question), label=0, lang=r.language, kind="direct")
        if r.language == "hi":
            roman = transliterate_devanagari(str(r.question))
            add(id=f"mmlu-{r.id}-hiL", source="mmlu-benign", text=roman, label=0, lang="hi-Latn", kind="direct")
            add(id=f"mmlu-{r.id}-hiLn", source="mmlu-benign", text=phonetic_noise(roman), label=0, lang="hi-Latn-noisy", kind="direct")

    jm = pd.read_csv(ie / "safeguards/datasets/jailbreak_multilingual.csv")
    for _, r in jm.iterrows():
        turns = [str(r[f"turn{k}"]) for k in range(1, 6) if isinstance(r[f"turn{k}"], str) and r[f"turn{k}"].strip()]
        add(id=f"mt-{r.id}-{r.language}", source="india-multiturn", text=turns[-1], turns=turns, label=1,
            lang=r.language, kind="multiturn", meta=r.attack_type)

    jb = pd.read_csv(raw / "jailbreak_prompts_2023_12_25.csv").drop_duplicates("prompt")
    for i, t in enumerate(jb.prompt.sample(min(600, len(jb)), random_state=1)):
        add(id=f"wild-jb-{i}", source="wild-jailbreak", text=str(t)[:6000], label=1, lang="en", kind="template")
    reg = pd.read_csv(raw / "regular_prompts_2023_12_25.csv", usecols=["prompt", "platform"]).drop_duplicates("prompt")
    reg = reg[reg.prompt.str.len().between(20, 4000)]
    for i, (_, r) in enumerate(reg.sample(1500, random_state=1).iterrows()):
        add(id=f"wild-reg-{i}", source="wild-regular", text=str(r.prompt), label=0, lang="en", kind="direct",
            meta=str(r.platform))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="bench/raw")
    ap.add_argument("--out", default="bench/data")
    a = ap.parse_args()
    R = rows(Path(a.raw))
    # 30 % dev (for any threshold tuning) / 70 % test (reported). Split by underlying item id so that
    # translations of the same prompt never straddle the split.
    for r in R:
        base = re.sub(r"-(en|hi|ta|te|bn|mr|gu|hiL|hiLn)$", "", r["id"])
        r["split"] = "dev" if random.Random(base).random() < 0.3 else "test"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in R:
                if r["split"] == split:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
    df = pd.DataFrame(R)
    print(df.groupby(["split", "source", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
