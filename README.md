# KAVACH: a plug-in jailbreak defence layer for LLMs (English, Hindi, Hinglish)

KAVACH (Hindi for *armour*) sits between users and **any** LLM. It reads the intent behind each prompt, recognises jailbreak attempts, including ones written in Hindi (Devanagari script), romanized Hindi and code-mixed Hinglish, and responds with a **graded** action rather than a simple allow-or-refuse, so the model stays useful.

```
prompt ─► 1 Canonicalize ─► 2 Tactic score (HOW) + Harm score (WHAT) ─► Router ─► 3 IDP LLM judge ─► 4 Graded policy ─► LLM ─► 5 Output audit
          Devanagari→Roman,                                    rules or       (only when        ALLOW · GUARD · REDIRECT
          phonetic keys, decoding                              fine-tuned     uncertain)        SAFE_REFUSE · BLOCK · SUPPORT
                                                               XLM-R
```

**Main ideas**

1. **Tactic/harm decoupling.** A jailbreak is treated as a harmful *goal* wrapped in an evasion *tactic*, and the two are scored separately. A role-play prompt with a harmless goal is answered; a harmful goal plus a tactic is blocked.
2. **Canonicalize-then-judge.** Devanagari transliteration, Hinglish phonetic keys and de-obfuscation mean one rule set covers three writing forms of Hindi.
3. **Intent Decomposition Prompting (IDP).** The LLM judge translates the prompt, removes the framing, rates the bare core request, and proposes a safe rewrite. The user's text is fenced inside a nonce so it cannot instruct the judge.
4. **Learned router (deep learning).** A fine-tuned XLM-R decides which prompts need the LLM judge.
5. **Model-agnostic plug-in.** Use KAVACH as a Python library, a decorator, or an OpenAI-compatible proxy (change only `base_url`). The live demo UI is included.

## Results (all from public datasets; evidence files in brackets)

| Finding | Number | Evidence |
|---|---|---|
| Decoupling avoids over-refusal: benign real-user prompts wrongly refused | tactic-only detector 11.1% → **KAVACH 0.2%** | `bench/results/test_offline.json` |
| Removing Devanagari transliteration | Hindi routing 64.3% → **0%** | same |
| IDP judge vs naive yes/no judge (Gemini 3.5 Flash-Lite), harmful refused | **81.2%** vs 80.9%, 0% false refusals | `bench/results/test_llm.json`, `run_llm_gemini.log` |
| … on India-specific harms / multi-turn / romanized Hindi | **88.1 / 95.2 / 75.0%** vs 73.8 / 85.7 / 50.0% | same |
| Learned XLM-R router vs rules: harmful routed to judge / benign routed | **95.0% / 15.9%** vs 71.8% / 38.2% | `dl/report/router_results.json` |
| Router trained without India-safety data, tested on it | XLM-R **49.2%**, MuRIL 47.6%, TF-IDF 13.2% | `dl/report/loso_table.tex` |
| Cascade refusal with learned router (vs rules router) | **74.4–79.5%** vs 62.4% | `bench/results/test_llm_router_*.json` |

The transformer training run, with all outputs, is in **`dl/kaggle/kavach_router_kaggle_executed.ipynb`**. The paper is in `paper/main.pdf`.

## Quick start

```powershell
git clone https://github.com/genomecrafter/Kavach.git ; cd Kavach
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[all]"                       # core + server + evaluation
Set-Content .env "GEMINI_API_KEY=your-key"              # free key: aistudio.google.com (never commit .env)
```

### Live demo

```powershell
python -m kavach.server --provider gemini        # reads GEMINI_API_KEY from .env
python -m kavach.server --provider mock          # offline, no key (rules only, echo model)
```
Then open **http://localhost:8000**. On Windows you can instead run
`powershell -ExecutionPolicy Bypass -File scripts\run_demo.ps1` (macOS/Linux: `./scripts/run_demo.sh`). It also installs
`torch` + `transformers` for the XLM-R router if they are missing and opens the browser.

**Router switch (top bar): `Router  Rules | TF-IDF | XLM-R`**

| Option | What decides whether the LLM judge is called | Needs |
|---|---|---|
| **Rules** | hand-written lexical router only | nothing |
| **TF-IDF** | rules **or** char n-gram TF-IDF + logistic regression | `dl/runs/tfidf/` (in git) |
| **XLM-R** | rules **or** the fine-tuned XLM-R transformer | `dl/best_router/model.safetensors` + `pip install torch transformers` |

Switching takes effect from the next message; previous replies keep a "router …" tag showing which router handled them. Each learned router is loaded the first time it is selected (XLM-R takes 5–15 s on a CPU, after which switching is instant). Unavailable options are greyed out, and hovering shows why. The server starts with the best available router (XLM-R, then TF-IDF, then Rules); to force one, pass `--router rules|tfidf|xlmr`.

**Other controls**

- **Mode:** *Cascade* (judge only when the router is unsure), *Always judge*, or *Rules only* (no LLM calls).
- **Compare with unprotected model:** shows the raw model's answer next to KAVACH's.
- **Decision inspector:** click any reply to see every stage (canonical text, tactics, harm score, router probability vs threshold, the judge's core request and safe rewrite, the policy and the output audit).
- **Examples:** benign demo prompts, plus random prompts drawn from the public benchmarks, including multi-turn attacks that play turn by turn.
- **Results tab:** the paper's numbers as charts.

Replies are rendered from Markdown (headings, lists, code, tables). Model output is HTML-escaped first, so it cannot inject scripts.

The same server is an **OpenAI-compatible proxy**:
```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
```

### As a library

```python
from kavach import Kavach, OpenAICompatLLM
from kavach.router import LearnedRouter
llm = OpenAICompatLLM.from_preset("gemini")                      # or groq / openai / ollama
guard = Kavach(llm=llm, judge_llm=llm, mode="cascade", router=LearnedRouter("dl/best_router"))
reply, decision = guard.chat([{"role": "user", "content": "Phishing se apne parivaar ko kaise bachaye?"}])
print(decision.action, decision.trace["router_prob"])
```
More examples are in `examples/quickstart.py`. Run `python -m pytest -q tests` for the 9 unit tests.

## Reproducing the experiments

| Step | Command | Needs |
|---|---|---|
| 1. Build the benchmark (already done, in `bench/data/`) | `bash bench/download.sh && python bench/build_bench.py` | internet |
| 2. Model-free systems + ablations | `python bench/evaluate.py --split test --tag test_offline` | CPU, about 3 min |
| 3. LLM judge experiments (Gemini) | `python bench/run_llm.py` | API key, about 1 h free tier; resumes from cache |
| 4. Train routers (XLM-R, MuRIL, DistilmBERT, TF-IDF) | run `dl/kaggle/kavach_router_kaggle.ipynb` on Kaggle (T4 x2) | GPU, about 45 min |
| 5. Router analysis + figures | `python dl/analyze.py` | CPU |
| 6. Cascade with a learned router | `python dl/cascade_eval.py --router dl/best_router --label xlmr` | cache (no key) |
| 7. Paper tables | `python bench/make_tables.py` | – |

**Reproducibility notes**

- Rules and thresholds were tuned on the 30% dev split; every reported number is on the 70% test split.
- Every Gemini response is cached in `bench/results/llm_cache.sqlite`. With `bench/results/nonce_key.txt`, steps 3 and 6 replay the published run without API calls. That key only reproduces the judge's prompt fence for this cache; a deployment generates a random key at start-up.
- `dl/best_router/model.safetensors` (556 MB) is too large for git. See `dl/best_router/README.md`.

## Repository layout

```
kavach/                 the library
  normalize.py            canonicalization (transliteration, phonetic keys, decoding)
  tactics.py  harm.py     tactic and harm scoring (rules in resources/*.json)
  judge.py                Intent Decomposition Prompting judge + output auditor
  router.py               learned router (transformer or TF-IDF)
  policy.py  memory.py    graded policy; attack memory + conversation risk
  guard.py                the Kavach class
  server.py  web/         OpenAI-compatible proxy + live demo UI
bench/                  benchmark construction, evaluation, Gemini runner, paper-table generator
  data/                   dev/test splits (public datasets)       results/  all result files + Gemini cache
dl/                     deep-learning router
  train_router.py  tfidf_baseline.py  analyze.py  cascade_eval.py  data.py
  kaggle/                 training notebook (template + executed run with outputs)
  runs/  report/  logs/   per-run metrics, figures, training logs
  best_router/            the XLM-R router used by the demo (weights not in git)
paper/                  IEEE paper: main.tex, refs.bib, tables/, figures/, main.pdf
scripts/                demo launchers
examples/  tests/
```

## Putting the repository on GitHub

What is and isn't committed (see `.gitignore`):

| Path | In git? | Why |
|---|---|---|
| `.env` | **never** | holds your API key |
| `bench/raw/` | no | third-party raw datasets; `bench/download.sh` re-downloads them |
| `dl/best_router/model.safetensors` (556 MB) | no | GitHub rejects files over 100 MB; publish it as a Release asset or with Git LFS (below) |
| everything else (code, `bench/data`, results, Gemini cache, Kaggle notebooks, paper) | yes | about 33 MB |

**First push**

```powershell
cd path\to\kavach
git init
git add .
git status            # check: no .env, no model.safetensors in the list
git commit -m "KAVACH: multilingual jailbreak defence layer"
git branch -M main
git remote add origin https://github.com/genomecrafter/Kavach.git
git push -u origin main
```

**Sharing the 556 MB XLM-R model.** Choose one option.

*Option A: GitHub Release (recommended, simplest).* On GitHub go to **Releases → Draft a new release**, use the tag `v1.0`, attach `dl/best_router/model.safetensors` (release assets can be up to 2 GB) and publish. Others download it and place it in `dl/best_router/`. The rest of that folder (config, tokenizer, metrics) is already in git.

*Option B: Git LFS.* This versions the file inside the repo; GitHub's free tier includes 1 GB of LFS storage and 1 GB of bandwidth per month.
```powershell
git lfs install
git lfs track "dl/best_router/model.safetensors"
# delete the line  dl/best_router/model.safetensors  from .gitignore, then:
git add .gitattributes .gitignore dl/best_router/model.safetensors
git commit -m "Add XLM-R router weights via Git LFS"
git push
```

## Limitations

- The LLM evaluation uses a stratified sample of 186 prompts and one judge model.
- Romanized-Hindi test data is machine-transliterated, not written by real users.
- Tamil, Telugu and Bengali have no dedicated canonicalizer; prompts in those scripts go straight to the judge.
- Learned routers are strongest in-distribution; on unseen sources the rules + model union is safest.

## Ethics

The repository contains no newly written attack prompts. All harmful prompts come from existing research datasets (AdvBench, XSTest, MultiJail, jailbreak_llms, inspect-india-evals), and the derived variants only change script and spelling.
