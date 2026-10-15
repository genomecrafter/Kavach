#!/usr/bin/env bash
# Downloads the public datasets used by the KAVACH benchmark into bench/raw/
set -e
RAW="$(dirname "$0")/raw"; mkdir -p "$RAW"; cd "$RAW"
curl -sSLO https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv
curl -sSLO https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv
curl -sSLO https://raw.githubusercontent.com/DAMO-NLP-SG/multilingual-safety-for-LLMs/main/data/MultiJail.csv
curl -sSLO https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/jailbreak_prompts_2023_12_25.csv
curl -sSLO https://raw.githubusercontent.com/verazuo/jailbreak_llms/main/data/prompts/regular_prompts_2023_12_25.csv
pip download inspect-india-evals==0.1.0 --no-deps -d . -q
python - << 'EOF'
import zipfile, glob
z = zipfile.ZipFile(glob.glob("inspect_india_evals-*.whl")[0])
for n in z.namelist():
    if n.startswith("india_evals/") and n.endswith(".csv"):
        z.extract(n, ".")
EOF
echo "done -> $(pwd)"
