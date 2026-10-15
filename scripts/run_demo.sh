#!/usr/bin/env bash
# KAVACH live demo launcher (macOS / Linux). The router (Rules / TF-IDF / XLM-R) is chosen in the web UI.
#   ./scripts/run_demo.sh          # Gemini (key from .env)
#   ./scripts/run_demo.sh --mock   # offline, no key
set -e
cd "$(dirname "$0")/.."
PROVIDER=gemini; [ "$1" = "--mock" ] && PROVIDER=mock
if [ -f dl/best_router/model.safetensors ] && ! python -c "import torch, transformers" 2>/dev/null; then
  echo "Installing torch + transformers for the XLM-R router (one time)..."; python -m pip install torch transformers
fi
echo "Open http://localhost:8000 once the server prints its address."
python -m kavach.server --provider "$PROVIDER"
