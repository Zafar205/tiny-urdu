#!/usr/bin/env bash
# One-time setup: venv, tiny-voca (pinned), aligner model, English starting weights.
# Needs ~3 GB of disk. Safe to re-run.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3.12}"
TINY_VOCA_REV=ca93ad1e35a834a9247347c6821cfeb4c82c6a96

[ -d .venv ] || "$PYTHON" -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip

mkdir -p vendor
if [ ! -d vendor/tiny-voca ]; then
  git clone https://github.com/n4ze3m/tiny-voca vendor/tiny-voca
fi
git -C vendor/tiny-voca checkout -q "$TINY_VOCA_REV"

# annotate: forced aligner deps. prepare: librosa/pyarrow. We use our own front end,
# so the "own" extra (dizione) and eSpeak are NOT needed.
pip install -q -e "vendor/tiny-voca[prepare,annotate]" datasets scipy

# Forced aligner (annotate runs with local_files_only=True) and starting weights.
hf download facebook/wav2vec2-lv-60-espeak-cv-ft >/dev/null
hf download n4ze3m/tiny-voca tiny-voca.pt config.json --local-dir weights >/dev/null

echo "setup done. Next: ./pipeline.sh"
