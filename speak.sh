#!/usr/bin/env bash
# ./speak.sh "aap kaise hain?" [out.wav] [runs/urdu1/inference.pt]
# Add --voice M2 etc. via VOICE=M2 (use the slot prepare_data.py printed).
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
python urdu/run.py tiny_voca.infer \
  --checkpoint "${3:-runs/urdu1/inference.pt}" \
  --voice "${VOICE:-$(python -c 'import json;print(json.loads(open("data/train.jsonl").readline())["speaker"])')}" \
  --text "$1" --output "${2:-out.wav}"
echo "wrote ${2:-out.wav}"
