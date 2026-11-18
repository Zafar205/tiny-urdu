#!/usr/bin/env bash
# data -> acoustic features -> forced-alignment annotations -> fine-tune.
# Every stage is skipped if its output exists, so re-running resumes.
#
#   ./pipeline.sh                      # 1000 clips, device auto-detected
#   DEVICE=cuda STEPS=8000 ./pipeline.sh
#   N_TRAIN=200 STEPS=300 ./pipeline.sh    # quick smoke test
set -euo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate

N_TRAIN="${N_TRAIN:-1000}"
N_VAL="${N_VAL:-32}"
STEPS="${STEPS:-6000}"
RUN="${RUN:-runs/urdu1}"
DEVICE="${DEVICE:-$(python -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")')}"
RUN_PY="python urdu/run.py"
echo "device=$DEVICE train=$N_TRAIN steps=$STEPS run=$RUN"

# 1. Stream + resample the clips, write data/{train,validation}.jsonl
[ -f data/validation.jsonl ] || python urdu/prepare_data.py --train "$N_TRAIN" --validation "$N_VAL" --output data

# 2. Acoustic features (waveform, mel, pitch, energy) at 24 kHz
[ -f features/train.preparation.json ] || \
  $RUN_PY tiny_voca.prepare --inputs data/train.jsonl --split train --output features --speaker-column speaker
[ -f features/validation.preparation.json ] || \
  $RUN_PY tiny_voca.prepare --inputs data/validation.jsonl --split validation --output features --speaker-column speaker

# 3. Phoneme tokens + per-phoneme durations (wav2vec2 forced alignment)
[ -f annotations/annotations.json ] || \
  $RUN_PY tiny_voca.annotate --features features --output annotations --splits train validation \
    --dataset-revision UrduTTS-streamed --device "$DEVICE"

# 4. English tiny-voca -> step-0 training state
[ -f runs/init.pt ] || python urdu/make_init_checkpoint.py --weights weights/tiny-voca.pt --output runs/init.pt

# 5. Fine-tune. Schedule is scaled down from the 104k-step English recipe.
GAN_START=$((STEPS / 2))
$RUN_PY tiny_voca.train \
  --train annotations/train.jsonl --validation annotations/validation.jsonl \
  --output "$RUN" --init-from runs/init.pt --new-data \
  --device "$DEVICE" --steps "$STEPS" --max-seconds 86400 \
  --learning-rate 2e-4 --warmup 200 \
  --gan-start "$GAN_START" --gan-ramp $((STEPS / 8)) \
  --teacher-hold $((STEPS / 6)) --teacher-ramp $((STEPS / 3)) --teacher-final 0 \
  --save-every 500 --keep-every 2000 --log-every 25 --validation-size "$N_VAL"

echo "done: $RUN/inference.pt   try: ./speak.sh \"aap kaise hain?\""
