#!/usr/bin/env bash
# Fixed-fold classification training. Fold membership comes from the CSV only.
# Required:
#   SEG_OUT  directory of ROI images and masks
#   L4_SPLIT fold CSV
#   LABELS   five-class label CSV
#   OUT      output directory (must not already exist)
# Optional: GPU, SEED, EPOCHS, BS, LR, SIZE, MODEL, RUN_FOLDS, PY, CODE
set -euo pipefail

GPU="${GPU:-0}"
SEED="${SEED:-20261024}"
EPOCHS="${EPOCHS:-100}"
BS="${BS:-8}"
LR="${LR:-1e-4}"
SIZE="${SIZE:-96}"
MODEL="${MODEL:-resnet18}"
RUN_FOLDS="${RUN_FOLDS:-0}"
SEG_OUT="${SEG_OUT:?set SEG_OUT}"
L4_SPLIT="${L4_SPLIT:?set L4_SPLIT}"
LABELS="${LABELS:?set LABELS}"
OUT="${OUT:?set OUT}"
CODE="${CODE:-$(cd "$(dirname "$0")/../src" && pwd)}"
PY="${PY:-python}"
LOG_DIR="${LOG_DIR:-$(dirname "$OUT")/logs}"

mkdir -p "$LOG_DIR"
if [[ -e "$OUT" ]]; then
  echo "OUT already exists, refuse overwrite: $OUT" >&2
  exit 1
fi

LOG="$LOG_DIR/cv_$(date +%Y%m%d_%H%M%S).log"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

echo "GPU=$GPU OUT=$OUT SEED=$SEED EPOCHS=$EPOCHS RUN_FOLDS=$RUN_FOLDS" | tee "$LOG"
echo "split=$L4_SPLIT" | tee -a "$LOG"

"$PY" "$CODE/train_cls.py" \
  --seg-out "$SEG_OUT" \
  --split "$L4_SPLIT" \
  --labels "$LABELS" \
  --out "$OUT" \
  --model "$MODEL" \
  --size "$SIZE" \
  --epochs "$EPOCHS" \
  --bs "$BS" \
  --lr "$LR" \
  --seed "$SEED" \
  --device cuda \
  --run-folds "$RUN_FOLDS" \
  2>&1 | tee -a "$LOG"
