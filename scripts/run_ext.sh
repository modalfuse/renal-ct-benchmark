#!/usr/bin/env bash
# External evaluation after CV weights exist. Inference only.
# Required: CKPT_DIR, RCC_ROOT, KITS_ROOT
# Optional: GPU, OUT_DIR, CODE, PY, RUN_RCC, RUN_KITS
set -euo pipefail

GPU="${GPU:-0}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR}"
RCC_ROOT="${RCC_ROOT:?set RCC_ROOT}"
KITS_ROOT="${KITS_ROOT:?set KITS_ROOT}"
CODE="${CODE:-$(cd "$(dirname "$0")/../src" && pwd)}"
PY="${PY:-python}"
OUT_DIR="${OUT_DIR:-$(dirname "$CKPT_DIR")}"
RUN_RCC="${RUN_RCC:-1}"
RUN_KITS="${RUN_KITS:-1}"

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1

if [[ "$RUN_RCC" == "1" ]]; then
  "$PY" "$CODE/eval_ext.py" \
    --ckpt-dir "$CKPT_DIR" \
    --dataset rccaid \
    --roi-root "$RCC_ROOT" \
    --out-csv "$OUT_DIR/ext_rccaid_metrics.csv" \
    --out-json "$OUT_DIR/ext_rccaid_metrics.json" \
    --cm-png "$OUT_DIR/ext_rccaid_cm.png" \
    --device cuda
fi

if [[ "$RUN_KITS" == "1" ]]; then
  "$PY" "$CODE/eval_ext.py" \
    --ckpt-dir "$CKPT_DIR" \
    --dataset kits23 \
    --roi-root "$KITS_ROOT" \
    --out-csv "$OUT_DIR/ext_kits23_metrics.csv" \
    --out-json "$OUT_DIR/ext_kits23_metrics.json" \
    --cm-png "$OUT_DIR/ext_kits23_cm.png" \
    --device cuda
fi
