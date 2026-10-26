#!/usr/bin/env bash
# Linear probe for one foundation model after its environment and weights are ready.
# Required: L4_ROOT, L4_SPLIT, L4_QC, L4_EXT, L4_ROI, and one of mmm|ctfm|fmcib
# Example:
#   L4_ROOT=/path/to/work L4_SPLIT=... L4_QC=... L4_EXT=... L4_ROI=... \
#     bash scripts/run_l4.sh mmm
set -euo pipefail

WHICH="${1:?usage: run_l4.sh mmm|ctfm|fmcib}"
L4_ROOT="${L4_ROOT:?set L4_ROOT}"
: "${L4_SPLIT:?set L4_SPLIT}"
: "${L4_QC:?set L4_QC}"
: "${L4_EXT:?set L4_EXT}"
: "${L4_ROI:?set L4_ROI}"

CODE="${CODE:-$(cd "$(dirname "$0")/../src" && pwd)}"
export L4_ROOT L4_SPLIT L4_QC L4_EXT L4_ROI
export PYTHONPATH="${CODE}${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

case "$WHICH" in
  mmm)
    PY="${MMM_PYTHON:-$L4_ROOT/envs/mmm/bin/python}"
    export CUDA_VISIBLE_DEVICES="${GPU:-2}"
    exec "$PY" "$CODE/l4_mmm_run.py"
    ;;
  ctfm)
    PY="${CTFM_PYTHON:-$L4_ROOT/envs/ctfm/bin/python}"
    export CUDA_VISIBLE_DEVICES="${GPU:-1}"
    exec "$PY" "$CODE/l4_ctfm_run.py"
    ;;
  fmcib)
    PY="${FMCIB_PYTHON:-$L4_ROOT/envs/fmcib/bin/python}"
    export CUDA_VISIBLE_DEVICES="${GPU:-0}"
    exec "$PY" "$CODE/l4_fmcib_run.py"
    ;;
  *)
    echo "unknown model: $WHICH" >&2
    exit 1
    ;;
esac
