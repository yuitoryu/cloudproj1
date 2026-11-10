#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# != 2 )); then
    echo "Usage: bash run_gpu.sh ENVIRONMENT RESULTS_DIRECTORY" >&2
    exit 2
fi

environment="$1"
run_root="$2"
status=0
export DEVICE="${DEVICE:-cuda}" THREADS="${THREADS:-1}"

echo "[2/3] Measuring on $DEVICE"
"${PYTHON:-python}" - "$DEVICE" <<'PY' || exit 2
import sys
from benchmark import configure_runtime

if configure_runtime(sys.argv[1]).type != "cuda":
    raise SystemExit("The gpu stage requires a CUDA GPU")
PY

mkdir -p -- "$run_root" || exit 2
# Reserve the directory so a retry cannot mix measurements with an earlier run.
if ! mkdir -- "$run_root/smoke"; then
    echo "Choose a fresh results directory for the GPU stage" >&2
    exit 2
fi
OUTPUT_DIR="$run_root/smoke" bash run.sh "$environment" smoke || exit 1
OUTPUT_DIR="$run_root/baseline" bash run.sh "$environment" baseline \
    --warmup-steps 50 --steps 1000 --repetitions 5 || status=1
OUTPUT_DIR="$run_root/nsight" CAPTURE_ONLY=1 bash run.sh "$environment" profile \
    --warmup-steps 50 || status=1
exit "$status"
