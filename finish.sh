#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# != 2 )); then
    echo "Usage: bash finish.sh ENVIRONMENT RESULTS_DIRECTORY" >&2
    exit 2
fi

environment="$1"
run_root="$2"
python="${PYTHON:-python}"
label="${environment//[^[:alnum:]_.-]/-}"
calibration="$run_root/nsight/${label}_calibration.json"
status=0
plotted=0

echo "[3/3] Plotting results in $run_root"
"$python" roofline.py results "$run_root/baseline" \
    --output-dir "$run_root/figures" || status=1
if [[ -f "$calibration" ]]; then
    for run_dir in "$run_root"/nsight/*/; do
        if [[ -f "$run_dir/summary.json" && -f "$run_dir/ncu.csv" ]]; then
            if "$python" roofline.py plot --runs "$run_dir" --calibration "$calibration" \
                --output "$run_dir/roofline.png"; then
                ((plotted += 1))
            else
                status=1
            fi
        fi
    done
fi
if (( plotted < 3 )); then
    echo "roofline-incomplete: plotted $plotted of 3 model captures" >&2
    status=1
fi
echo "Results saved to $run_root"
exit "$status"
