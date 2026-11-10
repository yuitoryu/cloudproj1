#!/usr/bin/env bash
# Run the formal experiment: bash run.sh nyu-hpc all
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

environment="${1:-local}"
group="${2:-baseline}"
if (( $# >= 2 )); then shift 2; else shift "$#"; fi
python="${PYTHON:-python}"
device="${DEVICE:-auto}"
output="${OUTPUT_DIR:-results}"
threads="${THREADS:-1}"
data_dir="${DATA_DIR:-data}"
label="${environment//[^[:alnum:]_.-]/-}"
common=(--environment "$environment" --device "$device" --threads "$threads"
        --data-dir "$data_dir" --output-dir "$output" --no-download)
failed=0

run_case() {
    echo "Running: $*"
    if ! "$python" benchmark.py "${common[@]}" "$@"; then failed=1; fi
}

prepare_run() {
    local run_root="$1"
    echo "[1/3] Preparing CIFAR-10 in $data_dir"
    "$python" - "$data_dir" <<'PY' || return 2
import sys
from torchvision.datasets import CIFAR10

CIFAR10(sys.argv[1], train=True, download=True)
PY
    mkdir -p -- "$run_root"
}

measure_gpu() {
    local run_root="$1" status=0
    export DEVICE="${DEVICE:-cuda}" THREADS="$threads"
    echo "[2/3] Measuring on $DEVICE"
    "$python" - "$DEVICE" <<'PY' || return 2
import sys
from benchmark import configure_runtime

if configure_runtime(sys.argv[1]).type != "cuda":
    raise SystemExit("The gpu stage requires a CUDA GPU")
PY
    mkdir -p -- "$run_root" || return 2
    # Reserve this run before measuring so retries cannot mix configurations.
    if ! mkdir -- "$run_root/smoke"; then
        echo "Choose a fresh results directory for the GPU stage" >&2
        return 2
    fi
    OUTPUT_DIR="$run_root/smoke" bash run.sh "$environment" smoke || return 1
    OUTPUT_DIR="$run_root/baseline" bash run.sh "$environment" baseline \
        --warmup-steps 50 --steps 1000 --repetitions 5 || status=1
    OUTPUT_DIR="$run_root/nsight" CAPTURE_ONLY=1 bash run.sh "$environment" profile \
        --warmup-steps 50 || status=1
    return "$status"
}

finish_run() {
    local run_root="$1" status=0 plotted=0 run_dir
    local calibration="$run_root/nsight/${label}_calibration.json"
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
    return "$status"
}

case "$group" in
    all)
        if (( $# )); then
            echo "Usage: bash run.sh ENVIRONMENT all (set DATA_DIR, OUTPUT_DIR or DEVICE if needed)" >&2
            exit 2
        fi
        run_root="$output/${label}_$(date +%Y-%m-%d_%H-%M-%S)"
        mkdir -p -- "$output" && mkdir -- "$run_root" || exit 2
        prepare_run "$run_root" || exit 2
        measure_gpu "$run_root" || failed=1
        if [[ -d "$run_root/baseline" ]]; then finish_run "$run_root" || failed=1; fi
        ;;
    prepare|gpu|finish)
        if (( $# != 1 )); then
            echo "Usage: bash run.sh ENVIRONMENT $group RESULTS_DIRECTORY" >&2
            exit 2
        fi
        case "$group" in
            prepare) prepare_run "$1" ;;
            gpu) measure_gpu "$1" ;;
            finish) finish_run "$1" ;;
        esac
        exit $?
        ;;
    baseline)
        for model in resnet18 vit mlp; do
            for batch in 8 32 64; do
                for scope in compute-only end-to-end; do
                    run_case --model "$model" --batch-size "$batch" --timing-scope "$scope" "$@"
                done
            done
        done
        ;;
    size)
        for model in vit mlp; do
            for budget in 1000000 5000000 10000000 20000000; do
                run_case --model "$model" --target-params "$budget" --batch-size 32 --timing-scope compute-only "$@"
            done
        done
        ;;
    smoke)
        for model in resnet18 vit mlp; do
            for scope in compute-only end-to-end; do
                run_case --model "$model" --batch-size 2 --timing-scope "$scope" \
                    --warmup-steps 1 --steps 3 --repetitions 1 "$@"
            done
        done
        ;;
    profile)
        mkdir -p "$output" || exit 2
        calibration="$output/${label}_calibration.json"
        "$python" roofline.py calibrate --environment "$environment" --device "$device" \
            --threads "$threads" --output "$calibration" || exit 2
        if ! command -v ncu >/dev/null; then
            echo "roofline-incomplete: ncu is not on PATH; calibration saved to $calibration" >&2
            exit 2
        fi
        metrics="gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum"
        metrics+=",smsp__sass_thread_inst_executed_op_fadd_pred_on.sum"
        metrics+=",smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"
        metrics+=",smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
        for model in resnet18 vit mlp; do
            # Reserve the readable directory before ncu opens its report/CSV files.
            run_dir=$("$python" - "$environment" "$model" "$device" "$output" <<'PY'
import sys
from benchmark import configure_runtime, new_run_directory
environment, model, device, output = sys.argv[1:]
print(new_run_directory(environment, model, configure_runtime(device), output).resolve())
PY
            ) || exit 2
            if NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT=1 ncu \
                --target-processes all --profile-from-start off \
                --cache-control none --clock-control none \
                --section SpeedOfLight_RooflineChart --metrics "$metrics" \
                --csv --page raw --print-units base --log-file "$run_dir/ncu.csv" \
                --export "$run_dir/nsight-roofline" \
                "$python" benchmark.py "${common[@]}" --model "$model" --batch-size 32 \
                "$@" --timing-scope compute-only --steps 1 --repetitions 1 \
                --capture --no-progress --run-dir "$run_dir"; then
                if [[ "${CAPTURE_ONLY:-0}" != 1 ]]; then
                    "$python" roofline.py plot --runs "$run_dir" --calibration "$calibration" \
                        --output "$run_dir/roofline.png" || failed=1
                fi
            else
                echo "roofline-incomplete: ncu capture failed for $model; see $run_dir/ncu.csv" >&2
                failed=1
            fi
        done
        ;;
    *)
        echo "Usage: bash run.sh ENVIRONMENT {all|prepare|gpu|finish|baseline|size|smoke|profile} [arguments]" >&2
        exit 2
        ;;
esac
exit "$failed"
