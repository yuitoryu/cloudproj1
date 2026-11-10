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

case "$group" in
    all)
        if (( $# )); then
            echo "Usage: bash run.sh ENVIRONMENT all (set DATA_DIR, OUTPUT_DIR or DEVICE if needed)" >&2
            exit 2
        fi
        run_root="$output/${label}_$(date +%Y-%m-%d_%H-%M-%S)"
        mkdir -p -- "$output" && mkdir -- "$run_root" || exit 2
        bash prepare.sh "$environment" "$run_root" || exit 2
        bash run_gpu.sh "$environment" "$run_root" || failed=1
        if [[ -d "$run_root/baseline" ]]; then
            bash finish.sh "$environment" "$run_root" || failed=1
        fi
        ;;
    prepare) exec bash prepare.sh "$environment" "$@" ;;
    gpu) exec bash run_gpu.sh "$environment" "$@" ;;
    finish) exec bash finish.sh "$environment" "$@" ;;
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
