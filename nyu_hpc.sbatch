#!/usr/bin/env bash
# Supply the site's account/partition at submission time.
#SBATCH --job-name=cifar-benchmark
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=cifar-benchmark-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Activate Python before sbatch. To submit just the GPU stage, pass `gpu RESULTS_DIRECTORY`.
group="${1:-baseline}"
if (( $# )); then shift; fi
if [[ "$group" == all || "$group" == gpu ]]; then
    export THREADS="${THREADS:-1}"
else
    export THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
fi
export DEVICE=cuda
export DATA_DIR="${CIFAR_DATA_DIR:-${DATA_DIR:-data}}"
bash run.sh nyu-hpc "$group" "$@"
