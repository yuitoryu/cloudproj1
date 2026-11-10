#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if (( $# != 2 )); then
    echo "Usage: bash prepare.sh ENVIRONMENT RESULTS_DIRECTORY" >&2
    exit 2
fi

data_dir="${DATA_DIR:-data}"
echo "[1/3] Preparing CIFAR-10 for $1 in $data_dir"
"${PYTHON:-python}" - "$data_dir" <<'PY' || exit 2
import sys
from torchvision.datasets import CIFAR10

CIFAR10(sys.argv[1], train=True, download=True)
PY
mkdir -p -- "$2"
