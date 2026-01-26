# #!/usr/bin/env bash
# set -e

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# cd "$SCRIPT_DIR"

# # Clean old builds
# rm -rf build
# rm -f svm_on_tree_cpp*.so svm_on_tree_cpp*.pyd

# # Build extension in-place
# python setup.py build_ext --inplace --force

# # Thread fairness (single thread)
# export OMP_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1

# # Run benchmark (all datasets)
# python complete_benchmark.py --dataset all --n-samples 2000 --repeats 7 --lamda 1.0 --standardize 0

#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Pick python interpreter
PY="${PYTHON:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x "${HERE}/../../.venv/bin/python" ]]; then
    PY="${HERE}/../../.venv/bin/python"
  elif [[ -x "${HERE}/.venv/bin/python" ]]; then
    PY="${HERE}/.venv/bin/python"
  else
    PY="$(command -v python3 || command -v python)"
  fi
fi

cd "${HERE}"

echo "[1/3] Clean old builds"
rm -rf build svm_on_tree_cpp*.so

echo "[2/3] Build extension (pybind11)"
"${PY}" setup.py build_ext --inplace --force

echo "[3/3] Run benchmark (all datasets)"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# You can override defaults by passing args to this script, e.g.:
# ./run_benchmark --repeats 10 --n-samples 5000 --lamda 0.7
"${PY}" complete_benchmark.py --dataset all "$@"
