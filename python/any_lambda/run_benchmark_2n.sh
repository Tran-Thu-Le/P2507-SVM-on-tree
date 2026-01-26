#!/usr/bin/env bash
set -euo pipefail

# ----------------------------
# Config (edit if you want)
# ----------------------------
PYTHON_BIN="${PYTHON_BIN:-../.venv/bin/python}"   # nếu .venv ở thư mục cha project, sửa lại cho đúng
SCRIPT="benchmark_svm_tree_vs_svc_2n.py"

LAMDA="${LAMDA:-10000.0}"
REPEATS="${REPEATS:-100}"
SEED="${SEED:-0}"
STANDARDIZE="${STANDARDIZE:-0}"  # 0/1

# sizes = n_per_class (tổng train = 2*n_per_class)
SIZES=(${SIZES:-100 200 400 1000 2000 5000})

# data params
SEP="${SEP:-6.0}"
SIGMA_PARA="${SIGMA_PARA:-2.5}"
SIGMA_PERP="${SIGMA_PERP:-2.5}"
RHO="${RHO:-0.0}"

# ----------------------------
# Fairness: pin threads
# ----------------------------
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

echo "=== Run benchmark_2n (FAIR) ==="
echo "Python: ${PYTHON_BIN}"
echo "Script: ${SCRIPT}"
echo "Threads: OMP=${OMP_NUM_THREADS}, MKL=${MKL_NUM_THREADS}, OPENBLAS=${OPENBLAS_NUM_THREADS}, VECLIB=${VECLIB_MAXIMUM_THREADS}"
echo "lamda=${LAMDA}, repeats=${REPEATS}, seed=${SEED}, standardize=${STANDARDIZE}"
echo "sizes (n_per_class) = ${SIZES[*]}"
echo "data: sep=${SEP}, sigma_para=${SIGMA_PARA}, sigma_perp=${SIGMA_PERP}, rho=${RHO}"
echo

# ----------------------------
# Run
# ----------------------------
"${PYTHON_BIN}" "${SCRIPT}" \
  --sizes "${SIZES[@]}" \
  --repeats "${REPEATS}" \
  --lamda "${LAMDA}" \
  --seed "${SEED}" \
  --standardize "${STANDARDIZE}" \
  --sep "${SEP}" \
  --sigma-para "${SIGMA_PARA}" \
  --sigma-perp "${SIGMA_PERP}" \
  --rho "${RHO}"
