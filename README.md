# SVM On Tree

C++ implementation (with Python/pybind11 bindings) of the **SVM On Tree** algorithm described in:

> *Fast SVM on Tree* — C.-H. Tran, N. Nguyen, K.-D. Le, T.-L. Tran

## Repository structure

```
.
├── src/
│   └── svm_on_tree_lambda_any.cpp   # Core C++ implementation (pybind11)
├── setup.py                          # Build script for the C++ extension
├── benchmark_svm_tree_vs_svc_2n.py   # Benchmark: SVM On Tree vs LinearSVC
├── plot_benchmark_results.py         # Plot benchmark results (lambda = 1)
├── plot_lambda_analysis.py           # Plot lambda sensitivity analysis
├── requirements.txt                  # Python dependencies
└── README.md
```

## Quick start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Build the C++ extension

```bash
python setup.py build_ext --inplace --force
```

### 3. Run the benchmark

Pin threads for reproducible timing:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmark_svm_tree_vs_svc_2n.py --lamda 1.0 --sizes 100 200 400 1000 2000 5000 --repeats 100
```

### 4. Generate plots

```bash
python plot_benchmark_results.py       # Benchmark comparison (lambda = 1)
python plot_lambda_analysis.py         # Lambda sensitivity analysis
```

Figures are saved to the `output/` folder.

## Requirements

- Python 3.8+
- C++17 compatible compiler
- macOS, Linux, or Windows
