# SVM On Tree - Fast C++ Implementation

Fast and efficient implementation of SVM On Tree algorithm using C++ with Python bindings via pybind11. This implementation provides significant speedup compared to traditional SVM while maintaining competitive accuracy.

## Project Information

- **Title**: Fast SVM on Tree
- **Abstract**: 
  - Support Vector Machines (SVM) are powerful but computationally expensive
  - We approximate data points with a tree structure
  - Our method achieves faster training and prediction times
  - Competitive accuracy with scikit-learn's SVC on suitable datasets
  
## Authors
- Cong-Huan Tran (First author)
- Ngan Nguyen
- Khanh-Duy Le
- Thu-Le Tran (Corresponding author)

## Timeline
- 14/07/2025: Initialized project
- 14-30/07/2025: Development and writing
- Deadline: 10/2025

## Quick Start

### Prerequisites
- Python 3.8+
- C++17 compatible compiler
- macOS, Linux, or Windows

### Installation

#### Option 1: Using virtual environment (Recommended)
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
pip install --upgrade pip
pip install pybind11 numpy scikit-learn matplotlib pandas

# Build the C++ extension (in python/any_lambda/)
cd python/any_lambda
python setup.py build_ext --inplace --force
```

#### Option 2: Using Anaconda
```bash
# Create and activate conda environment
conda create -n svmtree python=3.10 -y
conda activate svmtree

# Install dependencies
pip install pybind11 numpy scikit-learn matplotlib pandas

# Build the C++ extension
cd python/any_lambda
python setup.py build_ext --inplace --force
```

### Running Benchmarks

#### Complete benchmark (multiple datasets)
```bash
cd python/any_lambda
./run_benchmark.sh
```

#### 2N parametric benchmark (scaling test)
```bash
cd python/any_lambda
./run_benchmark_2n.sh
```

## Repository Structure

```
P2507-SVM-on-tree/
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── python/
│   └── any_lambda/          # Main implementation
│       ├── src/
│       │   └── svm_on_tree_lambda_any.cpp  # C++ core
│       ├── setup.py         # Build configuration
│       ├── complete_benchmark.py
│       ├── benchmark_svm_tree_vs_svc_2n.py
│       ├── run_benchmark.sh
│       └── run_benchmark_2n.sh
├── plots/                   # Benchmark results
├── latex/                   # Paper and documentation
└── .venv/                   # Virtual environment (not in repo)
```

## Thread Pinning (Important)

For fair comparison with scikit-learn, pin threads to 1:
```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```
This is automatically done in the provided shell scripts.

## Verify Installation

Test the C++ module:
```bash
python -c "import svm_on_tree_cpp; print('Module loaded successfully')"
```

## Benchmark Examples

### Run with custom parameters
```bash
cd python/any_lambda
python benchmark_svm_tree_vs_svc_2n.py \
  --sizes 100 200 500 1000 2000 \
  --repeats 7 \
  --lamda 10000.0 \
  --standardize 0
```

### Expected Output
```
N=  100 | Tree: ACC=0.9000, Time=0.05s | SVC: ACC=0.8950, Time=0.15s
Speedup: 3.0x
...
```

## Troubleshooting

### Build fails with "pybind11 not found"
```bash
pip install pybind11
```

### Import error: "svm_on_tree_cpp not found"
Make sure you built the extension in the correct directory:
```bash
cd python/any_lambda
python setup.py build_ext --inplace --force
```

### Performance issues
Ensure thread pinning is active:
```bash
echo $OMP_NUM_THREADS  # Should output: 1
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{svm_on_tree_2025,
  title = {SVM On Tree: Fast C++ Implementation},
  author = {Tran, Cong-Huan and Nguyen, Ngan and Le, Khanh-Duy and Tran, Thu-Le},
  year = {2025},
  url = {https://github.com/Tran-Thu-Le/P2507-SVM-on-tree}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the corresponding author.
