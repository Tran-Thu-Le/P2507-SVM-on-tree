# Project Infor 

## Project infor

- Title: Fast SVM on tree 
- Abstract:
    - SVM is important 
    - Solving SVM is time consuming 
    - We approximate the points with a tree
    - We then propose a new fast computing method for SVM on tree 

## Authors 
- Cong-Huan Tran (First author)
- Ngan Nguyen 
- Khanh-Duy Le 
- Thu-Le Tran (Corresponding author) 

## Time 
- 14/07/2025: Ininitilized project 
- 14->30 07/2025: Writing
- Deadline: 10/2025 

## Code 
- Code in python

# 📄 How to Run `benchmark_svm_tree_vs_svc_2n.py` from Scratch

## Install Anaconda (if not installed)
Download and install Anaconda:  
🔗 https://www.anaconda.com/download

## Create and activate a new Python environment
Open **Terminal** (Mac / Linux) or **Anaconda Prompt** (Windows) and run:
```bash
conda create -n svmtree python=3.10 -y
conda activate svmtree
```

## Install required dependencies
```bash
pip install pybind11 numpy scikit-learn matplotlib pandas
```

## Build the C++ module `svm_on_tree_cpp`
Go to the **python** directory in the repo:
```bash
cd python
pip install -e .
```
This will compile `src/svm_on_tree.cpp` into the Python module `svm_on_tree_cpp` and install it into the current environment.

## Verify the module installation
```bash
python -c "import svm_on_tree_cpp as m; print('OK:', m.fit_core)"
```
If you see something like:
```
OK: <built-in method fit_core ...>
```
then the C++ module has been built successfully.

## Run the benchmark script
Go back to the repository root and run:
```bash
cd ..
python python/benchmark_svm_tree_vs_svc_2n.py --sizes 50 100 200 500 1000 2000 --repeats 7 > python/logs/run.txt
```
Arguments:
- `--sizes`: list of N values to test (number of training points).
- `--repeats`: number of repetitions for averaging results.

## Expected output
The script will print:
- **Training time (FIT)** for SVM On Tree and sklearn SVC.
- **Prediction time (PRED)**.
- **Accuracy (ACC)** for each method.
- **Speedup** factor.

Example:
```
N=   100 | Tree-Spine: ACC=0.9000, SVC: ACC=0.8950
Speedup PRED (Spine vs SVC): 15.45x
...
```
