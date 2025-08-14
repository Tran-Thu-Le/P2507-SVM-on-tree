# complete_benchmark.py
# Complete SVM On Tree benchmark with suitable low-dimensional datasets
# Author: Based on your SVM On Tree implementation

import time
import argparse
import numpy as np
from statistics import median
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.datasets import (
    make_classification, make_blobs, make_moons, make_circles,
    load_breast_cancer, load_wine, load_iris
)
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Import your C++ core (make sure it's installed)
from svm_on_tree_cpp import fit_core


# =========================
# Dataset Generators
# =========================

def generate_linearly_separable_2d(n_samples=1000, seed=42):
    """Perfect for Tree: clear linear separation"""
    X, y = make_classification(
        n_samples=n_samples, n_features=2, n_redundant=0, n_informative=2,
        n_clusters_per_class=1, random_state=seed, class_sep=2.0
    )
    return X, y

def generate_elongated_clusters(n_samples=1000, seed=42):
    """Elongated clusters along spine direction - IDEAL for Tree"""
    np.random.seed(seed)
    
    # Class 0: elongated along x-axis, centered at (-3, 0)
    cov0 = np.array([[4.0, 0.5], [0.5, 1.0]])
    X0 = np.random.multivariate_normal([-3, 0], cov0, n_samples//2)
    
    # Class 1: elongated along x-axis, centered at (3, 0)  
    cov1 = np.array([[4.0, -0.5], [-0.5, 1.0]])
    X1 = np.random.multivariate_normal([3, 0], cov1, n_samples//2)
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    return X, y

def generate_banana_shaped(n_samples=1000, seed=42):
    """Two banana-shaped clusters"""
    np.random.seed(seed)
    t = np.linspace(0, np.pi, n_samples//2)
    
    # Class 0: upper banana
    r0 = 3 + 0.5*np.random.randn(n_samples//2)
    X0 = np.column_stack([r0*np.cos(t), r0*np.sin(t) + 1])
    
    # Class 1: lower banana (flipped)
    r1 = 3 + 0.5*np.random.randn(n_samples//2) 
    X1 = np.column_stack([r1*np.cos(t), -r1*np.sin(t) - 1])
    
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    return X, y

def load_breast_cancer_2d():
    """Wisconsin Breast Cancer - reduced to 2D with PCA"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(StandardScaler().fit_transform(X))
    
    print(f"Breast Cancer 2D: Explained variance ratio = {pca.explained_variance_ratio_}")
    return X_2d, y

def load_wine_binary_2d():
    """Wine dataset - convert to binary + 2D PCA"""
    data = load_wine()
    X, y = data.data, data.target
    
    # Convert to binary (class 0 vs rest)
    y_binary = (y == 0).astype(int)
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(StandardScaler().fit_transform(X))
    
    print(f"Wine Binary 2D: Explained variance ratio = {pca.explained_variance_ratio_}")
    return X_2d, y_binary

def load_iris_binary_2d():
    """Iris dataset - binary classification on 2 best features"""
    data = load_iris()
    X, y = data.data, data.target
    
    # Take only 2 classes for binary classification
    mask = y != 2  # Remove class 2 (Virginica)
    X_binary = X[mask]
    y_binary = y[mask]
    
    # Use petal length + petal width (features 2,3) - most separable
    X_2d = X_binary[:, [2, 3]]
    
    return X_2d, y_binary

def get_dataset(name, n_samples=1000, seed=42):
    """Get dataset by name"""
    datasets = {
        'elongated': generate_elongated_clusters(n_samples, seed),
        'linear': generate_linearly_separable_2d(n_samples, seed),
        'banana': generate_banana_shaped(n_samples, seed),
        'blobs': make_blobs(n_samples=n_samples, centers=2, n_features=2, 
                           cluster_std=1.5, random_state=seed),
        'moons': make_moons(n_samples=n_samples, noise=0.1, random_state=seed),
        'circles': make_circles(n_samples=n_samples, noise=0.1, factor=0.6, random_state=seed),
        'breast_cancer': load_breast_cancer_2d(),
        'wine': load_wine_binary_2d(),
        'iris': load_iris_binary_2d()
    }
    
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(datasets.keys())}")
    
    return datasets[name]


# =========================
# SVM On Tree Implementation
# =========================

def spine_params(X_train, y_train):
    """Calculate spine parameters (mean difference direction)"""
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    w = (m1 - m0).astype(np.float64)
    nrm = np.linalg.norm(w)
    w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
    return m0, w_unit

def t_coords(X_any, m0, w_unit):
    """Project points onto spine"""
    return (X_any - m0) @ w_unit

def predict_spine_threshold(X_eval, X_train, y_train, s_idx, p_idx):
    """Predict using spine threshold from support points"""
    m0, w_unit = spine_params(X_train, y_train)
    t_train = t_coords(X_train, m0, w_unit)
    t_eval = t_coords(X_eval, m0, w_unit)

    ts, tp = t_train[s_idx], t_train[p_idx]
    thr = 0.5 * (ts + tp)

    if ts <= tp:
        y_left, y_right = y_train[s_idx], y_train[p_idx]
    else:
        y_left, y_right = y_train[p_idx], y_train[s_idx]

    return np.where(t_eval < thr, y_left, y_right).astype(np.int64)


# =========================
# Benchmark Functions
# =========================

def run_tree_once(X_train, y_train, X_test, y_test, lam=1.0):
    """Run SVM On Tree once"""
    # Fit
    t0 = time.perf_counter()
    out = fit_core(X_train, y_train, lam)
    t_fit = time.perf_counter() - t0
    
    s_idx, p_idx = int(out["support_s"]), int(out["support_p"])
    
    # Predict
    t1 = time.perf_counter()
    y_pred = predict_spine_threshold(X_test, X_train, y_train, s_idx, p_idx)
    t_pred = time.perf_counter() - t1
    
    acc = accuracy_score(y_test, y_pred)
    
    return {
        "fit_time": float(t_fit),
        "pred_time": float(t_pred),
        "accuracy": float(acc),
        "support": (s_idx, p_idx),
        "loss": float(out["min_val"])
    }

# def run_svc_once(X_train, y_train, X_test, y_test, kernel='rbf', C=1.0):
#     """Run sklearn SVC once"""
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     # Fit
#     t0 = time.perf_counter()
#     svc = SVC(kernel=kernel, C=C)
#     svc.fit(X_train_scaled, y_train)
#     t_fit = time.perf_counter() - t0
    
#     # Predict
#     t1 = time.perf_counter()
#     y_pred = svc.predict(X_test_scaled)
#     t_pred = time.perf_counter() - t1
    
#     acc = accuracy_score(y_test, y_pred)
    
#     return float(t_fit), float(t_pred), float(acc)

def run_svc_once(X_train, y_train, X_test, y_test, kernel='linear', C=1.0):
    """Run sklearn SVC once with linear kernel and no scaling"""
    # Fit
    t0 = time.perf_counter()
    svc = SVC(kernel='linear')
    svc.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0
    
    # Predict
    t1 = time.perf_counter()
    y_pred = svc.predict(X_test)
    t_pred = time.perf_counter() - t1
    
    acc = accuracy_score(y_test, y_pred)
    
    return float(t_fit), float(t_pred), float(acc)

def evaluate_dataset_suitability(X, y, dataset_name):
    """Evaluate how suitable a dataset is for SVM On Tree"""
    X0, X1 = X[y==0], X[y==1]
    m0, m1 = X0.mean(axis=0), X1.mean(axis=0)
    spine_direction = m1 - m0
    spine_unit = spine_direction / (np.linalg.norm(spine_direction) + 1e-12)
    
    # Project data onto spine
    t0 = (X0 - m0) @ spine_unit
    t1 = (X1 - m0) @ spine_unit
    
    # Measure overlap on spine
    overlap_ratio = max(0, min(t1.max(), t0.max()) - max(t1.min(), t0.min())) / \
                   (max(t1.max(), t0.max()) - min(t1.min(), t0.min()) + 1e-12)
    
    # Spine dominance (variance ratio if 2D+)
    if X.shape[1] >= 2:
        pca = PCA(n_components=min(2, X.shape[1]))
        pca.fit(X)
        spine_dominance = pca.explained_variance_ratio_[0] / (pca.explained_variance_ratio_[1] + 1e-12)
    else:
        spine_dominance = 1.0
    
    # Class balance
    class_balance = min(np.sum(y==0), np.sum(y==1)) / len(y)
    
    # Suitability score (heuristic)
    suitability = (1 - overlap_ratio) * np.log(spine_dominance + 1) * (4 * class_balance * (1-class_balance))
    
    print(f"\n=== {dataset_name} Suitability Analysis ===")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
    print(f"Class balance: {class_balance:.3f}")
    print(f"Overlap on spine: {overlap_ratio:.3f} (lower is better)")
    print(f"Spine dominance: {spine_dominance:.3f} (higher is better)")
    print(f"Suitability score: {suitability:.3f} (higher is better)")
    
    return suitability

def benchmark_single_dataset(dataset_name, n_samples=1000, test_size=0.3, repeats=5, lam=1.0, seed=42):
    """Benchmark Tree vs SVC on a single dataset"""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {dataset_name.upper()}")
    print(f"{'='*60}")
    
    # Load dataset
    X, y = get_dataset(dataset_name, n_samples, seed)
    
    # Evaluate suitability
    suitability = evaluate_dataset_suitability(X, y, dataset_name)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    
    # Run Tree multiple times
    tree_results = []
    for _ in range(repeats):
        result = run_tree_once(X_train, y_train, X_test, y_test, lam)
        tree_results.append(result)
    
    # Run SVC multiple times  
    svc_results = []
    for _ in range(repeats):
        fit_time, pred_time, acc = run_svc_once(X_train, y_train, X_test, y_test, kernel='rbf')
        svc_results.append({'fit_time': fit_time, 'pred_time': pred_time, 'accuracy': acc})
    
    # Calculate medians
    tree_fit = median([r['fit_time'] for r in tree_results])
    tree_pred = median([r['pred_time'] for r in tree_results])
    tree_acc = median([r['accuracy'] for r in tree_results])
    
    svc_fit = median([r['fit_time'] for r in svc_results])
    svc_pred = median([r['pred_time'] for r in svc_results])
    svc_acc = median([r['accuracy'] for r in svc_results])
    
    # Calculate speedups
    fit_speedup = svc_fit / tree_fit if tree_fit > 0 else float('inf')
    pred_speedup = svc_pred / tree_pred if tree_pred > 0 else float('inf')
    
    # Print results
    print(f"\n=== RESULTS (median of {repeats} runs) ===")
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"")
    print(f"FIT TIME:")
    print(f"  SVM On Tree: {tree_fit:.6f}s")
    print(f"  SVC (RBF):   {svc_fit:.6f}s")
    print(f"  Speedup:     {fit_speedup:.2f}x")
    print(f"")
    print(f"PREDICTION TIME:")
    print(f"  SVM On Tree: {tree_pred:.6f}s")
    print(f"  SVC (RBF):   {svc_pred:.6f}s")
    print(f"  Speedup:     {pred_speedup:.2f}x")
    print(f"")
    print(f"ACCURACY:")
    print(f"  SVM On Tree: {tree_acc:.4f}")
    print(f"  SVC (RBF):   {svc_acc:.4f}")
    print(f"  Difference:  {tree_acc - svc_acc:+.4f}")
    print(f"")
    print(f"SUITABILITY SCORE: {suitability:.3f}")
    
    # Show support points from last run
    last_support = tree_results[-1]['support']
    print(f"Last run support points: s={last_support[0]}, p={last_support[1]}")

def benchmark_all_datasets(repeats=3, lam=1.0, n_samples=1000):
    """Benchmark all available datasets"""
    datasets_to_test = [
        'elongated',      # BEST for Tree
        'linear',         # Very good for Tree
        'breast_cancer',  # Real-world test
        'wine',           # Real-world test
        'iris',           # Classic benchmark
        'blobs',          # Moderate difficulty
        'moons',          # Non-linear challenge
        'banana',         # Curved clusters
        'circles'         # Hardest case
    ]
    
    print("=== COMPLETE BENCHMARK SUITE ===")
    print(f"Testing {len(datasets_to_test)} datasets with {repeats} repeats each")
    print(f"Sample size: {n_samples}, Lambda: {lam}")
    
    summary_results = []
    
    for dataset_name in datasets_to_test:
        try:
            benchmark_single_dataset(dataset_name, n_samples=n_samples, repeats=repeats, lam=lam)
            summary_results.append(dataset_name)
        except Exception as e:
            print(f"ERROR with {dataset_name}: {e}")
    
    print(f"\n{'='*60}")
    print("BENCHMARK COMPLETED")
    print(f"Successfully tested: {len(summary_results)}/{len(datasets_to_test)} datasets")
    print("Datasets ranked by expected Tree performance:")
    print("1. elongated (IDEAL - should dominate SVC)")
    print("2. linear (Very good)")  
    print("3. breast_cancer, wine, iris (Real-world)")
    print("4. blobs (Moderate)")
    print("5. moons, banana (Non-linear challenge)")
    print("6. circles (Hardest - SVC should win)")
    print(f"{'='*60}")


# =========================
# Main CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Complete benchmark of SVM On Tree vs sklearn SVC on suitable datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python complete_benchmark.py --dataset elongated --repeats 5
  python complete_benchmark.py --dataset all --n-samples 2000
  python complete_benchmark.py --dataset breast_cancer --lamda 2.0
        """
    )
    
    parser.add_argument("--dataset", type=str, default="elongated",
                       choices=['elongated', 'linear', 'banana', 'blobs', 'moons', 'circles',
                               'breast_cancer', 'wine', 'iris', 'all'],
                       help="Dataset to benchmark (or 'all' for complete suite)")
    
    parser.add_argument("--n-samples", type=int, default=1000,
                       help="Number of samples for synthetic datasets")
    
    parser.add_argument("--repeats", type=int, default=5,
                       help="Number of benchmark repeats for statistical reliability")
    
    parser.add_argument("--lamda", type=float, default=1.0,
                       help="Lambda parameter for SVM On Tree")
    
    parser.add_argument("--test-size", type=float, default=0.3,
                       help="Fraction of data to use for testing")
    
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.dataset == "all":
        benchmark_all_datasets(repeats=args.repeats, lam=args.lamda, n_samples=args.n_samples)
    else:
        benchmark_single_dataset(
            args.dataset, 
            n_samples=args.n_samples, 
            test_size=args.test_size,
            repeats=args.repeats, 
            lam=args.lamda, 
            seed=args.seed
        )