# benchmark_sklearn_datasets.py
import time
import argparse
import numpy as np
from statistics import median

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# Your C++ core (binary)
from svm_on_tree_cpp import fit_core


# =========================================================
# Tree binary model helpers (prediction-ready from C++ output)
# =========================================================
def build_tree_model_dict(out):
    # C++ fair-ready returns these
    m0 = np.asarray(out["m0"], dtype=np.float64)
    w = np.asarray(out["w"], dtype=np.float64)
    thr = float(out["thr"])
    y_left = int(out["y_left"])
    y_right = int(out["y_right"])
    return {"m0": m0, "w": w, "thr": thr, "y_left": y_left, "y_right": y_right}

def tree_binary_scores(model, X):
    """
    Return a signed score where higher => predict class 1.
    We align sign based on which side corresponds to label 1.
    """
    m0, w, thr = model["m0"], model["w"], model["thr"]
    yl, yr = model["y_left"], model["y_right"]

    t = (X - m0) @ w  # projection
    # If right side is class 1 and left is class 0 => score = t - thr
    if yl == 0 and yr == 1:
        return t - thr
    # If right side is class 0 and left is class 1 => score = thr - t
    if yl == 1 and yr == 0:
        return thr - t
    # Fallback (should not happen): return (t-thr) and let caller handle
    return t - thr

def tree_binary_predict(model, X):
    score = tree_binary_scores(model, X)
    return (score >= 0).astype(np.int64)


# =========================================================
# OvR for Tree (multi-class)
# =========================================================
def fit_tree_ovr(X_train, y_train, lam):
    classes = np.unique(y_train)
    models = []
    for c in classes:
        y_bin = (y_train == c).astype(np.int64)  # 1 for class c, 0 otherwise
        out = fit_core(X_train, y_bin, lam)
        model = build_tree_model_dict(out)
        models.append(model)
    return classes, models

def predict_tree_ovr(classes, models, X):
    # score matrix: [n_samples, n_classes], pick argmax
    S = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
    for j, model in enumerate(models):
        S[:, j] = tree_binary_scores(model, X)  # higher => more likely class j
    return classes[np.argmax(S, axis=1)].astype(np.int64)


# =========================================================
# Dataset loading
# =========================================================
def load_dataset(name):
    name = name.lower()
    if name == "iris":
        data = load_iris()
    elif name == "wine":
        data = load_wine()
    elif name in ["breast_cancer", "cancer", "bc"]:
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X = data.data.astype(np.float64)
    y = data.target
    # ensure labels are 0..K-1
    y = LabelEncoder().fit_transform(y).astype(np.int64)
    return X, y, data


# =========================================================
# One benchmark run
# =========================================================
def run_tree_once(X_train, y_train, X_test, y_test, lam):
    K = len(np.unique(y_train))

    t0 = time.perf_counter()
    if K == 2:
        out = fit_core(X_train, y_train, lam)
        model = build_tree_model_dict(out)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = tree_binary_predict(model, X_test)
        t_pred = time.perf_counter() - t1
    else:
        classes, models = fit_tree_ovr(X_train, y_train, lam)
        t_fit = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = predict_tree_ovr(classes, models, X_test)
        t_pred = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    return float(t_fit), float(t_pred), float(acc), K

def run_linearsvc_once(X_train, y_train, X_test, y_test):
    t0 = time.perf_counter()
    clf = LinearSVC(
        dual=False,          # primal
        C=1.0,
        fit_intercept=True,
        random_state=0,
        max_iter=10000
    )
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    t_pred = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    return float(t_fit), float(t_pred), float(acc)


# =========================================================
# Benchmark driver
# =========================================================
def bench_dataset(
    dataset_name,
    repeats=20,
    lam=1.0,
    test_size=0.3,
    seed=0,
    standardize=1,
    warmup=True
):
    X, y, data = load_dataset(dataset_name)
    n, d = X.shape
    K = len(np.unique(y))

    # stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Optional standardize (fit on train only) - applied to BOTH methods
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float64)
        X_test  = scaler.transform(X_test).astype(np.float64)

    tag_tree = f"Tree({'binary' if K==2 else f'OvR-{K}'})"
    tag_lin  = "LinearSVC(ovr)"

    print(f"\n=== Dataset: {dataset_name} ===")
    print(f"total={n}, d={d}, classes={K}, train={X_train.shape[0]}, test={X_test.shape[0]}")
    print(f"repeats={repeats}, lamda={lam}, standardize={standardize}, test_size={test_size}, seed={seed}")
    print("Timing: FIT=wall clock around fit; PRED=wall clock around predict (end-to-end on test set).")

    if warmup:
        _ = run_tree_once(X_train, y_train, X_test, y_test, lam)
        _ = run_linearsvc_once(X_train, y_train, X_test, y_test)

    # Tree repeats
    tree_fit, tree_pred, tree_acc = [], [], []
    for _ in range(repeats):
        tf, tp, ta, _K = run_tree_once(X_train, y_train, X_test, y_test, lam)
        tree_fit.append(tf); tree_pred.append(tp); tree_acc.append(ta)

    # LinearSVC repeats
    lin_fit, lin_pred, lin_acc = [], [], []
    for _ in range(repeats):
        lf, lp, la = run_linearsvc_once(X_train, y_train, X_test, y_test)
        lin_fit.append(lf); lin_pred.append(lp); lin_acc.append(la)

    m_tf = median(tree_fit);  m_tp = median(tree_pred);  m_ta = median(tree_acc)
    m_lf = median(lin_fit);   m_lp = median(lin_pred);   m_la = median(lin_acc)

    sp_fit  = (m_lf / m_tf) if m_tf > 0 else float("inf")
    sp_pred = (m_lp / m_tp) if m_tp > 0 else float("inf")

    print("\n------------------------------------------------------------------------")
    print("MEDIAN RESULTS")
    print("------------------------------------------------------------------------")
    print(f"{tag_tree:16s}: FIT={m_tf:.6f}s  PRED={m_tp:.6f}s  ACC={m_ta:.4f}")
    print(f"{tag_lin:16s}: FIT={m_lf:.6f}s  PRED={m_lp:.6f}s  ACC={m_la:.4f}")
    print(f"Speedup (LinearSVC/Tree): FIT={sp_fit:7.2f}x  PRED={sp_pred:7.2f}x")
    print("------------------------------------------------------------------------")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["iris", "wine", "breast_cancer"])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--standardize", type=int, default=1, choices=[0, 1])
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    print("=== Benchmark: Iris / Wine / Breast Cancer (Tree vs LinearSVC) ===")
    print("Threads pinned recommended:")
    print("OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1")

    for ds in args.datasets:
        bench_dataset(
            ds,
            repeats=args.repeats,
            lam=args.lamda,
            test_size=args.test_size,
            seed=args.seed,
            standardize=args.standardize,
            warmup=not args.no_warmup
        )

if __name__ == "__main__":
    main()
