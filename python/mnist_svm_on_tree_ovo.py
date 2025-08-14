# mnist_svm_on_tree_ovo.py
# SVM On Tree (One-vs-One) on MNIST in 784D — uses C++ core fit_core for each pair
# Author: you

import time
import argparse
import numpy as np
from statistics import median
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# C++ core (pybind11) — make sure you've run:  (cd python && pip install -e .)
from svm_on_tree_cpp import fit_core


# -----------------------------
# Data loading
# -----------------------------
def load_mnist():
    """
    Return:
      X: float64, shape (70000, 784)
      y: int64,   shape (70000,)
    Prefers tensorflow.keras.datasets.mnist; falls back to OpenML if TF missing.
    """
    try:
        from tensorflow.keras.datasets import mnist
        (Xtr, ytr), (Xte, yte) = mnist.load_data()
        Xtr = Xtr.reshape(-1, 28 * 28).astype(np.float64)
        Xte = Xte.reshape(-1, 28 * 28).astype(np.float64)
        ytr = ytr.astype(np.int64)
        yte = yte.astype(np.int64)
        X = np.vstack([Xtr, Xte])
        y = np.hstack([ytr, yte])
        return X, y
    except Exception:
        from sklearn.datasets import fetch_openml
        mn = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X = mn.data.astype(np.float64)
        y = mn.target.astype(np.int64)
        return X, y


def sample_by_class(X, y, classes, per_class_train, per_class_test, rng):
    """
    Balanced sampling per class, train/test disjoint (random but reproducible via rng).
    """
    Xtr, ytr, Xte, yte = [], [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        if len(idx) < per_class_train + per_class_test:
            raise ValueError(f"Class {c} has {len(idx)} samples < needed {per_class_train + per_class_test}.")
        rng.shuffle(idx)
        tr = idx[:per_class_train]
        te = idx[per_class_train:per_class_train + per_class_test]
        Xtr.append(X[tr])
        ytr.append(y[tr])
        Xte.append(X[te])
        yte.append(y[te])
    X_train = np.vstack(Xtr)
    y_train = np.hstack(ytr)
    X_test = np.vstack(Xte)
    y_test = np.hstack(yte)
    # Shuffle set-wise
    p = rng.permutation(len(y_train))
    q = rng.permutation(len(y_test))
    return X_train[p], y_train[p], X_test[q], y_test[q]


def maybe_standardize(X_train, X_test, enable=False):
    if not enable:
        return X_train, X_test, None
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test), scaler


# -----------------------------
# SVM On Tree OVO in 784D
# -----------------------------
def t_coords(X_any, m0, w_unit):
    """Project to spine: t = <x - m0, w_unit>"""
    return (X_any - m0) @ w_unit


def best_threshold_on_train(t_vals, y01):
    """
    Optimize 0/1 error on train along 1D projection.
    y01 ∈ {0,1}, convention: left->0, right->1.
    """
    order = np.argsort(t_vals)
    t_sorted = t_vals[order]
    # Candidates: just outside extremes & midpoints between consecutive projections
    cand = [t_sorted[0] - 1e-9]
    cand += [(t_sorted[i] + t_sorted[i + 1]) * 0.5 for i in range(len(t_sorted) - 1)]
    cand.append(t_sorted[-1] + 1e-9)

    best_err, best_thr = 1.0, cand[0]
    for thr in cand:
        pred = (t_vals >= thr).astype(np.int64)  # left=0, right=1
        err = np.mean(pred != y01)
        if err < best_err:
            best_err, best_thr = err, thr
    return float(best_thr), 0, 1


def spine_params_mean(X_train, y01):
    """
    Mean-diff spine for a binary set (y01 in {0,1}).
    Returns (m0, w_unit): m0 = mean of class 0; w_unit = (m1 - m0)/||...||
    """
    X0 = X_train[y01 == 0]
    X1 = X_train[y01 == 1]
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    w = (m1 - m0).astype(np.float64)
    nrm = np.linalg.norm(w)
    w_unit = w / (nrm + 1e-12)
    return m0, w_unit


def fit_pair_model_tree(X_train, y_train, a, b, lam=1.0, thr_mode="train_best"):
    """
    Train SVM On Tree for a pair (a,b) directly in 784D:
      - Take subset of classes a,b; map to {0,1} with b->1, a->0
      - Use C++ core 'fit_core' to select (s,p) by loss (for info)
      - Use mean-diff spine (784D), find threshold (train_best or from s,p midpoint)
    """
    mask = np.logical_or(y_train == a, y_train == b)
    Xab = X_train[mask]
    yab = y_train[mask]
    y01 = (yab == b).astype(np.int64)  # b -> 1, a -> 0

    # 1) loss-based support pair (s,p) from C++ core
    out = fit_core(Xab, y01, lam)
    s_idx = int(out["support_s"])
    p_idx = int(out["support_p"])

    # 2) spine in 784D
    m0, w_unit = spine_params_mean(Xab, y01)
    t_all = t_coords(Xab, m0, w_unit)

    # 3) choose threshold
    if thr_mode == "support":
        ts, tp = t_all[s_idx], t_all[p_idx]
        thr = 0.5 * (ts + tp)
        y_left, y_right = (0, 1) if ts <= tp else (1, 0)
    elif thr_mode == "train_best":
        thr, y_left, y_right = best_threshold_on_train(t_all, y01)
    else:
        raise ValueError("Unknown thr_mode. Use 'train_best' or 'support'.")

    return {
        "a": int(a), "b": int(b),
        "m0": m0,           # mean of class a (mapped to 0)
        "w_unit": w_unit,   # spine direction in 784D
        "thr": float(thr),
        "y_left": int(y_left),   # left->0, right->1 (on this binary mapping)
        "y_right": int(y_right)
    }


def predict_pair_with_margin(model, X):
    m0 = model["m0"]
    w_unit = model["w_unit"]
    thr = model["thr"]
    y_left = model["y_left"]
    y_right = model["y_right"]
    a, b = model["a"], model["b"]

    t = t_coords(X, m0, w_unit)
    side = (t >= thr).astype(np.int64)
    y01 = np.where(side == 0, y_left, y_right)  # mapped {0,1}
    y_ab = np.where(y01 == 0, a, b)             # back to {a,b}
    margin = np.abs(t - thr)                    # confidence for weighted vote
    return y_ab, margin


def train_tree_ovo(X_train, y_train, classes=None, lam=1.0, thr_mode="train_best"):
    """
    Train OVO: one pair-model per (a,b), a<b.
    """
    if classes is None:
        classes = sorted(set(int(v) for v in np.unique(y_train)))
    else:
        classes = sorted(set(int(v) for v in classes))
    classes = np.array(classes, dtype=np.int64)

    pair_models = []
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            a, b = int(classes[i]), int(classes[j])
            m = fit_pair_model_tree(X_train, y_train, a, b, lam=lam, thr_mode=thr_mode)
            pair_models.append(m)
    return {"classes": classes, "pair_models": pair_models}


def predict_tree_ovo_weighted(model_ovo, X):
    """
    Weighted voting by |t-thr| margin across OVO models.
    Tie-break by smallest class id.
    """
    classes = model_ovo["classes"]
    models = model_ovo["pair_models"]
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    votes = np.zeros((X.shape[0], len(classes)), dtype=np.float64)

    for m in models:
        y_ab, margin = predict_pair_with_margin(m, X)
        # accumulate margin to the predicted class
        for i, c in enumerate(y_ab):
            votes[i, class_to_idx[int(c)]] += float(margin[i])

    # tie-break: add tiny decreasing bias by class index to make argmax stable
    eps = 1e-12 * np.linspace(0.0, -1.0, votes.shape[1])
    idx = np.argmax(votes + eps, axis=1)
    return classes[idx]


# -----------------------------
# Benchmark wrapper
# -----------------------------
def run_benchmark(
    digits="all",
    per_class_train=2000,
    per_class_test=500,
    repeats=3,
    lam=1.0,
    thr_mode="train_best",
    standardize=False,
    seed=0
):
    print("=== SVM On Tree (OVO) on MNIST (784D) ===")
    print("Loading MNIST...")
    X, y = load_mnist()

    all_classes = sorted(set(int(v) for v in np.unique(y)))
    if digits == "all":
        classes = all_classes
    else:
        classes = sorted(set(int(t) for t in digits.split(",")))

    rng = np.random.default_rng(seed)
    print(f"Classes: {classes}")
    print(f"Sampling per class: train={per_class_train}, test={per_class_test}")

    X_train, y_train, X_test, y_test = sample_by_class(
        X, y, classes, per_class_train, per_class_test, rng
    )

    # Optional standardization for both train and test (often helps a bit)
    X_train_used, X_test_used, _ = maybe_standardize(X_train, X_test, enable=standardize)
    dim_used = X_train_used.shape[1]

    # ---- Fit (median over repeats)
    t_fit = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model = train_tree_ovo(X_train_used, y_train, classes=classes, lam=lam, thr_mode=thr_mode)
        t_fit.append(time.perf_counter() - t0)
    fit_med = median(t_fit)

    # ---- Predict (median over repeats)
    t_pred = []
    accs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        y_pred = predict_tree_ovo_weighted(model, X_test_used)
        t_pred.append(time.perf_counter() - t0)
        accs.append(accuracy_score(y_test, y_pred))
    pred_med = median(t_pred)
    acc_med = median(accs)

    # ---- Report
    print("\n=== Report ===")
    print(f"Classes: {classes}")
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]} | Dim used: {dim_used}")
    print(f"Repeats (median): {repeats}")
    print(f"Options: thr_mode={thr_mode}, lambda={lam}, standardize={standardize}\n")
    print(f"FIT   SVM-On-Tree (OVO): {fit_med:.3f}s")
    print(f"PRED  SVM-On-Tree (OVO): {pred_med:.3f}s")
    print(f"ACC   SVM-On-Tree (OVO): {acc_med:.4f}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--digits", type=str, default="all", help='e.g. "0,1,2" or "all"')
    ap.add_argument("--per-class-train", type=int, default=2000)
    ap.add_argument("--per-class-test", type=int, default=500)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--lamda", type=float, default=1.0)
    ap.add_argument("--threshold", type=str, default="train_best", choices=["train_best", "support"])
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    run_benchmark(
        digits=args.digits,
        per_class_train=args.per_class_train,
        per_class_test=args.per_class_test,
        repeats=args.repeats,
        lam=args.lamda,
        thr_mode=args.threshold,
        standardize=args.standardize,
        seed=args.seed
    )
