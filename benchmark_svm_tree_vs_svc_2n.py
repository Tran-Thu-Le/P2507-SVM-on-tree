# Benchmark: SVM On Tree vs LinearSVC on synthetic 2-class Gaussian data.
# Train and test sets are generated independently (same distribution, different seed).
# Timing is measured as wall-clock seconds (median over multiple repeats).

import os
import time
import argparse
import numpy as np
from statistics import median

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from svm_on_tree_cpp import fit_core


def gen_data(n_per_class, seed=0, sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
    """Generate 2-class Gaussian data in 2D. Total N = 2 * n_per_class."""
    rng = np.random.default_rng(seed)
    mu0 = np.array([-sep / 2.0, 0.0])
    mu1 = np.array([sep / 2.0, 0.0])
    Sigma = np.array(
        [[sigma_para**2, rho * sigma_para * sigma_perp],
         [rho * sigma_para * sigma_perp, sigma_perp**2]],
        dtype=np.float64,
    )
    X0 = rng.multivariate_normal(mu0, Sigma, n_per_class)
    X1 = rng.multivariate_normal(mu1, Sigma, n_per_class)
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.hstack([np.zeros(n_per_class, np.int64), np.ones(n_per_class, np.int64)])
    return X, y


def spine_params(X_train, y_train):
    """Compute spine origin (m0) and unit direction (w) from class means."""
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    w = (m1 - m0).astype(np.float64)
    nrm = np.linalg.norm(w)
    w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
    return m0.astype(np.float64), w_unit.astype(np.float64)


def build_tree_model_dict(out, X_train, y_train):
    """Extract threshold model from fit_core output."""
    # If C++ returns model parameters directly
    if all(k in out for k in ["m0", "w", "thr", "y_left", "y_right"]):
        m0 = np.asarray(out["m0"], dtype=np.float64)
        w = np.asarray(out["w"], dtype=np.float64)
        thr = float(out["thr"])
        yl = int(out["y_left"])
        yr = int(out["y_right"])
        return {"m0": m0, "w": w, "thr": thr, "y_left": yl, "y_right": yr}

    # Otherwise derive from support indices
    s_idx = p_idx = None
    for a, b in [("support_s", "support_p"), ("s_orig", "p_orig")]:
        if a in out and b in out:
            s_idx = int(out[a])
            p_idx = int(out[b])
            break
    if s_idx is None or p_idx is None:
        raise RuntimeError("fit_core output has no model params or support indices.")

    m0, w_unit = spine_params(X_train, y_train)
    t_train = (X_train - m0) @ w_unit
    ts = float(t_train[s_idx])
    tp = float(t_train[p_idx])
    thr = 0.5 * (ts + tp)

    if ts <= tp:
        yl, yr = int(y_train[s_idx]), int(y_train[p_idx])
    else:
        yl, yr = int(y_train[p_idx]), int(y_train[s_idx])

    return {"m0": m0, "w": w_unit, "thr": float(thr), "y_left": yl, "y_right": yr}


def predict_tree(model, X_eval):
    """Classify by thresholding the projection onto the spine direction."""
    t_eval = (X_eval - model["m0"]) @ model["w"]
    return np.where(t_eval < model["thr"], model["y_left"], model["y_right"]).astype(np.int64)


def run_tree_once(X_train, y_train, X_test, y_test, lam):
    """Single run of SVM On Tree: fit on train, predict on test."""
    t0 = time.perf_counter()
    out = fit_core(X_train, y_train, lam)
    t_fit = time.perf_counter() - t0

    model = build_tree_model_dict(out, X_train, y_train)

    t1 = time.perf_counter()
    y_pred = predict_tree(model, X_test)
    t_pred = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)

    s_idx = p_idx = None
    for a, b in [("support_s", "support_p"), ("s_orig", "p_orig")]:
        if a in out and b in out:
            s_idx, p_idx = int(out[a]), int(out[b])
            break

    return {
        "fit_s": float(t_fit),
        "pred_s": float(t_pred),
        "acc": float(acc),
        "support": (s_idx, p_idx),
        "loss": float(out["min_val"]) if "min_val" in out else float("nan"),
        "scan_mode": str(out.get("scan_mode", "")),
    }


def run_linearsvc_once(X_train, y_train, X_test, y_test):
    """Single run of LinearSVC: fit on train, predict on test."""
    t0 = time.perf_counter()
    clf = LinearSVC(dual=False, C=1.0, fit_intercept=True, random_state=0, max_iter=10000)
    clf.fit(X_train, y_train)
    t_fit = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = clf.predict(X_test)
    t_pred = time.perf_counter() - t1

    acc = accuracy_score(y_test, y_pred)
    return float(t_fit), float(t_pred), float(acc)


def bench(sizes, repeats=7, lam=1.0, warmup=True, seed=0,
          standardize=0, sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
    """Run benchmark across multiple dataset sizes."""
    TEST_SEED_OFFSET = 1_000_000

    print(f"(data) sep={sep}, sigma_para={sigma_para}, sigma_perp={sigma_perp}, rho={rho}")
    print(f"(eval) Test set = freshly generated (same distribution, different seed)")
    print(f"(timing) FIT=wall clock on train, PRED=on unseen test set")
    print(f"(standardize) {standardize}  (applied to BOTH methods if enabled)")

    for n_per_class in sizes:
        # Generate train and test sets
        X_train, y_train = gen_data(
            n_per_class, seed=seed,
            sep=sep, sigma_para=sigma_para, sigma_perp=sigma_perp, rho=rho,
        )
        X_test, y_test = gen_data(
            n_per_class, seed=seed + TEST_SEED_OFFSET,
            sep=sep, sigma_para=sigma_para, sigma_perp=sigma_perp, rho=rho,
        )

        # Standardize if requested (fit on train, transform both)
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train).astype(np.float64)
            X_test = scaler.transform(X_test).astype(np.float64)

        N_train = X_train.shape[0]
        N_test = X_test.shape[0]

        # Warm-up runs (discard results)
        if warmup:
            _ = run_tree_once(X_train, y_train, X_test, y_test, lam)
            _ = run_linearsvc_once(X_train, y_train, X_test, y_test)

        # SVM On Tree: repeat and take median
        R_tree = [run_tree_once(X_train, y_train, X_test, y_test, lam) for _ in range(repeats)]
        fit_tree = median([r["fit_s"] for r in R_tree])
        pred_tree = median([r["pred_s"] for r in R_tree])
        acc_tree = median([r["acc"] for r in R_tree])
        s_idx, p_idx = R_tree[-1]["support"]
        loss = R_tree[-1]["loss"]
        scan_mode = R_tree[-1]["scan_mode"]

        # LinearSVC: repeat and take median
        fit_svc_list, pred_svc_list, acc_svc_list = [], [], []
        for _ in range(repeats):
            fs, ps, ac = run_linearsvc_once(X_train, y_train, X_test, y_test)
            fit_svc_list.append(fs)
            pred_svc_list.append(ps)
            acc_svc_list.append(ac)

        fit_svc = median(fit_svc_list)
        pred_svc = median(pred_svc_list)
        acc_svc = median(acc_svc_list)

        speedup_fit = fit_svc / fit_tree if fit_tree > 0 else float("inf")
        speedup_pred = pred_svc / pred_tree if pred_tree > 0 else float("inf")

        print(f"\nTrain={N_train:6d} | Test={N_test:6d} (unseen) | repeats={repeats}, lamda={lam}")
        print(f"FIT   Tree(wall): {fit_tree:.6f}s   vs   LinearSVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
        print(f"PRED  Tree:       {pred_tree:.6f}s   vs   LinearSVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
        print(f"ACC   Tree:       {acc_tree:.4f}     vs   LinearSVC: {acc_svc:.4f}")
        print(f"Support=(s={s_idx}, p={p_idx}), Loss={loss:.6f}, Scan={scan_mode}")

    print("\nNotes:")
    print("- Test set is freshly generated with a DIFFERENT seed (never seen during training).")
    print("- Pin threads for repeatability:")
    print("  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: SVM On Tree vs LinearSVC (proper train/test split)"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[100, 200, 400, 1000, 2000, 5000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--standardize", type=int, default=0, choices=[0, 1])
    parser.add_argument("--sep", type=float, default=6.0)
    parser.add_argument("--sigma-para", type=float, default=2.5)
    parser.add_argument("--sigma-perp", type=float, default=2.5)
    parser.add_argument("--rho", type=float, default=0.0)
    args = parser.parse_args()

    print("Benchmark: SVM On Tree vs LinearSVC (proper train/test split)")
    print(f"repeats={args.repeats}, lamda={args.lamda}, seed={args.seed}, warmup={not args.no_warmup}")
    print("Threads pinned recommended: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1")

    bench(
        args.sizes,
        repeats=args.repeats,
        lam=args.lamda,
        warmup=not args.no_warmup,
        seed=args.seed,
        standardize=args.standardize,
        sep=args.sep,
        sigma_para=args.sigma_para,
        sigma_perp=args.sigma_perp,
        rho=args.rho,
    )


if __name__ == "__main__":
    main()

