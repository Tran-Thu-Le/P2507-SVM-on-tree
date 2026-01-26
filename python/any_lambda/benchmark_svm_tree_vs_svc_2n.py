# import time
# import argparse
# import numpy as np
# from statistics import median
# from sklearn.svm import SVC
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# from svm_on_tree_cpp import fit_core  # C++ core


# # =========================
# # Data generator (parametric)
# # =========================
# def gen_data(
#     n,
#     seed=0,
#     sep=6.0,            # khoảng cách giữa 2 tâm: m0=(-sep/2,0), m1=(+sep/2,0)
#     sigma_para=2.5,     # std theo hướng spine (trục x)
#     sigma_perp=2.5,     # std vuông góc spine (trục y)
#     rho=0.0             # tương quan (giữa x và y)
# ):
#     """
#     Sinh 2 cụm Gaussian với:
#       - m0 = (-sep/2, 0), m1 = (+sep/2, 0)  -> spine ~ trục x
#       - covariance ~ [[sigma_para^2, rho*sigma_para*sigma_perp],
#                       [rho*sigma_para*sigma_perp, sigma_perp^2]]
#     Gợi ý: tăng sep, giảm sigma_para -> ít chồng lắp theo spine -> ACC tăng rõ.
#     """
#     rng = np.random.default_rng(seed)
#     mu0, mu1 = np.array([-sep/2.0, 0.0]), np.array([sep/2.0, 0.0])

#     Sigma = np.array([
#         [sigma_para**2,        rho * sigma_para * sigma_perp],
#         [rho * sigma_para * sigma_perp, sigma_perp**2       ]
#     ], dtype=np.float64)

#     X0 = rng.multivariate_normal(mu0, Sigma, n)
#     X1 = rng.multivariate_normal(mu1, Sigma, n)
#     X = np.vstack([X0, X1]).astype(np.float64)
#     y = np.hstack([np.zeros(n, np.int64), np.ones(n, np.int64)])
#     return X, y


# # =========================
# # Spine helpers
# # =========================
# def spine_params(X_train, y_train):
#     X0 = X_train[y_train == 0]; X1 = X_train[y_train == 1]
#     m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
#     w = (m1 - m0).astype(np.float64)
#     nrm = np.linalg.norm(w)
#     w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
#     return m0, w_unit

# def t_coords(X_any, m0, w_unit):
#     return (X_any - m0) @ w_unit

# def make_projection_points(X_train, y_train):
#     m0, w_unit = spine_params(X_train, y_train)
#     t_all = t_coords(X_train, m0, w_unit)
#     X_proj = m0 + np.outer(t_all, w_unit)
#     return X_proj, t_all


# # =========================
# # SVM On Tree — phân loại theo boundary đúng (Tree-Spine)
# # =========================
# def predict_spine_threshold_2n(X_eval, X_train, y_train, s_idx, p_idx):
#     m0, w_unit = spine_params(X_train, y_train)
#     t_train = t_coords(X_train, m0, w_unit)
#     t_eval  = t_coords(X_eval,  m0, w_unit)

#     ts, tp = t_train[s_idx], t_train[p_idx]
#     thr = 0.5 * (ts + tp)

#     if ts <= tp:
#         y_left, y_right = y_train[s_idx], y_train[p_idx]
#     else:
#         y_left, y_right = y_train[p_idx], y_train[s_idx]

#     return np.where(t_eval < thr, y_left, y_right).astype(np.int64)


# # =========================
# # One-run evaluators (2N only)
# # =========================
# def run_tree_2n_once(X, y, lam=1.0):
#     # Fit to get s, p
#     t0 = time.perf_counter()
#     out = fit_core(X, y, lam)
#     t_fit = time.perf_counter() - t0
#     s_idx, p_idx = int(out["s_orig"]), int(out["p_orig"])

#     # 2N eval set
#     X_proj, _ = make_projection_points(X, y)
#     X_eval = np.vstack([X, X_proj])
#     y_eval = np.hstack([y, y])

#     # Predict (Tree-Spine)
#     t1 = time.perf_counter()
#     y_pred = predict_spine_threshold_2n(X_eval, X_train=X, y_train=y, s_idx=s_idx, p_idx=p_idx)
#     t_pred = time.perf_counter() - t1
#     acc = accuracy_score(y_eval, y_pred)

#     return {
#         "fit_s": float(t_fit),
#         "pred_s": float(t_pred),
#         "acc": float(acc),
#         "support": (s_idx, p_idx),
#         "loss": float(out["min_val"]),
#         "eval_size": int(X_eval.shape[0]),
#     }

# def run_svc_2n_once(X, y):
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     t0 = time.perf_counter()
#     svc = SVC(kernel="rbf", C=1.0)
#     svc.fit(Xs, y)
#     t_fit = time.perf_counter() - t0

#     X_proj, _ = make_projection_points(X, y)
#     X_eval = np.vstack([X, X_proj])
#     y_eval = np.hstack([y, y])
#     X_eval_s = scaler.transform(X_eval)

#     t1 = time.perf_counter()
#     y_pred = svc.predict(X_eval_s)
#     t_pred = time.perf_counter() - t1
#     acc = accuracy_score(y_eval, y_pred)

#     return float(t_fit), float(t_pred), float(acc)


# # =========================
# # Benchmark (2N only)
# # =========================
# def bench_2n(sizes, repeats=7, lam=1.0, warmup=True, seed=0,
#              sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
#     print(f"(data) sep={sep}, sigma_para={sigma_para}, sigma_perp={sigma_perp}, rho={rho}")
#     for n in sizes:
#         X, y = gen_data(n, seed, sep=sep, sigma_para=sigma_para, sigma_perp=sigma_perp, rho=rho)

#         # Warm-up
#         if warmup:
#             _ = run_tree_2n_once(X, y, lam)
#             _ = run_svc_2n_once(X, y)

#         # Tree (median)
#         R_tree = [run_tree_2n_once(X, y, lam) for _ in range(repeats)]
#         fit_tree = median([r["fit_s"] for r in R_tree])
#         pred_tree = median([r["pred_s"] for r in R_tree])
#         acc_tree = median([r["acc"] for r in R_tree])
#         (s_idx, p_idx) = R_tree[-1]["support"]
#         loss = R_tree[-1]["loss"]
#         eval_size = R_tree[-1]["eval_size"]

#         # SVC (median)
#         fit_svc_list, pred_svc_list, acc_svc_list = [], [], []
#         for _ in range(repeats):
#             fs, ps, as_ = run_svc_2n_once(X, y)
#             fit_svc_list.append(fs); pred_svc_list.append(ps); acc_svc_list.append(as_)
#         fit_svc = median(fit_svc_list)
#         pred_svc = median(pred_svc_list)
#         acc_svc = median(acc_svc_list)

#         speedup_fit = fit_svc / fit_tree if fit_tree > 0 else float("inf")
#         speedup_pred = pred_svc / pred_tree if pred_tree > 0 else float("inf")

#         print(f"\nN={2*n:6d} on 2N (orig + projections) | Eval size = {eval_size}")
#         print(f"  FIT   SVM-On-Tree: {fit_tree:.6f}s   vs   sklearn SVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
#         print(f"  PRED  SVM-On-Tree: {pred_tree:.6f}s   vs   sklearn SVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
#         print(f"  ACC   SVM-On-Tree: {acc_tree:.4f}     vs   sklearn SVC: {acc_svc:.4f}")
#         print(f"  Support=(s={s_idx}, p={p_idx}), Loss={loss:.4f}")

#     print("\nGợi ý:")
#     print("- Tăng --sep hoặc giảm --sigma-para để giảm chồng lắp theo spine -> ACC tăng mạnh.")
#     print("- Giữ --rho=0 để đơn giản; --sigma-perp có thể lớn mà không hại Tree (cắt theo t).")
#     print("- So sánh công bằng hơn với SVC tuyến tính: thử thêm baseline linear (nếu cần).")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000])
#     parser.add_argument("--repeats", type=int, default=7)
#     parser.add_argument("--lamda", type=float, default=1.0)
#     parser.add_argument("--no-warmup", action="store_true")
#     parser.add_argument("--seed", type=int, default=0)
#     # knobs for separability
#     parser.add_argument("--sep", type=float, default=6.0, help="Khoảng cách giữa hai mean (m0=-sep/2, m1=+sep/2)")
#     parser.add_argument("--sigma-para", type=float, default=2.5, help="Std dọc spine (trục x)")
#     parser.add_argument("--sigma-perp", type=float, default=2.5, help="Std vuông góc spine (trục y)")
#     parser.add_argument("--rho", type=float, default=0.0, help="Tương quan giữa trục x và y")
#     args = parser.parse_args()

#     print("=== SVM On Tree (C++) vs sklearn SVC — Time & Accuracy on 2N only (parametric data) ===")
#     bench_2n(
#         args.sizes, repeats=args.repeats, lam=args.lamda, warmup=not args.no_warmup, seed=args.seed,
#         sep=args.sep, sigma_para=args.sigma_para, sigma_perp=args.sigma_perp, rho=args.rho
#     )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import time
# import argparse
# import numpy as np
# from statistics import median
# from sklearn.svm import LinearSVC   # Primal form – công bằng tốc độ
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# from svm_on_tree_cpp import fit_core  # C++ core: trả về t_dpdfs (DFS+DP only)

# # =========================
# # Data generator (parametric)
# # =========================
# def gen_data(
#     n,
#     seed=0,
#     sep=6.0,            # khoảng cách giữa 2 tâm: m0=(-sep/2,0), m1=(+sep/2,0)
#     sigma_para=2.5,     # std theo hướng spine (trục x)
#     sigma_perp=2.5,     # std vuông góc spine (trục y)
#     rho=0.0             # tương quan (giữa x và y)
# ):
#     rng = np.random.default_rng(seed)
#     mu0, mu1 = np.array([-sep/2.0, 0.0]), np.array([sep/2.0, 0.0])

#     Sigma = np.array([
#         [sigma_para**2,                  rho * sigma_para * sigma_perp],
#         [rho * sigma_para * sigma_perp,  sigma_perp**2               ]
#     ], dtype=np.float64)

#     X0 = rng.multivariate_normal(mu0, Sigma, n)
#     X1 = rng.multivariate_normal(mu1, Sigma, n)
#     X = np.vstack([X0, X1]).astype(np.float64)
#     y = np.hstack([np.zeros(n, np.int64), np.ones(n, np.int64)])
#     return X, y

# # =========================
# # Spine helpers
# # =========================
# def spine_params(X_train, y_train):
#     X0 = X_train[y_train == 0]; X1 = X_train[y_train == 1]
#     m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
#     w = (m1 - m0).astype(np.float64)
#     nrm = np.linalg.norm(w)
#     w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
#     return m0, w_unit

# def t_coords(X_any, m0, w_unit):
#     return (X_any - m0) @ w_unit

# def make_projection_points(X_train, y_train):
#     m0, w_unit = spine_params(X_train, y_train)
#     t_all = t_coords(X_train, m0, w_unit)
#     X_proj = m0 + np.outer(t_all, w_unit)
#     return X_proj, t_all

# # =========================
# # Tree prediction (spine threshold on 2N)
# # =========================
# def predict_spine_threshold_2n(X_eval, X_train, y_train, s_idx, p_idx):
#     """
#     Phân loại theo ngưỡng trung điểm trên trục spine.
#     s_idx, p_idx là index của mẫu gốc (original indices).
#     """
#     m0, w_unit = spine_params(X_train, y_train)
#     t_train = t_coords(X_train, m0, w_unit)
#     t_eval  = t_coords(X_eval,  m0, w_unit)

#     ts, tp = t_train[s_idx], t_train[p_idx]
#     thr = 0.5 * (ts + tp)

#     if ts <= tp:
#         y_left, y_right = y_train[s_idx], y_train[p_idx]
#     else:
#         y_left, y_right = y_train[p_idx], y_train[s_idx]

#     return np.where(t_eval < thr, y_left, y_right).astype(np.int64)

# # =========================
# # One-run evaluators (2N only)
# # =========================
# def run_tree_2n_once(X, y, lam=1.0):
#     # CHỈ lấy thời gian DFS+DP trong C++ (t_dpdfs) làm fit time
#     out = fit_core(X, y, lam)
#     t_fit = float(out.get("t_dpdfs", 0.0))  # DFS + DP only
#     s_idx, p_idx = int(out["s_orig"]), int(out["p_orig"])

#     # 2N eval set (X + X_proj)
#     X_proj, _ = make_projection_points(X, y)
#     X_eval = np.vstack([X, X_proj])
#     y_eval = np.hstack([y, y])

#     # Predict (Tree)
#     t1 = time.perf_counter()
#     y_pred = predict_spine_threshold_2n(X_eval, X_train=X, y_train=y, s_idx=s_idx, p_idx=p_idx)
#     t_pred = time.perf_counter() - t1
#     acc = accuracy_score(y_eval, y_pred)

#     return {
#         "fit_s": t_fit,                # chỉ DFS+DP
#         "pred_s": float(t_pred),
#         "acc": float(acc),
#         "support": (s_idx, p_idx),
#         "loss": float(out["min_val"]),
#         "eval_size": int(X_eval.shape[0]),
#     }

# def run_linearsvc_2n_once(X, y):
#     """
#     Baseline công bằng (primal): LinearSVC(dual=False).
#     Đánh giá trên 2N điểm (X + X_proj) như Tree để ACC so sánh công bằng.
#     """
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     t0 = time.perf_counter()
#     svc = LinearSVC(dual=False, C=1.0, fit_intercept=True, random_state=0, max_iter=10000)
#     svc.fit(Xs, y)
#     t_fit = time.perf_counter() - t0

#     # Eval 2N
#     X_proj, _ = make_projection_points(X, y)
#     X_eval = np.vstack([X, X_proj])
#     y_eval = np.hstack([y, y])
#     X_eval_s = scaler.transform(X_eval)

#     t1 = time.perf_counter()
#     y_pred = svc.predict(X_eval_s)
#     t_pred = time.perf_counter() - t1
#     acc = accuracy_score(y_eval, y_pred)

#     return float(t_fit), float(t_pred), float(acc)

# # =========================
# # Benchmark (2N only)
# # =========================
# def bench_2n(sizes, repeats=7, lam=1.0, warmup=True, seed=0,
#              sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
#     print(f"(data) sep={sep}, sigma_para={sigma_para}, sigma_perp={sigma_perp}, rho={rho}")
#     for n in sizes:
#         X, y = gen_data(n, seed, sep=sep, sigma_para=sigma_para, sigma_perp=sigma_perp, rho=rho)

#         # Warm-up
#         if warmup:
#             _ = run_tree_2n_once(X, y, lam)
#             _ = run_linearsvc_2n_once(X, y)

#         # Tree (median)
#         R_tree = [run_tree_2n_once(X, y, lam) for _ in range(repeats)]
#         fit_tree  = median([r["fit_s"] for r in R_tree])    # DFS+DP only
#         pred_tree = median([r["pred_s"] for r in R_tree])
#         acc_tree  = median([r["acc"]   for r in R_tree])
#         (s_idx, p_idx) = R_tree[-1]["support"]
#         loss = R_tree[-1]["loss"]
#         eval_size = R_tree[-1]["eval_size"]

#         # LinearSVC (median)
#         fit_svc_list, pred_svc_list, acc_svc_list = [], [], []
#         for _ in range(repeats):
#             fs, ps, as_ = run_linearsvc_2n_once(X, y)
#             fit_svc_list.append(fs); pred_svc_list.append(ps); acc_svc_list.append(as_)
#         fit_svc  = median(fit_svc_list)
#         pred_svc = median(pred_svc_list)
#         acc_svc  = median(acc_svc_list)

#         speedup_fit  = fit_svc / fit_tree if fit_tree > 0 else float("inf")
#         speedup_pred = pred_svc / pred_tree if pred_tree > 0 else float("inf")

#         print(f"\nN={2*n:6d} on 2N (orig + projections) | Eval size = {eval_size}")
#         print(f"  FIT   SVM-On-Tree (DFS+DP only): {fit_tree:.6f}s   vs   sklearn LinearSVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
#         print(f"  PRED  SVM-On-Tree: {pred_tree:.6f}s   vs   sklearn LinearSVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
#         print(f"  ACC   SVM-On-Tree: {acc_tree:.4f}       vs   sklearn LinearSVC: {acc_svc:.4f}")
#         print(f"  Support=(s={s_idx}, p={p_idx}), Loss={loss:.4f}")

#     print("\nGợi ý:")
#     print("- Nếu ưu tiên công bằng về tốc độ, LinearSVC (primal) là baseline phù hợp.")
#     print("- Nếu muốn so ACC với kernel mạnh hơn, bạn có thể chạy thêm SVC(RBF) ở file khác.")
#     print("- Tree: FIT là DFS+DP-only; các phần build/proj/CSR/scan-pairs không tính vào FIT.")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000])
#     parser.add_argument("--repeats", type=int, default=7)
#     parser.add_argument("--lamda", type=float, default=1.0)
#     parser.add_argument("--no-warmup", action="store_true")
#     parser.add_argument("--seed", type=int, default=0)
#     # knobs for separability
#     parser.add_argument("--sep", type=float, default=6.0, help="Khoảng cách giữa hai mean (m0=-sep/2, m1=+sep/2)")
#     parser.add_argument("--sigma-para", type=float, default=2.5, help="Std dọc spine (trục x)")
#     parser.add_argument("--sigma-perp", type=float, default=2.5, help="Std vuông góc spine (trục y)")
#     parser.add_argument("--rho", type=float, default=0.0, help="Tương quan giữa trục x và y")
#     args = parser.parse_args()

#     print("=== SVM On Tree (C++) vs sklearn LinearSVC (primal) — 2N eval ===")
#     bench_2n(
#         args.sizes, repeats=args.repeats, lam=args.lamda, warmup=not args.no_warmup, seed=args.seed,
#         sep=args.sep, sigma_para=args.sigma_para, sigma_perp=args.sigma_perp, rho=args.rho
#     )

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import time
# import argparse
# import numpy as np
# from statistics import median
# from sklearn.svm import LinearSVC   # dùng primal như case λ=1
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# from svm_on_tree_cpp import fit_core  # C++ core (đã đo t_dpdfs & t_pairs)

# # --------------------- Data --------------------- #
# def gen_data(n, seed=0, sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
#     rng = np.random.default_rng(seed)
#     mu0, mu1 = np.array([-sep/2.0, 0.0]), np.array([sep/2.0, 0.0])
#     Sigma = np.array([
#         [sigma_para**2,                  rho * sigma_para * sigma_perp],
#         [rho * sigma_para * sigma_perp,  sigma_perp**2               ]
#     ], dtype=np.float64)
#     X0 = rng.multivariate_normal(mu0, Sigma, n)
#     X1 = rng.multivariate_normal(mu1, Sigma, n)
#     X = np.vstack([X0, X1]).astype(np.float64)
#     y = np.hstack([np.zeros(n, np.int64), np.ones(n, np.int64)])
#     return X, y

# # --------------------- Spine helpers --------------------- #
# def spine_params(X_train, y_train):
#     X0 = X_train[y_train == 0]; X1 = X_train[y_train == 1]
#     m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
#     w = (m1 - m0).astype(np.float64)
#     nrm = np.linalg.norm(w)
#     w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
#     return m0, w_unit

# def t_coords(X_any, m0, w_unit): return (X_any - m0) @ w_unit

# def make_projection_points(X_train, y_train):
#     m0, w_unit = spine_params(X_train, y_train)
#     t_all = t_coords(X_train, m0, w_unit)
#     X_proj = m0 + np.outer(t_all, w_unit)
#     return X_proj, t_all

# # --------------------- Tree prediction (threshold on spine) --------------------- #
# def precompute_tree_params(X_train, y_train, s_idx, p_idx):
#     X0 = X_train[y_train == 0]; X1 = X_train[y_train == 1]
#     m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
#     w = (m1 - m0).astype(np.float64)
#     nrm = np.linalg.norm(w)
#     w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
#     t_train = (X_train - m0) @ w_unit
#     ts, tp = t_train[s_idx], t_train[p_idx]
#     thr = 0.5 * (ts + tp)
#     if ts <= tp:
#         y_left, y_right = y_train[s_idx], y_train[p_idx]
#     else:
#         y_left, y_right = y_train[p_idx], y_train[s_idx]
#     return dict(t_train=t_train, thr=float(thr), y_left=int(y_left), y_right=int(y_right))

# def predict_spine_threshold_2n_cached(params, N):
#     t_train = params["t_train"]; thr = params["thr"]
#     yl = params["y_left"]; yr = params["y_right"]
#     t_eval = np.concatenate([t_train, t_train], axis=0)  # 2N
#     return np.where(t_eval < thr, yl, yr).astype(np.int64)

# # --------------------- One run --------------------- #
# def run_tree_2n_once(X, y, lam=1.0):
#     # FIT (chỉ lấy t_dpdfs + t_pairs, đúng yêu cầu)
#     out = fit_core(X, y, lam)
#     t_fit = float(out["t_fit_core"])
#     t_dpdfs = float(out["t_dpdfs"])
#     t_pairs = float(out["t_pairs"])
#     s_idx, p_idx = int(out["s_orig"]), int(out["p_orig"])

#     # EVAL set 2N — dự đoán O(2N) bằng ngưỡng trên t_eval
#     N = X.shape[0]
#     y_eval = np.hstack([y, y])
#     params = precompute_tree_params(X, y, s_idx, p_idx)
#     t1 = time.perf_counter()
#     y_pred = predict_spine_threshold_2n_cached(params, N)
#     t_pred = time.perf_counter() - t1
#     acc = accuracy_score(y_eval, y_pred)

#     return {
#         "fit_s": t_fit,
#         "fit_dpdfs": t_dpdfs,
#         "fit_pairs": t_pairs,
#         "pred_s": float(t_pred),
#         "acc": float(acc),
#         "support": (s_idx, p_idx),
#         "loss": float(out["min_val"]),
#         "eval_size": int(2*N),
#     }

# def run_linearsvc_2n_once(X, y):
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)
#     t0 = time.perf_counter()
#     svc = LinearSVC(dual=False, C=1.0, fit_intercept=True, random_state=0, max_iter=10000)
#     svc.fit(Xs, y)
#     t_fit = time.perf_counter() - t0

#     X_proj, _ = make_projection_points(X, y)
#     X_eval = np.vstack([X, X_proj])
#     y_eval = np.hstack([y, y])
#     X_eval_s = scaler.transform(X_eval)
#     t1 = time.perf_counter()
#     y_pred = svc.predict(X_eval_s)
#     t_pred = time.perf_counter() - t1
#     acc = accuracy_score(y_eval, y_pred)
#     return float(t_fit), float(t_pred), float(acc)

# # --------------------- Benchmark --------------------- #
# def bench_2n(sizes, repeats=7, lam=1.0, warmup=True, seed=0,
#              sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
#     print(f"(data) sep={sep}, sigma_para={sigma_para}, sigma_perp={sigma_perp}, rho={rho}")
#     for n in sizes:
#         X, y = gen_data(n, seed, sep=sep, sigma_para=sigma_para, sigma_perp=sigma_perp, rho=rho)

#         if warmup:
#             _ = run_tree_2n_once(X, y, lam)
#             _ = run_linearsvc_2n_once(X, y)

#         R_tree = [run_tree_2n_once(X, y, lam) for _ in range(repeats)]
#         fit_tree      = median([r["fit_s"] for r in R_tree])
#         fit_dpdfs_med = median([r["fit_dpdfs"] for r in R_tree])
#         fit_pairs_med = median([r["fit_pairs"] for r in R_tree])
#         pred_tree     = median([r["pred_s"] for r in R_tree])
#         acc_tree      = median([r["acc"] for r in R_tree])
#         (s_idx, p_idx)= R_tree[-1]["support"]
#         loss          = R_tree[-1]["loss"]
#         eval_size     = R_tree[-1]["eval_size"]

#         fit_svc_list, pred_svc_list, acc_svc_list = [], [], []
#         for _ in range(repeats):
#             fs, ps, as_ = run_linearsvc_2n_once(X, y)
#             fit_svc_list.append(fs); pred_svc_list.append(ps); acc_svc_list.append(as_)
#         fit_svc  = median(fit_svc_list)
#         pred_svc = median(pred_svc_list)
#         acc_svc  = median(acc_svc_list)

#         speedup_fit  = fit_svc / fit_tree if fit_tree > 0 else float("inf")
#         speedup_pred = pred_svc / pred_tree if pred_tree > 0 else float("inf")

#         print(f"\nN={2*n:6d} on 2N (orig + projections) | Eval size = {eval_size} | λ={lam}")
#         print(f"  FIT   SVM-On-Tree: {fit_tree:.6f}s   (DP+DFS={fit_dpdfs_med:.6f}s, pairs={fit_pairs_med:.6f}s) "
#               f"vs   LinearSVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
#         print(f"  PRED  SVM-On-Tree: {pred_tree:.6f}s   vs   LinearSVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
#         print(f"  ACC   SVM-On-Tree: {acc_tree:.4f}     vs   LinearSVC: {acc_svc:.4f}")
#         print(f"  Support=(s={s_idx}, p={p_idx}), Loss={loss:.4f}")

#     print("\nGợi ý:")
#     print("- Thời gian FIT của λ≠1 đã gồm cả O(n^2) scan pairs (đảm bảo so sánh công bằng).")
#     print("- Có thể cố định số luồng BLAS khi so sklearn: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000])
#     parser.add_argument("--repeats", type=int, default=7)
#     parser.add_argument("--lamda", type=float, default=2.0)  # λ≠1 điển hình
#     parser.add_argument("--no-warmup", action="store_true")
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--sep", type=float, default=6.0)
#     parser.add_argument("--sigma-para", type=float, default=2.5)
#     parser.add_argument("--sigma-perp", type=float, default=2.5)
#     parser.add_argument("--rho", type=float, default=0.0)
#     args = parser.parse_args()

#     print("=== SVM On Tree (C++) vs sklearn LinearSVC — 2N eval (arbitrary λ) ===")
#     bench_2n(
#         args.sizes, repeats=args.repeats, lam=args.lamda, warmup=not args.no_warmup, seed=args.seed,
#         sep=args.sep, sigma_para=args.sigma_para, sigma_perp=args.sigma_perp, rho=args.rho
#     )

# if __name__ == "__main__":
#     main()

import time
import argparse
import numpy as np
from statistics import median

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# C++ core (lambda-any). Tên hàm giữ đúng theo project bạn.
from svm_on_tree_cpp import fit_core


# --------------------- Data (2 classes, 2D) --------------------- #
def gen_data(n_per_class, seed=0, sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
    rng = np.random.default_rng(seed)
    mu0, mu1 = np.array([-sep / 2.0, 0.0]), np.array([sep / 2.0, 0.0])
    Sigma = np.array(
        [
            [sigma_para**2, rho * sigma_para * sigma_perp],
            [rho * sigma_para * sigma_perp, sigma_perp**2],
        ],
        dtype=np.float64,
    )
    X0 = rng.multivariate_normal(mu0, Sigma, n_per_class)
    X1 = rng.multivariate_normal(mu1, Sigma, n_per_class)
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.hstack([np.zeros(n_per_class, np.int64), np.ones(n_per_class, np.int64)])
    return X, y


# --------------------- Eval set: 2N (X + X_proj) --------------------- #
def spine_params(X_train, y_train):
    X0 = X_train[y_train == 0]
    X1 = X_train[y_train == 1]
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    w = (m1 - m0).astype(np.float64)
    nrm = np.linalg.norm(w)
    w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
    return m0.astype(np.float64), w_unit.astype(np.float64)

def make_projection_points(X_train, y_train):
    m0, w_unit = spine_params(X_train, y_train)
    t_all = (X_train - m0) @ w_unit
    X_proj = m0 + np.outer(t_all, w_unit)
    return X_proj.astype(np.float64)

def build_eval_set_2n(X, y):
    X_proj = make_projection_points(X, y)
    X_eval = np.vstack([X, X_proj]).astype(np.float64)
    y_eval = np.hstack([y, y]).astype(np.int64)
    return X_eval, y_eval


# --------------------- Tree model builder --------------------- #
def build_tree_model_dict(out, X_train, y_train):
    """
    Ưu tiên dùng model params nếu C++ đã trả về:
      m0, w, thr, y_left, y_right
    Nếu chưa có, ta suy ra từ (support_s, support_p) (hoặc s_orig, p_orig)
    và các thống kê trên X_train (tính 1 lần, không tính vào pred timer).
    """
    # Case A: C++ đã trả thẳng model params
    if all(k in out for k in ["m0", "w", "thr", "y_left", "y_right"]):
        m0 = np.asarray(out["m0"], dtype=np.float64)
        w  = np.asarray(out["w"], dtype=np.float64)
        thr = float(out["thr"])
        yl = int(out["y_left"])
        yr = int(out["y_right"])
        return {"m0": m0, "w": w, "thr": thr, "y_left": yl, "y_right": yr}

    # Case B: suy ra từ support indices
    s_idx = None
    p_idx = None
    for a, b in [("support_s", "support_p"), ("s_orig", "p_orig")]:
        if a in out and b in out:
            s_idx = int(out[a]); p_idx = int(out[b])
            break
    if s_idx is None or p_idx is None:
        raise RuntimeError("fit_core output does not contain model params or support indices.")

    m0, w_unit = spine_params(X_train, y_train)
    t_train = (X_train - m0) @ w_unit
    ts, tp = float(t_train[s_idx]), float(t_train[p_idx])
    thr = 0.5 * (ts + tp)

    # labels on left/right depend on order along spine
    if ts <= tp:
        yl, yr = int(y_train[s_idx]), int(y_train[p_idx])
    else:
        yl, yr = int(y_train[p_idx]), int(y_train[s_idx])

    return {"m0": m0, "w": w_unit, "thr": float(thr), "y_left": yl, "y_right": yr}


# --------------------- Tree prediction (FAIR E2E) --------------------- #
def predict_tree_e2e(model, X_eval):
    m0 = model["m0"]
    w  = model["w"]
    thr = model["thr"]
    yl = model["y_left"]
    yr = model["y_right"]

    t_eval = (X_eval - m0) @ w
    return np.where(t_eval < thr, yl, yr).astype(np.int64)


# --------------------- One run (FAIR wall-clock) --------------------- #
def run_tree_once(X, y, lam):
    # FIT: wall-clock around C++ call
    t0 = time.perf_counter()
    out = fit_core(X, y, lam)
    t_fit = time.perf_counter() - t0

    # Build eval set outside pred timer (same for both methods)
    X_eval, y_eval = build_eval_set_2n(X, y)

    # Prepare model params outside pred timer (like sklearn has model ready after fit)
    model = build_tree_model_dict(out, X, y)

    # PRED: end-to-end prediction on X_eval (includes dot-products)
    t1 = time.perf_counter()
    y_pred = predict_tree_e2e(model, X_eval)
    t_pred = time.perf_counter() - t1

    acc = accuracy_score(y_eval, y_pred)

    # some optional info
    s_idx = None; p_idx = None
    if "support_s" in out and "support_p" in out:
        s_idx, p_idx = int(out["support_s"]), int(out["support_p"])
    elif "s_orig" in out and "p_orig" in out:
        s_idx, p_idx = int(out["s_orig"]), int(out["p_orig"])

    return {
        "fit_s": float(t_fit),
        "pred_s": float(t_pred),
        "acc": float(acc),
        "support": (s_idx, p_idx),
        "loss": float(out["min_val"]) if "min_val" in out else float("nan"),
        "scan_mode": str(out.get("scan_mode", "")),
    }


def run_linearsvc_once(X, y):
    # FIT
    t0 = time.perf_counter()
    clf = LinearSVC(dual=False, C=1.0, fit_intercept=True, random_state=0, max_iter=10000)
    clf.fit(X, y)
    t_fit = time.perf_counter() - t0

    # Eval set outside pred timer (same definition as Tree)
    X_eval, y_eval = build_eval_set_2n(X, y)

    # PRED
    t1 = time.perf_counter()
    y_pred = clf.predict(X_eval)
    t_pred = time.perf_counter() - t1

    acc = accuracy_score(y_eval, y_pred)
    return float(t_fit), float(t_pred), float(acc)


# --------------------- Benchmark --------------------- #
def bench_2n(
    sizes,
    repeats=7,
    lam=2.0,
    warmup=True,
    seed=0,
    standardize=0,
    sep=6.0,
    sigma_para=2.5,
    sigma_perp=2.5,
    rho=0.0,
):
    print(f"(data) sep={sep}, sigma_para={sigma_para}, sigma_perp={sigma_perp}, rho={rho}")
    print(f"(timing) FIT=wall clock, PRED=end-to-end on 2N (includes dot-products for Tree)")
    print(f"(standardize) {standardize}  (applied to BOTH methods if enabled)")

    for n_per_class in sizes:
        X, y = gen_data(n_per_class, seed, sep=sep, sigma_para=sigma_para, sigma_perp=sigma_perp, rho=rho)

        # Optional standardize (FAIR: do it ONCE here, shared by both)
        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X).astype(np.float64)

        N_total = X.shape[0]
        eval_size = 2 * N_total

        if warmup:
            _ = run_tree_once(X, y, lam)
            _ = run_linearsvc_once(X, y)

        R_tree = [run_tree_once(X, y, lam) for _ in range(repeats)]
        fit_tree  = median([r["fit_s"] for r in R_tree])
        pred_tree = median([r["pred_s"] for r in R_tree])
        acc_tree  = median([r["acc"] for r in R_tree])
        s_idx, p_idx = R_tree[-1]["support"]
        loss = R_tree[-1]["loss"]
        scan_mode = R_tree[-1]["scan_mode"]

        fit_svc_list, pred_svc_list, acc_svc_list = [], [], []
        for _ in range(repeats):
            fs, ps, ac = run_linearsvc_once(X, y)
            fit_svc_list.append(fs)
            pred_svc_list.append(ps)
            acc_svc_list.append(ac)

        fit_svc  = median(fit_svc_list)
        pred_svc = median(pred_svc_list)
        acc_svc  = median(acc_svc_list)

        speedup_fit  = fit_svc / fit_tree if fit_tree > 0 else float("inf")
        speedup_pred = pred_svc / pred_tree if pred_tree > 0 else float("inf")

        print(f"\n{'='*70}")
        print(f"N={N_total:6d} (train) | Eval = 2N => {eval_size:6d} | repeats={repeats}, lamda={lam}")
        print(f"{'='*70}")
        print(f"FIT   Tree(wall): {fit_tree:.6f}s   vs   LinearSVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
        print(f"PRED  Tree(E2E):  {pred_tree:.6f}s   vs   LinearSVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
        print(f"ACC   Tree:       {acc_tree:.4f}     vs   LinearSVC: {acc_svc:.4f}")
        print(f"Support=(s={s_idx}, p={p_idx}), Loss={loss:.6f}, Scan={scan_mode}")

    print("\nNotes:")
    print("- If you want maximum repeatability, pin threads:")
    print("  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1")
    print("- For λ≠1, your C++ core may use O(n^2) scan; that should be included in FIT(wall) automatically.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000])
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

    print("=== FAIR Benchmark (arbitrary λ): SVM On Tree (C++/pybind) vs sklearn LinearSVC on 2N eval ===")
    print(f"repeats={args.repeats}, lamda={args.lamda}, seed={args.seed}, warmup={not args.no_warmup}")
    print("Threads pinned recommended: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1")

    bench_2n(
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

