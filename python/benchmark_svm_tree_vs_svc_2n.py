# import time
# import argparse
# import numpy as np
# from statistics import median
# from sklearn.svm import LinearSVC  # Primal form - công bằng hơn
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
#     s_idx, p_idx = int(out["support_s"]), int(out["support_p"])

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
#     # Sử dụng LinearSVC (primal form) thay vì SVC (dual form) để so sánh công bằng
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
#         print(f"  FIT   SVM-On-Tree: {fit_tree:.6f}s   vs   sklearn LinearSVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
#         print(f"  PRED  SVM-On-Tree: {pred_tree:.6f}s   vs   sklearn LinearSVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
#         print(f"  ACC   SVM-On-Tree: {acc_tree:.4f}     vs   sklearn LinearSVC: {acc_svc:.4f}")
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

#     print("=== SVM On Tree (C++) vs sklearn LinearSVC (primal) — Fair Comparison on 2N data ===")
#     bench_2n(
#         args.sizes, repeats=args.repeats, lam=args.lamda, warmup=not args.no_warmup, seed=args.seed,
#         sep=args.sep, sigma_para=args.sigma_para, sigma_perp=args.sigma_perp, rho=args.rho
#     )

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# """
# Benchmark công bằng giữa:
# - SVM On Tree (C++ core qua pybind11) với dự đoán dạng spine-threshold O(|X_eval|)
# - sklearn LinearSVC (primal) với dự đoán O(|X_eval| * d)

# Điểm chính sửa:
# - Cache m0, w_unit, t_train, thr, y_left, y_right sau fit -> prediction của Tree
#   chỉ còn so sánh ngưỡng trên t_eval = [t_train; t_train], không dot lại.
# """

# import time
# import argparse
# import numpy as np
# from statistics import median
# from sklearn.svm import LinearSVC       # Primal form – fair baseline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score

# from svm_on_tree_cpp import fit_core    # C++ core: trả về support_s, support_p, min_val


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
#     """
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
#     """
#     Dùng cho tập EVAL của LinearSVC để giống setup 2N (X + X_proj).
#     Với Tree, không cần tạo X_proj để dự đoán vì t_eval = [t_train; t_train].
#     """
#     m0, w_unit = spine_params(X_train, y_train)
#     t_all = t_coords(X_train, m0, w_unit)
#     X_proj = m0 + np.outer(t_all, w_unit)
#     return X_proj, t_all


# # =========================
# # SVM On Tree — cache & predict
# # =========================
# def precompute_tree_params(X_train, y_train, s_idx, p_idx):
#     """
#     Cache mọi thứ phụ thuộc TRAIN để prediction không lẫn chi phí O(N·d):
#     - m0, w_unit
#     - t_train
#     - threshold thr và nhãn phía trái/phải
#     """
#     X0 = X_train[y_train == 0]; X1 = X_train[y_train == 1]
#     m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
#     w = (m1 - m0).astype(np.float64)
#     nrm = np.linalg.norm(w)
#     w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)

#     t_train = (X_train - m0) @ w_unit  # (N,)

#     ts, tp = t_train[s_idx], t_train[p_idx]
#     thr = 0.5 * (ts + tp)
#     if ts <= tp:
#         y_left, y_right = y_train[s_idx], y_train[p_idx]
#     else:
#         y_left, y_right = y_train[p_idx], y_train[s_idx]

#     return dict(m0=m0, w_unit=w_unit, t_train=t_train,
#                 thr=float(thr), y_left=int(y_left), y_right=int(y_right))

# def predict_spine_threshold_2n_cached(params, N):
#     """
#     Eval set = concat(X, X_proj) → t_eval = concat(t_train, t_train)
#     Trả về y_pred cho 2N mẫu, không dot thêm lần nào.
#     """
#     t_train = params["t_train"]
#     thr = params["thr"]; yl = params["y_left"]; yr = params["y_right"]
#     # (2N,) – X_proj có cùng tọa độ t với X_train
#     t_eval = np.concatenate([t_train, t_train], axis=0)
#     return np.where(t_eval < thr, yl, yr).astype(np.int64)


# # =========================
# # One-run evaluators (2N only)
# # =========================
# def run_tree_2n_once(X, y, lam=1.0):
#     # FIT (C++)
#     t0 = time.perf_counter()
#     out = fit_core(X, y, lam)
#     t_fit = time.perf_counter() - t0
#     s_idx, p_idx = int(out["support_s"]), int(out["support_p"])

#     # Precompute 1 lần cho prediction
#     params = precompute_tree_params(X, y, s_idx, p_idx)

#     # Eval = 2N điểm (X và X_proj) — không cần dựng X_proj để dự đoán
#     N = X.shape[0]
#     y_eval = np.hstack([y, y])  # nhãn của X và X_proj giống nhau

#     # PRED – chỉ so sánh ngưỡng, không dot nào phụ thuộc TRAIN
#     t1 = time.perf_counter()
#     y_pred = predict_spine_threshold_2n_cached(params, N)
#     t_pred = time.perf_counter() - t1

#     acc = accuracy_score(y_eval, y_pred)
#     return {
#         "fit_s": float(t_fit),
#         "pred_s": float(t_pred),
#         "acc": float(acc),
#         "support": (s_idx, p_idx),
#         "loss": float(out["min_val"]),
#         "eval_size": int(2 * N),
#     }

# def run_linearsvc_2n_once(X, y):
#     """
#     Baseline công bằng: LinearSVC (primal). Đánh giá trên 2N điểm (X + X_proj)
#     như các thí nghiệm trước đây để ACC so sánh cùng điều kiện.
#     Việc dựng X_proj nằm ngoài timer prediction để không ảnh hưởng thời gian dự đoán.
#     """
#     scaler = StandardScaler()
#     Xs = scaler.fit_transform(X)

#     t0 = time.perf_counter()
#     svc = LinearSVC(dual=False, C=1.0, fit_intercept=True, random_state=0, max_iter=10000)
#     svc.fit(Xs, y)
#     t_fit = time.perf_counter() - t0

#     # Lập tập eval 2N (X + X_proj) – dựng X_proj TRƯỚC khi bấm giờ prediction
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

#         # Warm-up để “làm nóng” BLAS/JIT
#         if warmup:
#             _ = run_tree_2n_once(X, y, lam)
#             _ = run_linearsvc_2n_once(X, y)

#         # Tree (median)
#         R_tree = [run_tree_2n_once(X, y, lam) for _ in range(repeats)]
#         fit_tree = median([r["fit_s"] for r in R_tree])
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
#         print(f"  FIT   SVM-On-Tree: {fit_tree:.6f}s   vs   sklearn LinearSVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
#         print(f"  PRED  SVM-On-Tree: {pred_tree:.6f}s   vs   sklearn LinearSVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
#         print(f"  ACC   SVM-On-Tree: {acc_tree:.4f}     vs   sklearn LinearSVC: {acc_svc:.4f}")
#         print(f"  Support=(s={s_idx}, p={p_idx}), Loss={loss:.4f}")

#     print("\nGợi ý:")
#     print("- Tăng --sep hoặc giảm --sigma-para để giảm chồng lắp theo spine -> ACC tăng rõ.")
#     print("- Giữ --rho=0 để đơn giản; --sigma-perp có thể lớn mà không hại Tree (cắt theo t).")
#     print("- Có thể cố định số luồng BLAS: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1.")


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

#     print("=== SVM On Tree (C++) vs sklearn LinearSVC (primal) — Fair Comparison on 2N data ===")
#     bench_2n(
#         args.sizes, repeats=args.repeats, lam=args.lamda, warmup=not args.no_warmup, seed=args.seed,
#         sep=args.sep, sigma_para=args.sigma_para, sigma_perp=args.sigma_perp, rho=args.rho
#     )


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark công bằng giữa:
- SVM On Tree (C++ core qua pybind11) — PRED đã cache (siêu nhanh).
- sklearn LinearSVC (primal).

LƯU Ý:
- FIT của Tree trong file này = đúng "DFS + Quy hoạch động" (không gồm means/proj, không sort spine,
  không build CSR, không scan cặp biên). Ta lấy trực tiếp out["time_dp"] từ C++.
- PRED giữ nguyên như trước (chỉ so ngưỡng trên t_eval = [t_train; t_train]).
"""

import time
import argparse
import numpy as np
from statistics import median
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from svm_on_tree_cpp import fit_core  # C++ core (đã instrument trả về time_dp, v.v.)

# =========================
# Data generator (parametric)
# =========================
def gen_data(
    n,
    seed=0,
    sep=6.0,
    sigma_para=2.5,
    sigma_perp=2.5,
    rho=0.0
):
    """
    Sinh 2 cụm Gaussian:
      - m0 = (-sep/2, 0), m1 = (+sep/2, 0)  -> spine ~ trục x
      - covariance = [[sigma_para^2, rho*sigma_para*sigma_perp],
                      [rho*sigma_para*sigma_perp, sigma_perp^2]]
    """
    rng = np.random.default_rng(seed)
    mu0, mu1 = np.array([-sep/2.0, 0.0]), np.array([sep/2.0, 0.0])

    Sigma = np.array([
        [sigma_para**2,                  rho * sigma_para * sigma_perp],
        [rho * sigma_para * sigma_perp,  sigma_perp**2               ]
    ], dtype=np.float64)

    X0 = rng.multivariate_normal(mu0, Sigma, n)
    X1 = rng.multivariate_normal(mu1, Sigma, n)
    X = np.vstack([X0, X1]).astype(np.float64)
    y = np.hstack([np.zeros(n, np.int64), np.ones(n, np.int64)])
    return X, y


# =========================
# Spine helpers
# =========================
def spine_params(X_train, y_train):
    X0 = X_train[y_train == 0]; X1 = X_train[y_train == 1]
    m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
    w = (m1 - m0).astype(np.float64)
    nrm = np.linalg.norm(w)
    w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)
    return m0, w_unit

def t_coords(X_any, m0, w_unit):
    return (X_any - m0) @ w_unit

def make_projection_points(X_train, y_train):
    """
    Dùng cho tập EVAL của LinearSVC để giống setup 2N (X + X_proj).
    Với Tree, không cần tạo X_proj để dự đoán vì t_eval = [t_train; t_train].
    """
    m0, w_unit = spine_params(X_train, y_train)
    t_all = t_coords(X_train, m0, w_unit)
    X_proj = m0 + np.outer(t_all, w_unit)
    return X_proj, t_all


# =========================
# SVM On Tree — cache & predict
# =========================
def precompute_tree_params(X_train, y_train, s_idx, p_idx):
    """
    Cache m0, w_unit, t_train, và ngưỡng để prediction không mang chi phí O(N·d) phụ thuộc TRAIN.
    """
    X0 = X_train[y_train == 0]; X1 = X_train[y_train == 1]
    m0 = X0.mean(axis=0); m1 = X1.mean(axis=0)
    w = (m1 - m0).astype(np.float64)
    nrm = np.linalg.norm(w)
    w_unit = w / (nrm + 1e-12) if nrm > 0 else np.array([1.0, 0.0], dtype=np.float64)

    t_train = (X_train - m0) @ w_unit  # (N,)

    ts, tp = t_train[s_idx], t_train[p_idx]
    thr = 0.5 * (ts + tp)
    if ts <= tp:
        y_left, y_right = y_train[s_idx], y_train[p_idx]
    else:
        y_left, y_right = y_train[p_idx], y_train[s_idx]

    return dict(m0=m0, w_unit=w_unit, t_train=t_train,
                thr=float(thr), y_left=int(y_left), y_right=int(y_right))

def predict_spine_threshold_2n_cached(params, N):
    """
    Eval set = concat(X, X_proj) → t_eval = concat(t_train, t_train)
    Trả về y_pred cho 2N mẫu, không dot thêm lần nào.
    """
    t_train = params["t_train"]
    thr = params["thr"]; yl = params["y_left"]; yr = params["y_right"]
    t_eval = np.concatenate([t_train, t_train], axis=0)  # (2N,)
    return np.where(t_eval < thr, yl, yr).astype(np.int64)


# =========================
# One-run evaluators (2N only)
# =========================
def run_tree_2n_once(X, y, lam=1.0):
    # FIT (C++): lấy chỉ time_dp (DFS + Dynamic Programming) từ C++
    out = fit_core(X, y, lam)
    t_fit = float(out["time_dp"])  # Chỉ DFS + DP, không gồm preprocessing
    s_idx, p_idx = int(out["support_s"]), int(out["support_p"])

    # Precompute 1 lần cho prediction
    params = precompute_tree_params(X, y, s_idx, p_idx)

    # Eval = 2N (X và X_proj), nhưng Tree không cần dựng X_proj để dự đoán
    N = X.shape[0]
    y_eval = np.hstack([y, y])

    # PRED – chỉ so sánh ngưỡng, giữ nguyên như trước
    t1 = time.perf_counter()
    y_pred = predict_spine_threshold_2n_cached(params, N)
    t_pred = time.perf_counter() - t1

    acc = accuracy_score(y_eval, y_pred)
    return {
        "fit_s": t_fit,
        "pred_s": float(t_pred),
        "acc": float(acc),
        "support": (s_idx, p_idx),
        "loss": float(out["min_val"]),
        "eval_size": int(2 * N),
    }

def run_linearsvc_2n_once(X, y):
    """
    Baseline công bằng: LinearSVC (primal). Đánh giá trên 2N điểm (X + X_proj).
    Dựng X_proj TRƯỚC timer predict để không ảnh hưởng thời gian dự đoán.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    t0 = time.perf_counter()
    svc = LinearSVC(dual=False, C=1.0, fit_intercept=True, random_state=0, max_iter=10000)
    svc.fit(Xs, y)
    t_fit = time.perf_counter() - t0

    X_proj, _ = make_projection_points(X, y)
    X_eval = np.vstack([X, X_proj])
    y_eval = np.hstack([y, y])
    X_eval_s = scaler.transform(X_eval)

    t1 = time.perf_counter()
    y_pred = svc.predict(X_eval_s)
    t_pred = time.perf_counter() - t1
    acc = accuracy_score(y_eval, y_pred)

    return float(t_fit), float(t_pred), float(acc)


# =========================
# Benchmark (2N only)
# =========================
def bench_2n(sizes, repeats=7, lam=1.0, warmup=True, seed=0,
             sep=6.0, sigma_para=2.5, sigma_perp=2.5, rho=0.0):
    print(f"(data) sep={sep}, sigma_para={sigma_para}, sigma_perp={sigma_perp}, rho={rho}")
    for n in sizes:
        X, y = gen_data(n, seed, sep=sep, sigma_para=sigma_para, sigma_perp=sigma_perp, rho=rho)

        # Warm-up
        if warmup:
            _ = run_tree_2n_once(X, y, lam)
            _ = run_linearsvc_2n_once(X, y)

        # Tree (median)
        R_tree = [run_tree_2n_once(X, y, lam) for _ in range(repeats)]
        fit_tree = median([r["fit_s"] for r in R_tree])
        pred_tree = median([r["pred_s"] for r in R_tree])
        acc_tree  = median([r["acc"]   for r in R_tree])
        (s_idx, p_idx) = R_tree[-1]["support"]
        loss = R_tree[-1]["loss"]
        eval_size = R_tree[-1]["eval_size"]

        # LinearSVC (median)
        fit_svc_list, pred_svc_list, acc_svc_list = [], [], []
        for _ in range(repeats):
            fs, ps, as_ = run_linearsvc_2n_once(X, y)
            fit_svc_list.append(fs); pred_svc_list.append(ps); acc_svc_list.append(as_)
        fit_svc  = median(fit_svc_list)
        pred_svc = median(pred_svc_list)
        acc_svc  = median(acc_svc_list)

        speedup_fit  = fit_svc / fit_tree if fit_tree > 0 else float("inf")
        speedup_pred = pred_svc / pred_tree if pred_tree > 0 else float("inf")

        print(f"\nN={2*n:6d} on 2N (orig + projections) | Eval size = {eval_size}")
        print(f"  FIT   SVM-On-Tree (DFS+DP only): {fit_tree:.6f}s   vs   LinearSVC: {fit_svc:.6f}s   | Speedup FIT: {speedup_fit:.2f}x")
        print(f"  PRED  SVM-On-Tree: {pred_tree:.6f}s   vs   LinearSVC: {pred_svc:.6f}s   | Speedup PRED: {speedup_pred:.2f}x")
        print(f"  ACC   SVM-On-Tree: {acc_tree:.4f}       vs   LinearSVC: {acc_svc:.4f}")
        print(f"  Support=(s={s_idx}, p={p_idx}), Loss={loss:.4f}")

    print("\nGợi ý:")
    print("- Tăng --sep hoặc giảm --sigma-para để giảm chồng lắp theo spine -> ACC tăng rõ.")
    print("- Giữ --rho=0 để đơn giản; --sigma-perp có thể lớn mà không hại Tree (cắt theo t).")
    print("- Có thể cố định số luồng BLAS: OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[50, 100, 200, 500, 1000, 2000, 4000])
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--lamda", type=float, default=1.0)  # giữ tham số CLI cũ cho thuận tay
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    # knobs for separability
    parser.add_argument("--sep", type=float, default=6.0, help="Khoảng cách giữa hai mean (m0=-sep/2, m1=+sep/2)")
    parser.add_argument("--sigma-para", type=float, default=2.5, help="Std dọc spine (trục x)")
    parser.add_argument("--sigma-perp", type=float, default=2.5, help="Std vuông góc spine (trục y)")
    parser.add_argument("--rho", type=float, default=0.0, help="Tương quan giữa trục x và y")
    args = parser.parse_args()

    print("=== SVM On Tree (C++) vs sklearn LinearSVC (primal) — 2N eval ===")
    bench_2n(
        args.sizes, repeats=args.repeats, lam=args.lamda, warmup=not args.no_warmup, seed=args.seed,
        sep=args.sep, sigma_para=args.sigma_para, sigma_perp=args.sigma_perp, rho=args.rho
    )


if __name__ == "__main__":
    main()
