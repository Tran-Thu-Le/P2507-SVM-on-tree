import time
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from svm_on_tree_cpp import fit_core_lambda_any

def project_spine(X: np.ndarray, y: np.ndarray):
    y = y.astype(np.int64)
    X0, X1 = X[y == 0], X[y == 1]
    if len(X0) == 0 or len(X1) == 0:
        raise ValueError("Both classes 0 and 1 are required.")
    m0 = X0.mean(axis=0)
    m1 = X1.mean(axis=0)
    w = m1 - m0
    nw = np.linalg.norm(w)
    w_unit = np.zeros_like(w) if nw == 0 else w / nw
    t = (X - m0) @ w_unit
    Xproj = m0 + np.outer(t, w_unit)
    return Xproj, m0, w_unit

@dataclass
class FitResult:
    s_node: int
    p_node: int
    s_is_proj: bool
    p_is_proj: bool
    s_orig: int
    p_orig: int
    min_val: float
    dist: float
    lamda: float
    fit_time: float

class SVMOnTreeLambda:
    def __init__(self, lamda: float = 1.0):
        self.lamda = float(lamda)
        self.res_: FitResult | None = None
        self.N_: int | None = None
        self.Xproj_: np.ndarray | None = None
        self.m0_: np.ndarray | None = None
        self.w_unit_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> FitResult:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        t0 = time.perf_counter()
        out = fit_core_lambda_any(X, y, self.lamda)
        fit_t = time.perf_counter() - t0

        self.N_ = X.shape[0]
        self.Xproj_, self.m0_, self.w_unit_ = project_spine(X, y)

        self.res_ = FitResult(
            s_node=int(out["s_node"]),
            p_node=int(out["p_node"]),
            s_is_proj=bool(out["s_is_proj"]),
            p_is_proj=bool(out["p_is_proj"]),
            s_orig=int(out["s_orig"]),
            p_orig=int(out["p_orig"]),
            min_val=float(out["min_val"]),
            dist=float(out["dist"]),
            lamda=float(out["lambda"]),
            fit_time=fit_t,
        )
        return self.res_

    def _node_point(self, X: np.ndarray, node: int) -> np.ndarray:
        N = self.N_
        if node < N: return X[node]
        return self.Xproj_[node - N]

    def predict_on_2n(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
        if self.res_ is None: raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        N = X.shape[0]

        s_pt = self._node_point(X, self.res_.s_node)
        p_pt = self._node_point(X, self.res_.p_node)
        mid = 0.5*(s_pt + p_pt)
        normal = p_pt - s_pt

        X_eval = np.vstack([X, self.Xproj_])
        y_eval = np.hstack([y, y])

        t1 = time.perf_counter()
        side = (X_eval - mid) @ normal
        y_pred = (side >= 0).astype(np.int64)
        pred_t = time.perf_counter() - t1
        acc = (y_pred == y_eval).mean()
        return y_pred, float(acc), float(pred_t)
