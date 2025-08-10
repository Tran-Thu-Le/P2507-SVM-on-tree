import time
import numpy as np
from .svm_on_tree_cpp import fit_core

class SVMOnTreeCPP:
    def __init__(self, lamda: float = 1.0):
        self.lamda = float(lamda)
        self.support_s = None
        self.support_p = None
        self.min_val = None
        self.training_time = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        t0 = time.time()
        out = fit_core(np.asarray(X, dtype=np.float64),
                       np.asarray(y, dtype=np.int64),
                       self.lamda)
        self.support_s = int(out["support_s"])
        self.support_p = int(out["support_p"])
        self.min_val = float(out["min_val"])
        self.training_time = time.time() - t0
        return self
