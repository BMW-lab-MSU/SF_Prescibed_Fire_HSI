import numpy as np


def stratified_subset(X, y, n_per_class=1000):
    """
    Select up to `n_per_class` samples from each class to create a balanced subset.
    """
    X_sub, y_sub = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) == 0:
            continue
        n = min(n_per_class, len(cls_idx))
        selected = cls_idx[:n]  # or use np.random.choice(cls_idx, n, replace=False)
        X_sub.append(X[selected])
        y_sub.append(y[selected])
    return np.vstack(X_sub), np.hstack(y_sub)
