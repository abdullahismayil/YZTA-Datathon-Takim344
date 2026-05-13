"""
Ensemble fonksiyonları.

İki yaklaşım:
1. Weighted average: OOF skorlarına göre ağırlıklandırma veya optimize.
2. Stacking: OOF tahminleri meta-learner (Ridge) için feature olur.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

from .cv import rmse


def weighted_average(
    oof_list: Sequence[np.ndarray],
    test_preds_list: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Verilen ağırlıklarla weighted average. weights=None ise eşit ağırlık."""
    n = len(oof_list)
    assert n == len(test_preds_list), "OOF ve test_preds uzunluk eşleşmiyor"
    if weights is None:
        weights = [1.0 / n] * n
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()

    oof_ens = sum(w * p for w, p in zip(weights, oof_list))
    test_ens = sum(w * p for w, p in zip(weights, test_preds_list))
    return oof_ens, test_ens


def optimize_blend_weights(
    y_true: np.ndarray,
    oof_list: Sequence[np.ndarray],
) -> np.ndarray:
    """OOF üzerinde RMSE'yi minimize eden ağırlıkları bulur.

    Kısıtlar: weights >= 0, sum(weights) = 1
    """
    n = len(oof_list)
    oof_matrix = np.column_stack(oof_list)

    def loss(w):
        w = np.array(w)
        if w.sum() == 0:
            return 1e10
        w = w / w.sum()
        pred = oof_matrix @ w
        return rmse(y_true, pred)

    # Başlangıç: eşit ağırlık
    x0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    result = minimize(loss, x0, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    weights = result.x / result.x.sum()
    return weights


def stack_with_ridge(
    y_true: np.ndarray,
    oof_list: Sequence[np.ndarray],
    test_preds_list: Sequence[np.ndarray],
    alpha: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, Ridge]:
    """OOF tahminleri Ridge meta-learner için feature.

    Geri dönüş: (stacked_oof, stacked_test_preds, fitted_ridge)
    """
    X_meta_train = np.column_stack(oof_list)
    X_meta_test = np.column_stack(test_preds_list)

    ridge = Ridge(alpha=alpha, positive=True)  # negatif ağırlık vermesin
    ridge.fit(X_meta_train, y_true)

    stacked_oof = ridge.predict(X_meta_train)
    stacked_test = ridge.predict(X_meta_test)

    return stacked_oof, stacked_test, ridge
