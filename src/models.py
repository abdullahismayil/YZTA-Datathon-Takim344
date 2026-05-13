"""
Model train fonksiyonları — LightGBM, XGBoost, CatBoost.

Her model `cv.run_cv` ile uyumlu olacak şekilde
`train_one_fold(X_tr, y_tr, X_va, y_va, X_test, fold_idx, **kwargs)`
imzasıyla yazılır.

Geri dönüş: (val_pred, test_pred, extras_dict)
"""
from __future__ import annotations
from typing import Tuple, Any
import numpy as np
import pandas as pd
import lightgbm as lgb

from .config import SEED, CATEGORICAL_COLS


# ============================================================
# LIGHTGBM
# ============================================================

DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "verbosity": -1,
    "seed": SEED,
}


def train_one_fold_lgb(
    X_tr, y_tr, X_va, y_va, X_test, fold_idx,
    *,
    params: dict | None = None,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    cat_features: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """LightGBM ile tek fold eğitimi."""
    params = {**DEFAULT_LGB_PARAMS, **(params or {})}
    if cat_features is None:
        # CATEGORICAL_COLS'tan yalnızca X'te bulunanları seç
        cat_features = [c for c in CATEGORICAL_COLS if c in X_tr.columns]

    train_set = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_features)
    valid_set = lgb.Dataset(X_va, y_va, categorical_feature=cat_features,
                             reference=train_set)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=[valid_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_pred = model.predict(X_va, num_iteration=model.best_iteration)
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)

    fi = pd.DataFrame({
        "feature": X_tr.columns.tolist(),
        "importance": model.feature_importance(importance_type="gain"),
        "fold": fold_idx,
    })

    return val_pred, test_pred, {
        "best_iter": model.best_iteration,
        "fi": fi,
    }


# ============================================================
# XGBOOST  (lazy import — kurulumu yoksa hata vermesin)
# ============================================================

DEFAULT_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "seed": SEED,
    "tree_method": "hist",
    "verbosity": 0,
}


def train_one_fold_xgb(
    X_tr, y_tr, X_va, y_va, X_test, fold_idx,
    *,
    params: dict | None = None,
    num_boost_round: int = 5000,
    early_stopping: int = 200,
    cat_features: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """XGBoost ile tek fold eğitimi.

    XGBoost 2.0+'da kategorik veri desteği var ama enable_categorical=True
    gerekiyor. Pandas 'category' tipindeki sütunları otomatik tanır.
    """
    import xgboost as xgb

    params = {**DEFAULT_XGB_PARAMS, **(params or {})}

    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dval = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dval, "valid")],
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )

    val_pred = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
    test_pred = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))

    fi_dict = model.get_score(importance_type="gain")
    fi = pd.DataFrame({
        "feature": list(fi_dict.keys()),
        "importance": list(fi_dict.values()),
        "fold": fold_idx,
    })

    return val_pred, test_pred, {
        "best_iter": model.best_iteration,
        "fi": fi,
    }


# ============================================================
# CATBOOST
# ============================================================

DEFAULT_CB_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "random_seed": SEED,
    "verbose": 0,
    "allow_writing_files": False,
}


def train_one_fold_cb(
    X_tr, y_tr, X_va, y_va, X_test, fold_idx,
    *,
    params: dict | None = None,
    iterations: int = 5000,
    early_stopping: int = 200,
    cat_features: list[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """CatBoost ile tek fold eğitimi.

    CatBoost native kategorik desteği güçlü — string olarak verebiliriz
    ama 'category' tipini de algılıyor.
    """
    from catboost import CatBoostRegressor, Pool

    if cat_features is None:
        cat_features = [c for c in CATEGORICAL_COLS if c in X_tr.columns]

    # CatBoost NaN'ı kategorik için sevmez → string'e çevir
    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()
    X_test2 = X_test.copy()
    for col in cat_features:
        X_tr2[col] = X_tr2[col].astype(str).fillna("missing")
        X_va2[col] = X_va2[col].astype(str).fillna("missing")
        X_test2[col] = X_test2[col].astype(str).fillna("missing")

    train_pool = Pool(X_tr2, y_tr, cat_features=cat_features)
    val_pool = Pool(X_va2, y_va, cat_features=cat_features)
    test_pool = Pool(X_test2, cat_features=cat_features)

    params = {**DEFAULT_CB_PARAMS, **(params or {})}
    model = CatBoostRegressor(iterations=iterations,
                              early_stopping_rounds=early_stopping,
                              **params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict(val_pool)
    test_pred = model.predict(test_pool)

    fi = pd.DataFrame({
        "feature": X_tr2.columns.tolist(),
        "importance": model.get_feature_importance(),
        "fold": fold_idx,
    })

    return val_pred, test_pred, {
        "best_iter": model.get_best_iteration(),
        "fi": fi,
    }

# ============================================================
# YENİ MODELLER (sklearn ailesi) — çeşitlilik için
# ============================================================

def train_one_fold_hgb(
    X_tr, y_tr, X_va, y_va, X_test, fold_idx,
    *, params: dict | None = None, **kwargs,
):
    """HistGradientBoosting (sklearn) - native NaN ve kategorik destegi."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    default_params = {
        "loss": "squared_error",
        "learning_rate": 0.05,
        "max_iter": 1000,
        "max_depth": 7,
        "min_samples_leaf": 30,
        "l2_regularization": 0.1,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": 50,
        "random_state": SEED,
    }
    params = {**default_params, **(params or {})}

    cat_features = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
    cat_indices = [X_tr.columns.get_loc(c) for c in cat_features]

    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()
    X_test2 = X_test.copy()
    for col in cat_features:
        X_tr2[col] = X_tr2[col].cat.codes.replace(-1, np.nan).astype(float) if hasattr(X_tr2[col], "cat") else X_tr2[col]
        X_va2[col] = X_va2[col].cat.codes.replace(-1, np.nan).astype(float) if hasattr(X_va2[col], "cat") else X_va2[col]
        X_test2[col] = X_test2[col].cat.codes.replace(-1, np.nan).astype(float) if hasattr(X_test2[col], "cat") else X_test2[col]

    model = HistGradientBoostingRegressor(
        categorical_features=cat_indices, **params
    )
    model.fit(X_tr2, y_tr)

    val_pred = model.predict(X_va2)
    test_pred = model.predict(X_test2)
    return val_pred, test_pred, {"best_iter": model.n_iter_, "fi": None}


def train_one_fold_et(
    X_tr, y_tr, X_va, y_va, X_test, fold_idx,
    *, params: dict | None = None, **kwargs,
):
    """ExtraTreesRegressor - rastgele orman ailesi, farkli paradigma."""
    from sklearn.ensemble import ExtraTreesRegressor
    import numpy as np
    import pandas as pd

    default_params = {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_leaf": 5,
        "max_features": 0.7,
        "n_jobs": -1,
        "random_state": SEED,
    }
    params = {**default_params, **(params or {})}

    # ExtraTrees NaN sevmez ve kategorikleri sayisal sanir; encode et
    cat_features = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()
    X_test2 = X_test.copy()
    for col in cat_features:
        if hasattr(X_tr2[col], "cat"):
            X_tr2[col] = X_tr2[col].cat.codes.astype(float).replace(-1, np.nan)
            X_va2[col] = X_va2[col].cat.codes.astype(float).replace(-1, np.nan)
            X_test2[col] = X_test2[col].cat.codes.astype(float).replace(-1, np.nan)

    # NaN'lari medyan ile doldur (ExtraTrees NaN kabul etmez)
    for col in X_tr2.columns:
        if X_tr2[col].isna().any() or X_va2[col].isna().any() or X_test2[col].isna().any():
            med = X_tr2[col].median()
            X_tr2[col] = X_tr2[col].fillna(med)
            X_va2[col] = X_va2[col].fillna(med)
            X_test2[col] = X_test2[col].fillna(med)

    model = ExtraTreesRegressor(**params)
    model.fit(X_tr2, y_tr)

    val_pred = model.predict(X_va2)
    test_pred = model.predict(X_test2)
    return val_pred, test_pred, {"best_iter": None, "fi": None}


def train_one_fold_ridge(
    X_tr, y_tr, X_va, y_va, X_test, fold_idx,
    *, params: dict | None = None, **kwargs,
):
    """Ridge regression - lineer model, tamamen farkli sinyal."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    default_params = {"alpha": 1.0, "random_state": SEED}
    params = {**default_params, **(params or {})}

    cat_features = [c for c in CATEGORICAL_COLS if c in X_tr.columns]
    X_tr2 = X_tr.copy()
    X_va2 = X_va.copy()
    X_test2 = X_test.copy()
    for col in cat_features:
        if hasattr(X_tr2[col], "cat"):
            X_tr2[col] = X_tr2[col].cat.codes.astype(float).replace(-1, np.nan)
            X_va2[col] = X_va2[col].cat.codes.astype(float).replace(-1, np.nan)
            X_test2[col] = X_test2[col].cat.codes.astype(float).replace(-1, np.nan)

    # NaN doldur
    for col in X_tr2.columns:
        if X_tr2[col].isna().any() or X_va2[col].isna().any() or X_test2[col].isna().any():
            med = X_tr2[col].median()
            X_tr2[col] = X_tr2[col].fillna(med)
            X_va2[col] = X_va2[col].fillna(med)
            X_test2[col] = X_test2[col].fillna(med)

    # Standardize (Ridge icin sart)
    scaler = StandardScaler()
    X_tr3 = scaler.fit_transform(X_tr2)
    X_va3 = scaler.transform(X_va2)
    X_test3 = scaler.transform(X_test2)

    model = Ridge(**params)
    model.fit(X_tr3, y_tr)

    val_pred = model.predict(X_va3)
    test_pred = model.predict(X_test3)
    return val_pred, test_pred, {"best_iter": None, "fi": None}
