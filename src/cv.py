"""
Cross-validation çerçevesi.

Tüm modeller bu modülün `run_cv` fonksiyonu üzerinden çalışır.
Bu sayede:
- Aynı fold'lar her seferinde kullanılır → CV skorları kıyaslanabilir
- OOF tahminleri otomatik kaydedilir → stacking için hazır
- Submission dosyası standart formatta üretilir
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from .config import (
    SEED, N_FOLDS, ID_COL, TARGET,
    OOF_DIR, SUBMISSIONS_DIR, LOGS_DIR,
    SAMPLE_SUB_PATH,
)


def rmse(y_true, y_pred) -> float:
    """sklearn versiyon-bağımsız RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_kfold(n_splits: int = N_FOLDS, seed: int = SEED) -> KFold:
    """Sabit fold'lar — bütün modellerde aynı bölme kullanılır."""
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    train_one_fold: Callable[..., Tuple[np.ndarray, np.ndarray, Any]],
    *,
    experiment_name: str,
    n_splits: int = N_FOLDS,
    seed: int = SEED,
    verbose: bool = True,
    fold_kwargs: dict | None = None,
) -> dict:
    """K-fold CV koşturucu.

    `train_one_fold(X_tr, y_tr, X_va, y_va, X_test, fold_idx, **fold_kwargs)`
    şu üçlüyü dönmeli: (val_pred, test_pred, extras_dict)

    extras_dict en azından {"best_iter": int|None, "fi": pd.DataFrame|None}
    içerebilir (zorunlu değil).

    Geri dönüş:
        {
          "oof": np.ndarray,
          "test_preds": np.ndarray,
          "fold_scores": [...],
          "cv_rmse": float,
          "cv_std": float,
          "extras": [...],
        }
    """
    fold_kwargs = fold_kwargs or {}
    kf = get_kfold(n_splits=n_splits, seed=seed)

    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    fold_scores: list[float] = []
    extras_per_fold: list[dict] = []

    t0 = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        val_pred, test_pred, extras = train_one_fold(
            X_tr, y_tr, X_va, y_va, X_test, fold_idx, **fold_kwargs
        )

        oof[va_idx] = val_pred
        test_preds += test_pred / n_splits

        score = rmse(y_va, val_pred)
        fold_scores.append(score)
        extras_per_fold.append(extras or {})

        if verbose:
            extra_msg = ""
            if extras and "best_iter" in extras and extras["best_iter"] is not None:
                extra_msg = f" (best_iter={extras['best_iter']})"
            print(f"  Fold {fold_idx}: RMSE={score:.5f}{extra_msg}")

    elapsed = time.time() - t0
    cv_rmse = rmse(y, oof)
    cv_std = float(np.std(fold_scores))

    if verbose:
        print(f"\n{experiment_name} | OOF RMSE: {cv_rmse:.5f} "
              f"(mean fold: {np.mean(fold_scores):.5f} ± {cv_std:.5f}) "
              f"| {elapsed:.1f}s")

    return {
        "experiment_name": experiment_name,
        "oof": oof,
        "test_preds": test_preds,
        "fold_scores": fold_scores,
        "cv_rmse": cv_rmse,
        "cv_std": cv_std,
        "extras": extras_per_fold,
        "elapsed_sec": elapsed,
    }


def save_oof(oof: np.ndarray, experiment_name: str) -> Path:
    """OOF tahminlerini kaydeder (stacking için)."""
    path = OOF_DIR / f"oof_{experiment_name}.npy"
    np.save(path, oof)
    return path


def save_submission(
    test_preds: np.ndarray,
    experiment_name: str,
    cv_rmse: float | None = None,
    clip_range: tuple[float, float] | None = (0.0, 10.0),
) -> Path:
    """Submission CSV'sini sample_submission formatında üretir.

    `clip_range` verilirse tahminler bu aralığa kırpılır
    (hedefin gerçek aralığı 0–10).
    """
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    target_col = [c for c in sample_sub.columns if c != ID_COL][0]

    preds = test_preds.copy()
    if clip_range is not None:
        preds = np.clip(preds, clip_range[0], clip_range[1])

    # Test setiyle aynı id sırasını kullan: sub dosyası test_x'e göre düzenlenir
    test = pd.read_csv(SAMPLE_SUB_PATH.parent / "test_x.csv", usecols=[ID_COL])
    submission = pd.DataFrame({ID_COL: test[ID_COL].values, target_col: preds})

    fname = f"sub_{experiment_name}.csv"
    if cv_rmse is not None:
        fname = f"sub_{experiment_name}_cv{cv_rmse:.5f}.csv"
    path = SUBMISSIONS_DIR / fname
    submission.to_csv(path, index=False)
    return path


def log_experiment(result: dict, extra_info: dict | None = None) -> Path:
    """CV sonuçlarını JSON log dosyasına ekler."""
    log_path = LOGS_DIR / "experiments.jsonl"
    entry = {
        "experiment_name": result["experiment_name"],
        "cv_rmse": result["cv_rmse"],
        "cv_std": result["cv_std"],
        "fold_scores": result["fold_scores"],
        "elapsed_sec": result["elapsed_sec"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_info:
        entry.update(extra_info)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return log_path
