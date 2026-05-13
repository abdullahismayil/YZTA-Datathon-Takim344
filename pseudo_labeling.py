"""
Pseudo-labeling — duzeltilmis versiyon.

Guven kriteri: tahmin degeri 4-7 araliginda olmasi (modelin en az
hata yaptigi orta bolge). Bu satirlari sahte etiketle train'e ekle.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import (
    train_one_fold_lgb, train_one_fold_xgb, train_one_fold_cb,
)
from src.cv import run_cv, save_oof, save_submission, rmse
from src.ensemble import stack_with_ridge
from src.config import (
    ID_COL, TARGET, SEED, OOF_DIR, SUBMISSIONS_DIR, LOGS_DIR
)


# Pseudo-label icin tahmin araligi (modelin en az hata yaptigi orta bolge)
PSEUDO_MIN = 4.0
PSEUDO_MAX = 7.0


def load_best_params():
    with open(LOGS_DIR / "best_params_lightgbm.json") as f:
        lgb = json.load(f)
    with open(LOGS_DIR / "best_params_xgboost.json") as f:
        xgb = json.load(f)
    with open(LOGS_DIR / "best_params_catboost.json") as f:
        cb = json.load(f)
    return lgb, xgb, cb


def main():
    seed_everything()
    lgb_p, xgb_p, cb_p = load_best_params()

    print("Veri + FE v1 yukleniyor...")
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    train, test = to_categorical(train, test)

    FEATURES = get_feature_columns(train, ID_COL, TARGET)
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]


    # Test tahminleri: mevcut seed-avg stacking submission'i al
    sub_files = sorted(SUBMISSIONS_DIR.glob("sub_ensemble_stacking_seedavg_cv*.csv"))
    if not sub_files:
        print("HATA: seedavg stacking submission bulunamadi")
        return
    df = pd.read_csv(sub_files[-1])
    target_col = [c for c in df.columns if c != ID_COL][0]
    test_pred = df[target_col].values

    print(f"Test tahmin dagilimi:")
    print(f"  Mean: {test_pred.mean():.3f}, Std: {test_pred.std():.3f}")
    print(f"  Min: {test_pred.min():.3f}, Max: {test_pred.max():.3f}")

    # Orta bolgedeki tahminleri sec
    confident_mask = (test_pred >= PSEUDO_MIN) & (test_pred <= PSEUDO_MAX)
    n_confident = confident_mask.sum()
    print(f"\n[{PSEUDO_MIN}, {PSEUDO_MAX}] araliginda: {n_confident} satir ({100*n_confident/len(test_pred):.1f}%)")

    pseudo_X = X_test.iloc[confident_mask].copy()
    pseudo_y = test_pred[confident_mask]

    print(f"\nPseudo-label dagilimi:")
    print(f"  Mean: {pseudo_y.mean():.3f} (orig train mean: {y.mean():.3f})")
    print(f"  Std: {pseudo_y.std():.3f} (orig train std: {y.std():.3f})")


    # Custom CV: orijinal train uzerinde 5-fold,
    # pseudo'lar her fold'un train kismina ekleniyor
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    n_orig = len(X)


    def run_pseudo_cv(name, train_fn, params, fold_kwargs_extra):
        oof = np.zeros(n_orig)
        test_preds = np.zeros(len(X_test))
        fold_scores = []

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
            # Train: orig_train_part + tum pseudo
            X_tr = pd.concat([X.iloc[tr_idx], pseudo_X], axis=0, ignore_index=True)
            y_tr_arr = np.concatenate([y.values[tr_idx], pseudo_y])
            y_tr = pd.Series(y_tr_arr, index=range(len(X_tr)))

            X_va = X.iloc[va_idx]
            y_va = y.iloc[va_idx]

            val_pred, test_pred_fold, _ = train_fn(
                X_tr, y_tr, X_va, y_va, X_test, fold,
                params=params, **fold_kwargs_extra
            )
            oof[va_idx] = val_pred
            test_preds += test_pred_fold / 5
            score = rmse(y_va.values, val_pred)
            fold_scores.append(score)
            print(f"  Fold {fold}: RMSE={score:.5f}")

        cv = rmse(y.values, oof)
        print(f"  {name} CV: {cv:.5f} (mean fold: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f})")
        return oof, test_preds, cv


    # CB pseudo
    print("\n" + "="*55)
    print("CB + pseudo-labels")
    print("="*55)
    oof_cb_p, test_cb_p, cv_cb = run_pseudo_cv(
        "cb_pseudo", train_one_fold_cb,
        {**cb_p, "loss_function": "RMSE",
         "verbose": 0, "allow_writing_files": False,
         "random_seed": SEED},
        {"iterations": 5000, "early_stopping": 200},
    )
    save_oof(oof_cb_p, "cb_pseudo")
    save_submission(test_cb_p, "cb_pseudo", cv_rmse=cv_cb)
    print(f"  Reference: 1.21600, Delta: {cv_cb - 1.21600:+.5f}")


    # LGB pseudo
    print("\n" + "="*55)
    print("LGB + pseudo-labels")
    print("="*55)
    oof_lgb_p, test_lgb_p, cv_lgb = run_pseudo_cv(
        "lgb_pseudo", train_one_fold_lgb,
        {**lgb_p, "objective": "regression", "metric": "rmse",
         "verbosity": -1, "seed": SEED},
        {"num_boost_round": 5000, "early_stopping": 200},
    )
    save_oof(oof_lgb_p, "lgb_pseudo")
    save_submission(test_lgb_p, "lgb_pseudo", cv_rmse=cv_lgb)
    print(f"  Reference: 1.22006, Delta: {cv_lgb - 1.22006:+.5f}")


    # XGB pseudo
    print("\n" + "="*55)
    print("XGB + pseudo-labels")
    print("="*55)
    oof_xgb_p, test_xgb_p, cv_xgb = run_pseudo_cv(
        "xgb_pseudo", train_one_fold_xgb,
        {**xgb_p, "objective": "reg:squarederror",
         "eval_metric": "rmse", "tree_method": "hist",
         "verbosity": 0, "seed": SEED},
        {"num_boost_round": 5000, "early_stopping": 200},
    )
    save_oof(oof_xgb_p, "xgb_pseudo")
    save_submission(test_xgb_p, "xgb_pseudo", cv_rmse=cv_xgb)
    print(f"  Reference: 1.21890, Delta: {cv_xgb - 1.21890:+.5f}")


    # Stacking
    print("\n" + "="*55)
    print("PSEUDO STACKING")
    print("="*55)
    stacked_oof, stacked_test, ridge_model = stack_with_ridge(
        y.values,
        [oof_lgb_p, oof_xgb_p, oof_cb_p],
        [test_lgb_p, test_xgb_p, test_cb_p],
    )
    cv_stack = rmse(y.values, stacked_oof)
    print(f"Coefs: lgb={ridge_model.coef_[0]:.3f}, xgb={ridge_model.coef_[1]:.3f}, cb={ridge_model.coef_[2]:.3f}")
    print(f"  CV: {cv_stack:.5f}")

    save_submission(stacked_test, "ensemble_stacking_pseudo", cv_rmse=cv_stack)


    # OZET
    print("\n" + "="*55)
    print("OZET")
    print("="*55)
    print(f"  Pseudo-label sayisi: {n_confident} ({100*n_confident/len(test_pred):.1f}%)")
    print()
    print(f"  Reference     | Pseudo")
    print(f"  --------------|--------------")
    print(f"  LGB:  1.22006 | {cv_lgb:.5f}  ({cv_lgb - 1.22006:+.5f})")
    print(f"  XGB:  1.21890 | {cv_xgb:.5f}  ({cv_xgb - 1.21890:+.5f})")
    print(f"  CB:   1.21600 | {cv_cb:.5f}  ({cv_cb - 1.21600:+.5f})")
    print(f"  Stack:1.21520 | {cv_stack:.5f}  ({cv_stack - 1.21520:+.5f})")
    print(f"  vs SEED-AVG (1.21412): {cv_stack - 1.21412:+.5f}")


if __name__ == "__main__":
    main()
