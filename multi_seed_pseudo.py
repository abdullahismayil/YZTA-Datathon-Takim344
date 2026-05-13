"""
Multi-seed + Pseudo-labeling birle\u015fimi.

5 farkli seed icin her modeli pseudo-label augmented train ile egit.
Tahminleri seed boyunca ortala (multi-seed varyans azaltma)
+ pseudo-label kazanci.

Beklenen kazanc: -0.00108 (multi-seed) + -0.00054 (pseudo) = ~-0.0015
Yeni CV beklentisi: ~1.213
"""
from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import (
    train_one_fold_lgb, train_one_fold_xgb, train_one_fold_cb,
)
from src.cv import save_oof, save_submission, rmse
from src.ensemble import stack_with_ridge
from src.config import (
    ID_COL, TARGET, SEED, OOF_DIR, SUBMISSIONS_DIR, LOGS_DIR
)


SEEDS = [42, 123, 456, 789, 2024]
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


def run_pseudo_cv_for_seed(X, y, X_test, pseudo_X, pseudo_y, train_fn,
                             params, fold_kwargs, seed):
    """Tek seed icin pseudo-label augmented CV."""
    n_orig = len(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)

    oof = np.zeros(n_orig)
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr = pd.concat([X.iloc[tr_idx], pseudo_X], axis=0, ignore_index=True)
        y_tr_arr = np.concatenate([y.values[tr_idx], pseudo_y])
        y_tr = pd.Series(y_tr_arr, index=range(len(X_tr)))

        X_va = X.iloc[va_idx]
        y_va = y.iloc[va_idx]

        val_pred, test_pred_fold, _ = train_fn(
            X_tr, y_tr, X_va, y_va, X_test, fold,
            params=params, **fold_kwargs
        )
        oof[va_idx] = val_pred
        test_preds += test_pred_fold / 5

    return oof, test_preds


def main():
    t0 = time.time()
    seed_everything(42)
    lgb_p, xgb_p, cb_p = load_best_params()

    # Veri
    print("Veri + FE v1 yukleniyor...")
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    train, test = to_categorical(train, test)

    FEATURES = get_feature_columns(train, ID_COL, TARGET)
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]

    # Pseudo-label hazir et (mevcut seed-avg stacking'den)
    sub_files = sorted(SUBMISSIONS_DIR.glob("sub_ensemble_stacking_seedavg_cv*.csv"))
    df = pd.read_csv(sub_files[-1])
    target_col = [c for c in df.columns if c != ID_COL][0]
    test_pred = df[target_col].values

    confident_mask = (test_pred >= PSEUDO_MIN) & (test_pred <= PSEUDO_MAX)
    pseudo_X = X_test.iloc[confident_mask].copy()
    pseudo_y = test_pred[confident_mask]
    print(f"Pseudo-label: {len(pseudo_y)} satir ({100*len(pseudo_y)/len(test_pred):.1f}%)")
    print(f"  Mean: {pseudo_y.mean():.3f}, Std: {pseudo_y.std():.3f}\n")

    # Her seed icin 3 model
    all_oofs = {"lgb": [], "xgb": [], "cb": []}
    all_tests = {"lgb": [], "xgb": [], "cb": []}

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n{'='*55}\n>>> SEED {seed} ({seed_idx+1}/{len(SEEDS)}) <<<\n{'='*55}")

        # LGB
        params_lgb = {**lgb_p, "objective": "regression", "metric": "rmse",
                      "verbosity": -1, "seed": seed}
        oof, test_pred_seed = run_pseudo_cv_for_seed(
            X, y, X_test, pseudo_X, pseudo_y, train_one_fold_lgb,
            params_lgb,
            {"num_boost_round": 5000, "early_stopping": 200},
            seed,
        )
        cv = rmse(y.values, oof)
        all_oofs["lgb"].append(oof)
        all_tests["lgb"].append(test_pred_seed)
        print(f"  LGB seed{seed}: {cv:.5f}")

        # XGB
        params_xgb = {**xgb_p, "objective": "reg:squarederror",
                      "eval_metric": "rmse", "tree_method": "hist",
                      "verbosity": 0, "seed": seed}
        oof, test_pred_seed = run_pseudo_cv_for_seed(
            X, y, X_test, pseudo_X, pseudo_y, train_one_fold_xgb,
            params_xgb,
            {"num_boost_round": 5000, "early_stopping": 200},
            seed,
        )
        cv = rmse(y.values, oof)
        all_oofs["xgb"].append(oof)
        all_tests["xgb"].append(test_pred_seed)
        print(f"  XGB seed{seed}: {cv:.5f}")

        # CB
        params_cb = {**cb_p, "loss_function": "RMSE",
                     "verbose": 0, "allow_writing_files": False,
                     "random_seed": seed}
        oof, test_pred_seed = run_pseudo_cv_for_seed(
            X, y, X_test, pseudo_X, pseudo_y, train_one_fold_cb,
            params_cb,
            {"iterations": 5000, "early_stopping": 200},
            seed,
        )
        cv = rmse(y.values, oof)
        all_oofs["cb"].append(oof)
        all_tests["cb"].append(test_pred_seed)
        print(f"  CB seed{seed}: {cv:.5f}")


    # Seed averaging
    print(f"\n\n{'='*55}\n>>> SEED AVERAGING (with pseudo) <<<\n{'='*55}")
    avg_oofs = {}
    avg_tests = {}
    for model in ["lgb", "xgb", "cb"]:
        avg_oofs[model] = np.mean(all_oofs[model], axis=0)
        avg_tests[model] = np.mean(all_tests[model], axis=0)
        cv = rmse(y.values, avg_oofs[model])
        print(f"  {model} (5-seed avg, pseudo): {cv:.5f}")

    # OOF kaydet (sonradan baska analizler icin)
    for name, oof in avg_oofs.items():
        np.save(OOF_DIR / f"oof_{name}_seedavg_pseudo.npy", oof)


    # Final stacking
    print(f"\n>>> STACKING (multi-seed + pseudo) <<<")
    oof_list = [avg_oofs["lgb"], avg_oofs["xgb"], avg_oofs["cb"]]
    test_list = [avg_tests["lgb"], avg_tests["xgb"], avg_tests["cb"]]

    stacked_oof, stacked_test, ridge_model = stack_with_ridge(y.values, oof_list, test_list)
    cv_stack = rmse(y.values, stacked_oof)
    print(f"Coefs: lgb={ridge_model.coef_[0]:.3f}, xgb={ridge_model.coef_[1]:.3f}, cb={ridge_model.coef_[2]:.3f}")
    print(f"  Intercept: {ridge_model.intercept_:.3f}")
    print(f"  CV RMSE: {cv_stack:.5f}")

    save_submission(stacked_test, "ensemble_stacking_seedavg_pseudo", cv_rmse=cv_stack)


    # OZET
    print(f"\n{'='*55}\nOZET\n{'='*55}")
    print(f"  Tek seed (no pseudo, no avg) stacking: 1.21520")
    print(f"  Multi-seed avg (no pseudo) stacking:   1.21412  (-0.00108)")
    print(f"  Single-seed pseudo stacking:           1.21466  (-0.00054)")
    print(f"  MULTI-SEED + PSEUDO stacking:          {cv_stack:.5f}  ({cv_stack-1.21520:+.5f})")
    print(f"\n  vs SEED-AVG (mevcut en iyi 1.21412): {cv_stack - 1.21412:+.5f}")
    if cv_stack < 1.21412:
        print(f"  + KAZANC! Yeni en iyi.")
    else:
        print(f"  - Kazanc yok, mevcut en iyi (1.21412) korunur.")

    elapsed = time.time() - t0
    print(f"\nToplam sure: {elapsed/60:.1f} dakika")


if __name__ == "__main__":
    main()
