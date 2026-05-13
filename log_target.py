"""
Deneme 1: Hedef d\u00f6n\u00fc\u015f\u00fcm\u00fc.

log(target+1) ile e\u011fit, exp(pred)-1 ile geri d\u00f6n\u00fc\u015ft\u00fcr.
Hedef da\u011f\u0131l\u0131m\u0131 sa\u011fa carpiksa bu y\u00f6ntem RMSE'yi azaltabilir.

3 model (LGB+XGB+CB) ile dene, ensemble yap.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import (
    train_one_fold_lgb, train_one_fold_xgb, train_one_fold_cb,
)
from src.cv import run_cv, save_oof, save_submission, rmse
from src.ensemble import stack_with_ridge
from src.config import ID_COL, TARGET, SEED, LOGS_DIR


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
    y_orig = train[TARGET].values
    X_test = test[FEATURES]

    # Log donusum
    y_log = np.log1p(y_orig)
    print(f"Original target: mean={y_orig.mean():.3f}, std={y_orig.std():.3f}, skew={pd.Series(y_orig).skew():.3f}")
    print(f"Log target:      mean={y_log.mean():.3f}, std={y_log.std():.3f}, skew={pd.Series(y_log).skew():.3f}")

    # Pandas Series wrapper'i (run_cv .iloc bekliyor)
    y_log_s = pd.Series(y_log, index=train.index)


    runs = [
        ("lgb_logy",
         train_one_fold_lgb,
         {"params": {**lgb_p, "objective": "regression", "metric": "rmse",
                     "verbosity": -1, "seed": SEED},
          "num_boost_round": 5000, "early_stopping": 200}),
        ("xgb_logy",
         train_one_fold_xgb,
         {"params": {**xgb_p, "objective": "reg:squarederror",
                     "eval_metric": "rmse", "tree_method": "hist",
                     "verbosity": 0, "seed": SEED},
          "num_boost_round": 5000, "early_stopping": 200}),
        ("cb_logy",
         train_one_fold_cb,
         {"params": {**cb_p, "loss_function": "RMSE",
                     "verbose": 0, "allow_writing_files": False,
                     "random_seed": SEED},
          "iterations": 5000, "early_stopping": 200}),
    ]

    results = {}
    for name, fn, fold_kwargs in runs:
        print(f"\n=== {name} (log target) ===")
        result = run_cv(X, y_log_s, X_test,
                        train_one_fold=fn,
                        experiment_name=name,
                        fold_kwargs=fold_kwargs)

        # OOF ve test tahminlerini orijinal skala'ya geri donustur
        oof_orig = np.expm1(result["oof"])
        test_orig = np.expm1(result["test_preds"])

        # Orijinal skala RMSE'sini hesapla
        cv_orig_scale = rmse(y_orig, oof_orig)
        print(f"  Log scale CV: {result['cv_rmse']:.5f}")
        print(f"  Original scale CV: {cv_orig_scale:.5f}")

        # Onceki single-seed CV ile kiyas
        single_seed_cv = {"lgb_logy": 1.22006, "xgb_logy": 1.21890, "cb_logy": 1.21600}
        ref = single_seed_cv[name]
        print(f"  Reference (no log): {ref:.5f}, Delta: {cv_orig_scale - ref:+.5f}")

        save_oof(oof_orig, name)
        save_submission(test_orig, name, cv_rmse=cv_orig_scale)
        results[name] = {"oof": oof_orig, "test": test_orig, "cv": cv_orig_scale}


    # Stacking (log-target modelleriyle)
    print("\n" + "="*55)
    print("LOG-TARGET STACKING (3 model)")
    print("="*55)
    oof_list = [results[n]["oof"] for n in ["lgb_logy", "xgb_logy", "cb_logy"]]
    test_list = [results[n]["test"] for n in ["lgb_logy", "xgb_logy", "cb_logy"]]

    stacked_oof, stacked_test, ridge_model = stack_with_ridge(y_orig, oof_list, test_list)
    cv_stack = rmse(y_orig, stacked_oof)
    print(f"Stacking coefs: lgb={ridge_model.coef_[0]:.3f}, xgb={ridge_model.coef_[1]:.3f}, cb={ridge_model.coef_[2]:.3f}")
    print(f"  CV: {cv_stack:.5f}")
    print(f"  Reference (seed42 stacking): 1.21520")
    print(f"  Delta: {cv_stack - 1.21520:+.5f}")

    save_submission(stacked_test, "ensemble_stacking_logy", cv_rmse=cv_stack)


    print("\n" + "="*55)
    print("OZET")
    print("="*55)
    for n, r in results.items():
        print(f"  {n}: {r['cv']:.5f}")
    print(f"  stacking: {cv_stack:.5f}")
    best = min(cv_stack, *[r["cv"] for r in results.values()])
    print(f"\n  En iyi: {best:.5f}")
    print(f"  Onceki en iyi (seed-avg stacking): 1.21412")
    print(f"  Delta: {best - 1.21412:+.5f}")


if __name__ == "__main__":
    main()
