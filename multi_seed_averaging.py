"""
Strateji A: Multi-seed averaging.

Ayni 3 tuned modeli (LGB+XGB+CB) 5 farkli seed ile yeniden egitir.
Tahminleri ortalar, varyansi azaltir, daha stabil sonuc verir.

Sure: ~2 saat (CB en uzun)
"""
from __future__ import annotations
import json
import time
import numpy as np
from pathlib import Path

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import (
    train_one_fold_lgb, train_one_fold_xgb, train_one_fold_cb,
)
from src.cv import run_cv, save_oof, save_submission, log_experiment, rmse
from src.ensemble import stack_with_ridge
from src.config import ID_COL, TARGET, LOGS_DIR


SEEDS = [42, 123, 456, 789, 2024]


def load_best_params():
    """Onceden tune edilmis hyperparametreleri yukle."""
    with open(LOGS_DIR / "best_params_lightgbm.json") as f:
        lgb_params = json.load(f)
    with open(LOGS_DIR / "best_params_xgboost.json") as f:
        xgb_params = json.load(f)
    with open(LOGS_DIR / "best_params_catboost.json") as f:
        cb_params = json.load(f)
    return lgb_params, xgb_params, cb_params


def main():
    t0 = time.time()
    seed_everything(42)

    # 1) Veri
    print("Veri + FE v1 yukleniyor...")
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    train, test = to_categorical(train, test)

    FEATURES = get_feature_columns(train, ID_COL, TARGET)
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]
    print(f"X: {X.shape}\n")

    # 2) Tuned hyperparametreleri yukle
    lgb_params, xgb_params, cb_params = load_best_params()

    # 3) Her seed icin her modeli egit
    all_oofs = {"lgb": [], "xgb": [], "cb": []}
    all_tests = {"lgb": [], "xgb": [], "cb": []}

    for seed in SEEDS:
        print(f"\n{'='*50}\n>>> SEED {seed} <<<\n{'='*50}")

        # LGBM
        params = {**lgb_params, "objective": "regression", "metric": "rmse",
                  "verbosity": -1, "seed": seed}
        fold_kwargs = {"params": params, "num_boost_round": 5000, "early_stopping": 200}
        result = run_cv(X, y, X_test, train_one_fold=train_one_fold_lgb,
                        experiment_name=f"lgb_seed{seed}", fold_kwargs=fold_kwargs,
                        seed=seed)
        all_oofs["lgb"].append(result["oof"])
        all_tests["lgb"].append(result["test_preds"])

        # XGBoost
        params = {**xgb_params, "objective": "reg:squarederror",
                  "eval_metric": "rmse", "tree_method": "hist", "verbosity": 0,
                  "seed": seed}
        fold_kwargs = {"params": params, "num_boost_round": 5000, "early_stopping": 200}
        result = run_cv(X, y, X_test, train_one_fold=train_one_fold_xgb,
                        experiment_name=f"xgb_seed{seed}", fold_kwargs=fold_kwargs,
                        seed=seed)
        all_oofs["xgb"].append(result["oof"])
        all_tests["xgb"].append(result["test_preds"])

        # CatBoost
        params = {**cb_params, "loss_function": "RMSE",
                  "verbose": 0, "allow_writing_files": False,
                  "random_seed": seed}
        fold_kwargs = {"params": params, "iterations": 5000, "early_stopping": 200}
        result = run_cv(X, y, X_test, train_one_fold=train_one_fold_cb,
                        experiment_name=f"cb_seed{seed}", fold_kwargs=fold_kwargs,
                        seed=seed)
        all_oofs["cb"].append(result["oof"])
        all_tests["cb"].append(result["test_preds"])

    # 4) Her model icin seed-averaged tahminleri olustur
    print(f"\n\n{'='*50}\n>>> SEED AVERAGING <<<\n{'='*50}")
    avg_oofs = {}
    avg_tests = {}
    for model_name in ["lgb", "xgb", "cb"]:
        avg_oofs[model_name] = np.mean(all_oofs[model_name], axis=0)
        avg_tests[model_name] = np.mean(all_tests[model_name], axis=0)
        cv = rmse(y, avg_oofs[model_name])
        print(f"  {model_name} (5-seed avg): {cv:.5f}")

    # OOF'lari kaydet
    for name in ["lgb", "xgb", "cb"]:
        np.save(LOGS_DIR.parent / "oof" / f"oof_{name}_seedavg.npy", avg_oofs[name])

    # 5) Seed-averaged modellerle yeni stacking
    print(f"\n>>> Seed-averaged stacking <<<")
    oof_list = [avg_oofs["lgb"], avg_oofs["xgb"], avg_oofs["cb"]]
    test_list = [avg_tests["lgb"], avg_tests["xgb"], avg_tests["cb"]]

    stacked_oof, stacked_test, ridge = stack_with_ridge(y, oof_list, test_list)
    stack_rmse = rmse(y, stacked_oof)
    print(f"Stacking coefs: lgb={ridge.coef_[0]:.3f}, xgb={ridge.coef_[1]:.3f}, cb={ridge.coef_[2]:.3f}")
    print(f"  Intercept: {ridge.intercept_:.3f}")
    print(f"  CV RMSE: {stack_rmse:.5f}")

    save_submission(stacked_test, "ensemble_stacking_seedavg", cv_rmse=stack_rmse)

    print(f"\n>>> En iyi seed-avg stacking CV: {stack_rmse:.5f} <<<")
    print(f">>> Onceki en iyi (single-seed): 1.21520 <<<")
    print(f">>> Delta: {stack_rmse - 1.21520:+.5f} <<<")

    elapsed = time.time() - t0
    print(f"\nToplam sure: {elapsed/60:.1f} dakika")


if __name__ == "__main__":
    main()
