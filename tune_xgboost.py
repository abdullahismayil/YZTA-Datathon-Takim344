"""F4c - XGBoost Optuna Hyperparameter Tuning."""
from __future__ import annotations
import json
import time
import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import train_one_fold_xgb
from src.cv import run_cv, save_oof, save_submission, log_experiment
from src.config import ID_COL, TARGET, SEED, LOGS_DIR


N_TRIALS = 40
TIMEOUT_SECONDS = None


def objective(trial, X, y, X_test):
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "verbosity": 0,
        "seed": SEED,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    fold_kwargs = {"params": params, "num_boost_round": 3000, "early_stopping": 100}
    result = run_cv(X, y, X_test,
                    train_one_fold=train_one_fold_xgb,
                    experiment_name=f"xgb_trial_{trial.number}",
                    fold_kwargs=fold_kwargs, verbose=False)
    rmse = result["cv_rmse"]
    print(f"  Trial {trial.number}: RMSE={rmse:.5f}  (depth={params['max_depth']}, lr={params['learning_rate']:.4f})")
    return rmse


def main():
    seed_everything()
    print("Veri + FE v1 yukleniyor...")
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    train, test = to_categorical(train, test)

    FEATURES = get_feature_columns(train, ID_COL, TARGET)
    X = train[FEATURES]; y = train[TARGET]; X_test = test[FEATURES]
    print(f"X: {X.shape}, features: {len(FEATURES)}")

    print(f"\n=== Optuna XGBoost Tuning ({N_TRIALS} trials) ===\n")
    sampler = TPESampler(seed=SEED, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name="xgboost_tuning")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.time()
    study.optimize(lambda trial: objective(trial, X, y, X_test),
                   n_trials=N_TRIALS, timeout=TIMEOUT_SECONDS, show_progress_bar=False)
    elapsed = time.time() - t0

    print(f"\n=== Tuning bitti ({elapsed/60:.1f} dakika) ===")
    print(f"En iyi RMSE: {study.best_value:.5f}")
    print(f"En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print(f"\n=== Final XGB modeli en iyi parametrelerle egitiliyor ===")
    best_params = {"objective": "reg:squarederror", "eval_metric": "rmse",
                   "tree_method": "hist", "verbosity": 0, "seed": SEED,
                   **study.best_params}
    fold_kwargs = {"params": best_params, "num_boost_round": 5000, "early_stopping": 200}
    final_result = run_cv(X, y, X_test,
                          train_one_fold=train_one_fold_xgb,
                          experiment_name="xgb_tuned_v1",
                          fold_kwargs=fold_kwargs)
    save_oof(final_result["oof"], "xgb_tuned_v1")
    save_submission(final_result["test_preds"], "xgb_tuned_v1", cv_rmse=final_result["cv_rmse"])
    log_experiment(final_result, extra_info={
        "model": "xgboost", "fe_version": "v1", "tuning": "optuna",
        "n_trials": N_TRIALS, "best_params": study.best_params,
    })

    params_path = LOGS_DIR / "best_params_xgboost.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nEn iyi parametreler kaydedildi: {params_path}")
    print(f"\n>>> XGB tuned final CV: {final_result['cv_rmse']:.5f} <<<")
    print(f">>> XGB FE v1 onceki: 1.22745 <<<")
    print(f">>> Delta: {final_result['cv_rmse'] - 1.22745:+.5f} <<<")


if __name__ == "__main__":
    main()
