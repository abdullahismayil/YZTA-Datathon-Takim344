"""
F4a — CatBoost Optuna Hyperparameter Tuning.

Strateji:
- 5-fold CV ile her trial'da RMSE hesapla
- Optuna'nın TPE sampler'ı kullan (Bayes-benzeri, akıllı arama)
- En iyi parametreleri kaydet, final modeli OOF + test tahminleri ile eğit
- Submission havuzuna ekle, log'a yaz

Çalıştırma:
    python tune_catboost.py
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import optuna
from optuna.samplers import TPESampler

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import train_one_fold_cb, DEFAULT_CB_PARAMS
from src.cv import run_cv, save_oof, save_submission, log_experiment
from src.config import ID_COL, TARGET, SEED, LOGS_DIR


# Trial sayısı — daha fazla = daha iyi sonuç ama daha uzun
N_TRIALS = 40
TIMEOUT_SECONDS = None  # None = sınırsız; örn. 7200 = 2 saat


def objective(trial, X, y, X_test):
    """Optuna her trial'da bu fonksiyonu çağırır.
    Hyperparametre seç → 5-fold CV koş → RMSE döndür."""
    params = {
        "loss_function": "RMSE",
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
        # Aranacak hyperparametre uzayı
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
        "border_count": trial.suggest_int("border_count", 32, 255),
    }

    # Sadece hızlı CV — early stopping ile
    fold_kwargs = {"params": params, "iterations": 3000, "early_stopping": 100}

    result = run_cv(
        X, y, X_test,
        train_one_fold=train_one_fold_cb,
        experiment_name=f"cb_trial_{trial.number}",
        fold_kwargs=fold_kwargs,
        verbose=False,  # Trial içinde sessiz
    )

    rmse = result["cv_rmse"]
    print(f"  Trial {trial.number}: RMSE={rmse:.5f}  (depth={params['depth']}, lr={params['learning_rate']:.4f})")
    return rmse


def main():
    seed_everything()

    # 1) Veri + FE v1 (en iyi performans veren feature seti)
    print("Veri + FE v1 yükleniyor...")
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    train, test = to_categorical(train, test)

    FEATURES = get_feature_columns(train, ID_COL, TARGET)
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]
    print(f"X: {X.shape}, features: {len(FEATURES)}")

    # 2) Optuna study
    print(f"\n=== Optuna CatBoost Tuning ({N_TRIALS} trials) ===")
    print(f"Tahmini süre: {N_TRIALS * 3} - {N_TRIALS * 5} dakika\n")

    sampler = TPESampler(seed=SEED, n_startup_trials=10)
    study = optuna.create_study(direction="minimize", sampler=sampler,
                                study_name="catboost_tuning")
    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Sessiz

    t0 = time.time()
    study.optimize(
        lambda trial: objective(trial, X, y, X_test),
        n_trials=N_TRIALS,
        timeout=TIMEOUT_SECONDS,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    print(f"\n=== Tuning bitti ({elapsed/60:.1f} dakika) ===")
    print(f"En iyi RMSE: {study.best_value:.5f}")
    print(f"En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # 3) En iyi parametrelerle final eğitim (full CV + test tahminleri)
    print(f"\n=== Final CB modeli en iyi parametrelerle eğitiliyor ===")
    best_params = {
        "loss_function": "RMSE",
        "random_seed": SEED,
        "verbose": 0,
        "allow_writing_files": False,
        **study.best_params,
    }

    fold_kwargs = {"params": best_params, "iterations": 5000, "early_stopping": 200}
    final_result = run_cv(
        X, y, X_test,
        train_one_fold=train_one_fold_cb,
        experiment_name="cb_tuned_v1",
        fold_kwargs=fold_kwargs,
    )

    # 4) Kayıtlar
    save_oof(final_result["oof"], "cb_tuned_v1")
    save_submission(final_result["test_preds"], "cb_tuned_v1",
                    cv_rmse=final_result["cv_rmse"])
    log_experiment(final_result, extra_info={
        "model": "catboost",
        "fe_version": "v1",
        "tuning": "optuna",
        "n_trials": N_TRIALS,
        "best_params": study.best_params,
    })

    # 5) En iyi parametreleri ayrıca kaydet (sonra reuse için)
    params_path = LOGS_DIR / "best_params_catboost.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nEn iyi parametreler kaydedildi: {params_path}")

    print(f"\n>>> CB tuned final CV: {final_result['cv_rmse']:.5f} <<<")
    print(f">>> CB FE v1 önceki: 1.21766 <<<")
    print(f">>> Δ: {final_result['cv_rmse'] - 1.21766:+.5f} <<<")


if __name__ == "__main__":
    main()
