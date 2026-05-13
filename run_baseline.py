"""
F1 — Baseline LightGBM submission.

Hiçbir feature engineering yapılmadan, sadece veri kalitesi
düzeltmeleri (ülke normalize) ile baseline kurar.

Çıktılar:
- outputs/submissions/sub_lgb_baseline_cv*.csv
- outputs/oof/oof_lgb_baseline.npy
- outputs/logs/experiments.jsonl (yeni satır)

Çalıştırma:
    cd yzta_datathon
    python run_baseline.py
"""
from __future__ import annotations
from src.data import load_and_preprocess, to_categorical, seed_everything
from src.models import train_one_fold_lgb
from src.cv import run_cv, save_oof, save_submission, log_experiment
from src.config import ID_COL, TARGET


def main() -> None:
    seed_everything()

    # 1) Veri
    train, test, _ = load_and_preprocess()
    train, test = to_categorical(train, test)

    FEATURES = [c for c in train.columns if c not in (ID_COL, TARGET)]
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]

    print(f"X shape: {X.shape} | features: {len(FEATURES)}")

    # 2) CV
    experiment_name = "lgb_baseline"
    print(f"\n=== {experiment_name} ===")
    result = run_cv(
        X, y, X_test,
        train_one_fold=train_one_fold_lgb,
        experiment_name=experiment_name,
    )

    # 3) Kayıtlar
    oof_path = save_oof(result["oof"], experiment_name)
    sub_path = save_submission(
        result["test_preds"], experiment_name,
        cv_rmse=result["cv_rmse"],
    )
    log_path = log_experiment(result, extra_info={
        "model": "lightgbm",
        "n_features": len(FEATURES),
        "fe_version": "v0",
        "notes": "Sadece ülke normalize. FE yok.",
    })

    print(f"\nKayıtlar:")
    print(f"  OOF        → {oof_path}")
    print(f"  Submission → {sub_path}")
    print(f"  Log        → {log_path}")


if __name__ == "__main__":
    main()
