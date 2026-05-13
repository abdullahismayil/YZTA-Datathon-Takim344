"""
F2 — Baseline + Feature Engineering v1.

LightGBM, XGBoost, CatBoost'u ayrı ayrı eğitir.
FE v1: uyku verimliliği, etkileşim, demografik türev değişkenler.

Çıktılar:
- 3 ayrı OOF dosyası ve 3 ayrı submission
- (Bir sonraki adım: ensemble.py bunları birleştirecek)
"""
from __future__ import annotations
from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import (
    train_one_fold_lgb,
    train_one_fold_xgb,
    train_one_fold_cb,
)
from src.cv import run_cv, save_oof, save_submission, log_experiment
from src.config import ID_COL, TARGET


def main() -> None:
    seed_everything()

    # 1) Veri + FE v1
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    train, test = to_categorical(train, test)

    FEATURES = get_feature_columns(train, ID_COL, TARGET)
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]
    print(f"X shape: {X.shape} | features: {len(FEATURES)}")

    # 2) Üç model ayrı ayrı
    runs = [
        ("lgb_fe_v1", train_one_fold_lgb),
        ("xgb_fe_v1", train_one_fold_xgb),
        ("cb_fe_v1",  train_one_fold_cb),
    ]

    results = {}
    for name, train_fn in runs:
        print(f"\n=== {name} ===")
        result = run_cv(X, y, X_test,
                        train_one_fold=train_fn,
                        experiment_name=name)
        save_oof(result["oof"], name)
        save_submission(result["test_preds"], name, cv_rmse=result["cv_rmse"])
        log_experiment(result, extra_info={
            "model": name.split("_")[0],
            "n_features": len(FEATURES),
            "fe_version": "v1",
        })
        results[name] = result

    # 3) Özet
    print("\n=== ÖZET (CV RMSE) ===")
    for name, r in results.items():
        print(f"  {name}: {r['cv_rmse']:.5f} (±{r['cv_std']:.5f})")


if __name__ == "__main__":
    main()
