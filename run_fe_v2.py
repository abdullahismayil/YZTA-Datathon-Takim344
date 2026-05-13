"""
F5 — Advanced Feature Engineering v2.

v1 + ek feature'lar:
- Missing indicators (6 sütun + toplam)
- İkili etkileşimler (5 yeni)
- Polinom feature'lar (4 yeni)
- Oran feature'ları (4 yeni)
- Group statistics (ülke/meslek/kronotip × yaş/stres/çalışma/rem/vki)
- KFold-aware target encoding (kategorik sütunlar)

3 model eğitir, ensemble yapar, en iyiyi submission havuzuna ekler.

Çıktılar:
- outputs/submissions/sub_lgb_fe_v2_cv*.csv (ve xgb, cb)
- outputs/submissions/sub_ensemble_*_v2_cv*.csv
- outputs/oof/oof_*_v2.npy
"""
from __future__ import annotations
import numpy as np

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v2, add_target_encoding, get_feature_columns
from src.models import (
    train_one_fold_lgb,
    train_one_fold_xgb,
    train_one_fold_cb,
)
from src.cv import run_cv, save_oof, save_submission, log_experiment, rmse
from src.ensemble import (
    weighted_average,
    optimize_blend_weights,
    stack_with_ridge,
)
from src.config import ID_COL, TARGET, CATEGORICAL_COLS


def main() -> None:
    seed_everything()

    # 1) Veri + FE v2
    print("Veri yükleniyor + FE v2 uygulanıyor...")
    train, test, _ = load_and_preprocess()
    train, test = make_features_v2(train, test)
    print(f"FE v2 sonrası — train: {train.shape}, test: {test.shape}")

    # 2) Target encoding (KFold-aware, train hedefini kullanır)
    print("\nTarget encoding uygulanıyor (KFold-aware)...")
    te_cols = ["ulke", "meslek", "kronotip", "ruh_sagligi_durumu"]
    train, test = add_target_encoding(
        train, test, target_col=TARGET, cat_cols=te_cols,
    )
    print(f"TE sonrası — train: {train.shape}, test: {test.shape}")

    # 3) Categorical dtype
    train, test = to_categorical(train, test)

    FEATURES = get_feature_columns(train, ID_COL, TARGET)
    X = train[FEATURES]
    y = train[TARGET]
    X_test = test[FEATURES]
    print(f"\nÖzellik sayısı: {len(FEATURES)}")
    print(f"Yeni FE sütunları: {sum(1 for c in FEATURES if c.startswith('fe_'))}")

    # 4) 3 model eğit
    runs = [
        ("lgb_fe_v2", train_one_fold_lgb),
        ("xgb_fe_v2", train_one_fold_xgb),
        ("cb_fe_v2",  train_one_fold_cb),
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
            "fe_version": "v2",
        })
        results[name] = result

    # 5) Tek model özeti
    print("\n=== TEK MODEL ÖZETİ (CV RMSE) ===")
    for name, r in results.items():
        print(f"  {name}: {r['cv_rmse']:.5f} (±{r['cv_std']:.5f})")

    # 6) Ensemble
    print("\n=== ENSEMBLE ===")
    oof_list = [results[n]["oof"] for n, _ in runs]
    test_list = [results[n]["test_preds"] for n, _ in runs]
    model_names = [n for n, _ in runs]
    y_arr = y.values

    # Equal
    oof_eq, test_eq = weighted_average(oof_list, test_list)
    eq_rmse = rmse(y_arr, oof_eq)
    print(f"Equal weight average: {eq_rmse:.5f}")

    # Optimized
    weights = optimize_blend_weights(y_arr, oof_list)
    oof_opt, test_opt = weighted_average(oof_list, test_list, weights=weights)
    opt_rmse = rmse(y_arr, oof_opt)
    print(f"Optimized weights: " +
          ", ".join(f"{n}={w:.3f}" for n, w in zip(model_names, weights)))
    print(f"  CV RMSE: {opt_rmse:.5f}")

    # Stacking
    stacked_oof, stacked_test, ridge = stack_with_ridge(y_arr, oof_list, test_list)
    stack_rmse = rmse(y_arr, stacked_oof)
    print(f"Ridge stacking coefs: " +
          ", ".join(f"{n}={c:.3f}" for n, c in zip(model_names, ridge.coef_)))
    print(f"  CV RMSE: {stack_rmse:.5f}")

    # Save ensemble submissions
    save_submission(test_eq, "ensemble_equal_v2", cv_rmse=eq_rmse)
    save_submission(test_opt, "ensemble_optimized_v2", cv_rmse=opt_rmse)
    save_submission(stacked_test, "ensemble_stacking_v2", cv_rmse=stack_rmse)

    log_experiment(
        {"experiment_name": "ensemble_v2",
         "cv_rmse": min(eq_rmse, opt_rmse, stack_rmse),
         "cv_std": 0.0, "fold_scores": [], "elapsed_sec": 0.0},
        extra_info={
            "type": "ensemble",
            "fe_version": "v2",
            "individual": {n: results[n]["cv_rmse"] for n, _ in runs},
            "equal_avg_rmse": eq_rmse,
            "optimized_rmse": opt_rmse,
            "stacking_rmse": stack_rmse,
            "optimized_weights": dict(zip(model_names, weights.tolist())),
        }
    )

    # 7) En iyi ile karşılaştırma
    best_v2 = min(eq_rmse, opt_rmse, stack_rmse)
    print(f"\n>>> v2 en iyi ensemble CV: {best_v2:.5f} <<<")
    print(f">>> v1 en iyi ensemble CV referansı: ~1.21661 <<<")
    print(f">>> Δ (v2 - v1): {best_v2 - 1.21661:+.5f} <<<")


if __name__ == "__main__":
    main()
