"""
F3 — Ensemble.

Önceden eğitilmiş modellerin (run_fe_v1.py çıktıları) OOF ve test
tahminlerini alır, üç farklı ensemble denemesi yapar:
1. Equal weight average
2. Optimize edilmiş weighted average (SLSQP)
3. Ridge stacking

En iyi CV'li olanı submission olarak kaydeder.

Önkoşul: Önce run_fe_v1.py çalıştırılmış olmalı.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.cv import rmse, save_submission, log_experiment
from src.ensemble import (
    weighted_average,
    optimize_blend_weights,
    stack_with_ridge,
)
from src.config import OOF_DIR, SUBMISSIONS_DIR, ID_COL, TARGET


def load_oof(name: str) -> np.ndarray:
    return np.load(OOF_DIR / f"oof_{name}.npy")


def load_test_preds_from_submission(name: str) -> np.ndarray:
    """Submission CSV'den test tahminlerini yeniden okur.
    (run_fe_v1.py her modelin submission'ını kaydetmişti.)"""
    files = list(SUBMISSIONS_DIR.glob(f"sub_{name}_cv*.csv"))
    if not files:
        raise FileNotFoundError(f"sub_{name}_cv*.csv bulunamadı")
    # En son üretilmiş olanı al (genelde tek tane olur)
    files.sort()
    df = __import__("pandas").read_csv(files[-1])
    target_col = [c for c in df.columns if c != ID_COL][0]
    return df[target_col].values


def main() -> None:
    seed_everything()

    # 1) Veri (sadece y'yi almak için yeniden yüklüyoruz)
    train, _, _ = load_and_preprocess()
    y = train[TARGET].values

    # 2) Modellerin OOF ve test tahminlerini yükle
    model_names = ["lgb_fe_v1", "xgb_fe_v1", "cb_fe_v1"]
    oof_list = [load_oof(n) for n in model_names]
    test_list = [load_test_preds_from_submission(n) for n in model_names]

    # Bireysel skorlar
    print("=== Bireysel modeller (OOF RMSE) ===")
    individual_scores = {}
    for n, oof in zip(model_names, oof_list):
        s = rmse(y, oof)
        individual_scores[n] = s
        print(f"  {n}: {s:.5f}")
    best_single = min(individual_scores, key=individual_scores.get)
    print(f"En iyi tek model: {best_single} ({individual_scores[best_single]:.5f})")

    # 3) Equal weight average
    print("\n=== Equal weight average ===")
    oof_eq, test_eq = weighted_average(oof_list, test_list)
    eq_rmse = rmse(y, oof_eq)
    print(f"  CV RMSE: {eq_rmse:.5f}")

    # 4) Optimize edilmiş ağırlıklar
    print("\n=== Optimize edilmiş ağırlıklar ===")
    weights = optimize_blend_weights(y, oof_list)
    oof_opt, test_opt = weighted_average(oof_list, test_list, weights=weights)
    opt_rmse = rmse(y, oof_opt)
    print(f"  Ağırlıklar: " +
          ", ".join(f"{n}={w:.3f}" for n, w in zip(model_names, weights)))
    print(f"  CV RMSE: {opt_rmse:.5f}")

    # 5) Ridge stacking
    print("\n=== Ridge stacking ===")
    stacked_oof, stacked_test, ridge = stack_with_ridge(y, oof_list, test_list)
    stack_rmse = rmse(y, stacked_oof)
    print(f"  Coefs: " +
          ", ".join(f"{n}={c:.3f}" for n, c in zip(model_names, ridge.coef_)))
    print(f"  Intercept: {ridge.intercept_:.3f}")
    print(f"  CV RMSE: {stack_rmse:.5f}")

    # 6) Hepsini kaydet
    save_submission(test_eq, "ensemble_equal_v1", cv_rmse=eq_rmse)
    save_submission(test_opt, "ensemble_optimized_v1", cv_rmse=opt_rmse)
    save_submission(stacked_test, "ensemble_stacking_v1", cv_rmse=stack_rmse)

    log_experiment(
        {"experiment_name": "ensemble_v1",
         "cv_rmse": min(eq_rmse, opt_rmse, stack_rmse),
         "cv_std": 0.0, "fold_scores": [],
         "elapsed_sec": 0.0},
        extra_info={
            "type": "ensemble",
            "individual": individual_scores,
            "equal_avg_rmse": eq_rmse,
            "optimized_rmse": opt_rmse,
            "stacking_rmse": stack_rmse,
            "optimized_weights": dict(zip(model_names, weights.tolist())),
        }
    )

    # 7) En iyi adayı işaret et
    best = min([("equal", eq_rmse), ("optimized", opt_rmse),
                ("stacking", stack_rmse)], key=lambda x: x[1])
    print(f"\n>>> En iyi ensemble: {best[0]} (RMSE: {best[1]:.5f}) <<<")


if __name__ == "__main__":
    main()
