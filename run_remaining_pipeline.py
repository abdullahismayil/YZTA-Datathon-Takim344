"""
Otomatik chained pipeline:
1. LGBM tuning (~30-45 dk)
2. XGBoost tuning (~45-60 dk)
3. Final ensemble (CB tuned + LGB tuned + XGB tuned)

Calistirma:
    python run_remaining_pipeline.py
"""
from __future__ import annotations
import subprocess
import sys
import numpy as np
import pandas as pd

print("="*60)
print(">>> ASAMA 1: LightGBM Tuning <<<")
print("="*60)
result = subprocess.run([sys.executable, "tune_lightgbm.py"], capture_output=False)
if result.returncode != 0:
    print("LGBM tuning hata verdi, durdu.")
    sys.exit(1)

print("\n" + "="*60)
print(">>> ASAMA 2: XGBoost Tuning <<<")
print("="*60)
result = subprocess.run([sys.executable, "tune_xgboost.py"], capture_output=False)
if result.returncode != 0:
    print("XGB tuning hata verdi, durdu.")
    sys.exit(1)

print("\n" + "="*60)
print(">>> ASAMA 3: Final Ensemble (3 tuned model) <<<")
print("="*60)

from src.config import OOF_DIR, SUBMISSIONS_DIR, TARGET, ID_COL
from src.cv import rmse, save_submission, log_experiment
from src.ensemble import weighted_average, optimize_blend_weights, stack_with_ridge
from src.data import load_and_preprocess

train, _, _ = load_and_preprocess()
y = train[TARGET].values

model_names = ["lgb_tuned_v1", "xgb_tuned_v1", "cb_tuned_v1"]
oof_list = []
test_list = []

for name in model_names:
    oof_path = OOF_DIR / f"oof_{name}.npy"
    if not oof_path.exists():
        print(f"UYARI: {oof_path} bulunamadi, atlaniyor")
        continue
    oof_list.append(np.load(oof_path))
    sub_files = sorted(SUBMISSIONS_DIR.glob(f"sub_{name}_cv*.csv"))
    df = pd.read_csv(sub_files[-1])
    target_col = [c for c in df.columns if c != ID_COL][0]
    test_list.append(df[target_col].values)

print(f"\n{len(oof_list)} model yuklendi")

print("\n--- Bireysel CV ---")
for n, oof in zip(model_names, oof_list):
    print(f"  {n}: {rmse(y, oof):.5f}")

# Equal
oof_eq, test_eq = weighted_average(oof_list, test_list)
eq_rmse = rmse(y, oof_eq)
print(f"\nEqual avg: {eq_rmse:.5f}")

# Optimized
weights = optimize_blend_weights(y, oof_list)
oof_opt, test_opt = weighted_average(oof_list, test_list, weights=weights)
opt_rmse = rmse(y, oof_opt)
print(f"Optimized: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(model_names, weights)))
print(f"  CV: {opt_rmse:.5f}")

# Stacking
stacked_oof, stacked_test, ridge = stack_with_ridge(y, oof_list, test_list)
stack_rmse = rmse(y, stacked_oof)
print(f"Stacking coefs: " + ", ".join(f"{n}={c:.3f}" for n, c in zip(model_names, ridge.coef_)))
print(f"  CV: {stack_rmse:.5f}")

save_submission(test_eq, "ensemble_equal_alltuned", cv_rmse=eq_rmse)
save_submission(test_opt, "ensemble_optimized_alltuned", cv_rmse=opt_rmse)
save_submission(stacked_test, "ensemble_stacking_alltuned", cv_rmse=stack_rmse)

best = min(eq_rmse, opt_rmse, stack_rmse)
print(f"\n>>> En iyi tuned ensemble CV: {best:.5f} <<<")
print(f">>> Onceki en iyi: 1.21544 (delta={best - 1.21544:+.5f}) <<<")

print("\n" + "="*60)
print("PIPELINE TAMAMLANDI")
print("="*60)
