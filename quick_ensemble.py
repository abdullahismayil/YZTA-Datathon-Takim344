"""
Quick mini-ensemble: CB tuned + LGB v1 + XGB v1.
Mevcut OOF dosyalarini birlestir, hizli submission cikar.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.config import OOF_DIR, SUBMISSIONS_DIR, TARGET, ID_COL
from src.cv import rmse, save_submission
from src.ensemble import weighted_average, optimize_blend_weights, stack_with_ridge
from src.data import load_and_preprocess

train, _, _ = load_and_preprocess()
y = train[TARGET].values

model_names = ["lgb_fe_v1", "xgb_fe_v1", "cb_tuned_v1"]
oof_list, test_list = [], []

for name in model_names:
    oof_path = OOF_DIR / f"oof_{name}.npy"
    oof_list.append(np.load(oof_path))
    sub_files = sorted(SUBMISSIONS_DIR.glob(f"sub_{name}_cv*.csv"))
    df = pd.read_csv(sub_files[-1])
    target_col = [c for c in df.columns if c != ID_COL][0]
    test_list.append(df[target_col].values)

print("--- Bireysel CV ---")
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

# Save all 3
save_submission(test_eq, "ensemble_equal_cbtuned", cv_rmse=eq_rmse)
save_submission(test_opt, "ensemble_optimized_cbtuned", cv_rmse=opt_rmse)
save_submission(stacked_test, "ensemble_stacking_cbtuned", cv_rmse=stack_rmse)

best = min(eq_rmse, opt_rmse, stack_rmse)
print(f"\n>>> En iyi: {best:.5f} (onceki en iyi 1.21661, delta={best-1.21661:+.5f}) <<<")
