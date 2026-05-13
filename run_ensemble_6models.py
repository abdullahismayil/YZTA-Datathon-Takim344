"""
6-model final ensemble: lgb_tuned + xgb_tuned + cb_tuned + hgb + et + ridge.
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

model_names = [
    "lgb_tuned_v1", "xgb_tuned_v1", "cb_tuned_v1",
    "hgb_fe_v1", "et_fe_v1", "ridge_fe_v1",
]

oof_list = []
test_list = []
loaded_names = []

for name in model_names:
    oof_path = OOF_DIR / f"oof_{name}.npy"
    if not oof_path.exists():
        print(f"UYARI: {oof_path} yok, atlanıyor")
        continue
    oof_list.append(np.load(oof_path))

    sub_files = sorted(SUBMISSIONS_DIR.glob(f"sub_{name}_cv*.csv"))
    df = pd.read_csv(sub_files[-1])
    target_col = [c for c in df.columns if c != ID_COL][0]
    test_list.append(df[target_col].values)
    loaded_names.append(name)

print(f"\n{len(loaded_names)} model yuklendi: {loaded_names}\n")

print("--- Bireysel CV ---")
for n, oof in zip(loaded_names, oof_list):
    print(f"  {n}: {rmse(y, oof):.5f}")

oof_eq, test_eq = weighted_average(oof_list, test_list)
eq_rmse = rmse(y, oof_eq)
print(f"\nEqual avg: {eq_rmse:.5f}")

weights = optimize_blend_weights(y, oof_list)
oof_opt, test_opt = weighted_average(oof_list, test_list, weights=weights)
opt_rmse = rmse(y, oof_opt)
print(f"Optimized weights:")
for n, w in zip(loaded_names, weights):
    print(f"  {n}: {w:.3f}")
print(f"  CV: {opt_rmse:.5f}")

stacked_oof, stacked_test, ridge = stack_with_ridge(y, oof_list, test_list)
stack_rmse = rmse(y, stacked_oof)
print(f"\nStacking coefs:")
for n, c in zip(loaded_names, ridge.coef_):
    print(f"  {n}: {c:.3f}")
print(f"Intercept: {ridge.intercept_:.3f}")
print(f"  CV: {stack_rmse:.5f}")

save_submission(test_eq, "ensemble_equal_6m", cv_rmse=eq_rmse)
save_submission(test_opt, "ensemble_optimized_6m", cv_rmse=opt_rmse)
save_submission(stacked_test, "ensemble_stacking_6m", cv_rmse=stack_rmse)

best = min(eq_rmse, opt_rmse, stack_rmse)
print(f"\n>>> En iyi 6-model ensemble: {best:.5f} <<<")
print(f">>> Onceki en iyi: 1.21520 (delta: {best - 1.21520:+.5f}) <<<")
