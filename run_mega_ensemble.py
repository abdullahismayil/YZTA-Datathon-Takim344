"""
MEGA ENSEMBLE: Tum mevcut OOF dosyalarini kullanarak optimal stacking.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import OOF_DIR, SUBMISSIONS_DIR, TARGET, ID_COL
from src.cv import rmse, save_submission
from src.ensemble import optimize_blend_weights, stack_with_ridge
from src.data import load_and_preprocess

train, _, _ = load_and_preprocess()
y = train[TARGET].values

# Mevcut tum OOF dosyalarini tara
oof_files = sorted(OOF_DIR.glob("oof_*.npy"))
print(f"Bulunan OOF dosyalari ({len(oof_files)}):")

oof_list = []
test_list = []
names = []

for f in oof_files:
    name = f.stem.replace("oof_", "")
    oof = np.load(f)
    
    # Karsilik gelen submission dosyasini bul
    sub_files = sorted(SUBMISSIONS_DIR.glob(f"sub_{name}_cv*.csv"))
    if not sub_files:
        print(f"  {name}: OOF var ama submission yok, atlaniyor")
        continue
    
    df = pd.read_csv(sub_files[-1])
    target_col_sub = [c for c in df.columns if c != ID_COL][0]
    test_pred = df[target_col_sub].values
    
    cv = rmse(y, oof)
    oof_list.append(oof)
    test_list.append(test_pred)
    names.append(name)
    print(f"  {name}: CV={cv:.5f}")

print(f"\nToplam {len(names)} model yuklendi")

# Optimized weights
print("\n--- Optimized Weights ---")
weights = optimize_blend_weights(y, oof_list)
oof_opt = sum(w * o for w, o in zip(weights, oof_list))
test_opt = sum(w * t for w, t in zip(weights, test_list))
opt_rmse = rmse(y, oof_opt)

for n, w in zip(names, weights):
    if w > 0.01:
        print(f"  {n}: {w:.3f}")
print(f"  CV: {opt_rmse:.5f}")

save_submission(np.clip(test_opt, 0, 10), "mega_ensemble_optimized", cv_rmse=opt_rmse)

# Ridge stacking
print("\n--- Ridge Stacking ---")
stacked_oof, stacked_test, ridge = stack_with_ridge(y, oof_list, test_list)
stack_rmse = rmse(y, stacked_oof)

for n, c in zip(names, ridge.coef_):
    if abs(c) > 0.01:
        print(f"  {n}: {c:.3f}")
print(f"  Intercept: {ridge.intercept_:.3f}")
print(f"  CV: {stack_rmse:.5f}")

save_submission(np.clip(stacked_test, 0, 10), "mega_ensemble_stacking", cv_rmse=stack_rmse)

# En iyi
best = min(opt_rmse, stack_rmse)
print(f"\n>>> MEGA ENSEMBLE en iyi: {best:.5f} <<<")
print(f">>> Onceki en iyi (seedavg_pseudo): 1.21352 <<<")
print(f">>> Delta: {best - 1.21352:+.5f} <<<")
