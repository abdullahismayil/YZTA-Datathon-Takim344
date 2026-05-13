"""
Isotonic'in leakage olup olmadigini kontrol et.
KFold-aware uygula: her fold icin fit ayri, validation predict ayri.
"""
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

from src.config import OOF_DIR, SUBMISSIONS_DIR, TARGET, ID_COL, SEED
from src.cv import rmse, save_submission
from src.data import load_and_preprocess
from src.ensemble import stack_with_ridge

print("Veri ve OOF yukleniyor...")
train, _, _ = load_and_preprocess()
y_true = train[TARGET].values

oof_lgb = np.load(OOF_DIR / "oof_lgb_seedavg.npy")
oof_xgb = np.load(OOF_DIR / "oof_xgb_seedavg.npy")
oof_cb = np.load(OOF_DIR / "oof_cb_seedavg.npy")

# Test tahmini
sub_files = sorted(SUBMISSIONS_DIR.glob("sub_ensemble_stacking_seedavg_cv*.csv"))
df = pd.read_csv(sub_files[-1])
test_target_col = [c for c in df.columns if c != ID_COL][0]
test_pred = df[test_target_col].values

# Stacking OOF
oof_list = [oof_lgb, oof_xgb, oof_cb]
stacked_oof, _, _ = stack_with_ridge(y_true, oof_list, oof_list)

print(f"Orijinal CV: {rmse(y_true, stacked_oof):.5f}")

# YONTEM A: Naive (leakage var) — onceki sonucumuz
iso_naive = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=10)
iso_naive.fit(stacked_oof, y_true)
oof_naive = iso_naive.predict(stacked_oof)
print(f"\nNaive Isotonic CV (leakage'li): {rmse(y_true, oof_naive):.5f}")

# YONTEM B: KFold-aware (leakage yok) — gercek CV
print(f"\n--- KFold-aware Isotonic (gercek CV) ---")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_clean = np.zeros(len(y_true))
test_predictions = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(stacked_oof), 1):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=10)
    iso.fit(stacked_oof[tr_idx], y_true[tr_idx])
    oof_clean[va_idx] = iso.predict(stacked_oof[va_idx])
    test_predictions.append(iso.predict(test_pred))

oof_clean_clipped = np.clip(oof_clean, 0, 10)
test_clean = np.mean(test_predictions, axis=0)
test_clean = np.clip(test_clean, 0, 10)

clean_rmse = rmse(y_true, oof_clean_clipped)
print(f"  CV RMSE: {clean_rmse:.5f}")
print(f"  Delta vs orijinal: {clean_rmse - 1.21412:+.5f}")

# Bias profilleri kiyas
bins = [0, 2, 4, 5, 6, 8, 10]
target_bins = pd.cut(y_true, bins=bins, include_lowest=True)

for name, oof in [("Orijinal", stacked_oof),
                    ("Naive (leakage)", oof_naive),
                    ("KFold-aware", oof_clean_clipped)]:
    df_bias = pd.DataFrame({"bin": target_bins, "err": y_true - oof})
    bias = df_bias.groupby("bin", observed=True)["err"].mean()
    biases_str = ", ".join(f"{b:+.2f}" for b in bias.values)
    print(f"  {name:20s}: [{biases_str}]")

# Final submission (KFold-aware)
if clean_rmse < 1.21412:
    save_submission(test_clean, "corrected_isotonic_kfold", cv_rmse=clean_rmse)
    print(f"\n>>> KFold-aware Isotonic kaydedildi: {clean_rmse:.5f} <<<")
else:
    print(f"\n>>> KFold-aware iyilesme yetersiz, naive yontem leakage'liydi <<<")
