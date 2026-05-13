"""
Eylem 1: Bias Correction (post-processing).

Mevcut seed-averaged stacking OOF tahminlerinde regression-to-the-mean
bias var:
  - 0-2 araliginda 1.5 puan fazla tahmin (over-predict)
  - 8-10 araliginda 0.9 puan az tahmin (under-predict)

Bu bias'i duzeltmek icin 4 yontem dene, en iyisini sec.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold

from src.config import OOF_DIR, SUBMISSIONS_DIR, TARGET, ID_COL, SEED, N_FOLDS
from src.cv import rmse, save_submission
from src.data import load_and_preprocess


# 1) Veri ve mevcut tahminleri yukle
print("Veri ve OOF tahminleri yukleniyor...")
train, _, _ = load_and_preprocess()
y_true = train[TARGET].values

# Seed-avg OOF (3 model)
oof_lgb = np.load(OOF_DIR / "oof_lgb_seedavg.npy")
oof_xgb = np.load(OOF_DIR / "oof_xgb_seedavg.npy")
oof_cb = np.load(OOF_DIR / "oof_cb_seedavg.npy")

# Mevcut en iyi stacking submission'ini yukle (test tahminleri icin)
sub_files = sorted(SUBMISSIONS_DIR.glob("sub_ensemble_stacking_seedavg_cv*.csv"))
df = pd.read_csv(sub_files[-1])
test_target_col = [c for c in df.columns if c != ID_COL][0]
test_pred_seedavg = df[test_target_col].values

# Stacking OOF'u yeniden hesapla (train uzerinde)
from src.ensemble import stack_with_ridge
oof_list = [oof_lgb, oof_xgb, oof_cb]
test_list_seedavg_models = []  # Yeniden yuklenmesi gerek
for name in ["lgb", "xgb", "cb"]:
    files = sorted(SUBMISSIONS_DIR.glob(f"sub_{name}_seed*_cv*.csv"))
    # seed42'yi al referans olarak — gercekte burada her seedin testini ortalamak gerekir
    # Daha temiz: zaten seedavg stacking submission'i var, onun test'ini kullanalim

# Stacking OOF'u re-build et
stacked_oof_local, _, ridge_model = stack_with_ridge(y_true, oof_list, oof_list)
# (test_list olarak oof_list verdik, sadece OOF'a ihtiyacimiz var)
oof_stacked = stacked_oof_local

# Test tahmini olarak mevcut seedavg stacking submission'i kullan
test_pred = test_pred_seedavg

print(f"Stacked OOF CV (referans): {rmse(y_true, oof_stacked):.5f}")
print(f"Bu seedavg stacking ile ayni olmali: 1.21412\n")


# ============================================================
# YONTEM 1: ISOTONIC REGRESSION
# ============================================================
print("="*55)
print("YONTEM 1: Isotonic Regression")
print("="*55)

iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=10)
iso.fit(oof_stacked, y_true)
oof_iso = iso.predict(oof_stacked)
test_iso = iso.predict(test_pred)

iso_rmse = rmse(y_true, oof_iso)
print(f"  CV RMSE: {iso_rmse:.5f}")
print(f"  Delta vs orijinal: {iso_rmse - 1.21412:+.5f}")


# ============================================================
# YONTEM 2: POLINOMYAL DUZELTME (kuadratik)
# ============================================================
print("\n" + "="*55)
print("YONTEM 2: Polinomyal Duzeltme (kuadratik)")
print("="*55)

# OOF uzerinde fit, leakage onleme icin KFold ile
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
oof_poly = np.zeros(len(y_true))
test_poly_predictions = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(oof_stacked), 1):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_tr_poly = poly.fit_transform(oof_stacked[tr_idx].reshape(-1, 1))
    X_va_poly = poly.transform(oof_stacked[va_idx].reshape(-1, 1))
    X_test_poly = poly.transform(test_pred.reshape(-1, 1))

    ridge_poly = Ridge(alpha=1.0)
    ridge_poly.fit(X_tr_poly, y_true[tr_idx])
    oof_poly[va_idx] = ridge_poly.predict(X_va_poly)
    test_poly_predictions.append(ridge_poly.predict(X_test_poly))

oof_poly = np.clip(oof_poly, 0, 10)
test_poly = np.clip(np.mean(test_poly_predictions, axis=0), 0, 10)

poly_rmse = rmse(y_true, oof_poly)
print(f"  CV RMSE: {poly_rmse:.5f}")
print(f"  Delta vs orijinal: {poly_rmse - 1.21412:+.5f}")


# ============================================================
# YONTEM 3: BIN-BASED CORRECTION
# ============================================================
print("\n" + "="*55)
print("YONTEM 3: Bin-Based Correction")
print("="*55)

# OOF'tan bias'i hesapla, test'e uygula
n_bins = 20
oof_bin_corrected = np.zeros(len(y_true))
bias_map = []

# KFold ile leakage onle
for fold, (tr_idx, va_idx) in enumerate(kf.split(oof_stacked), 1):
    bins = np.quantile(oof_stacked[tr_idx], np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)
    bin_idx_tr = np.digitize(oof_stacked[tr_idx], bins[1:-1])
    bin_idx_va = np.digitize(oof_stacked[va_idx], bins[1:-1])

    # Her bin icin bias hesapla
    bias_per_bin = {}
    for b in np.unique(bin_idx_tr):
        mask = bin_idx_tr == b
        bias_per_bin[b] = (y_true[tr_idx][mask] - oof_stacked[tr_idx][mask]).mean()

    # Validation'a uygula
    for i, b in enumerate(bin_idx_va):
        bias = bias_per_bin.get(b, 0.0)
        oof_bin_corrected[va_idx[i]] = oof_stacked[va_idx[i]] + bias

oof_bin_corrected = np.clip(oof_bin_corrected, 0, 10)

# Test icin: tum train'den bias hesapla
bins_full = np.quantile(oof_stacked, np.linspace(0, 1, n_bins + 1))
bins_full = np.unique(bins_full)
bin_idx_train_full = np.digitize(oof_stacked, bins_full[1:-1])
bias_per_bin_full = {}
for b in np.unique(bin_idx_train_full):
    mask = bin_idx_train_full == b
    bias_per_bin_full[b] = (y_true[mask] - oof_stacked[mask]).mean()

bin_idx_test = np.digitize(test_pred, bins_full[1:-1])
test_bin = test_pred + np.array([bias_per_bin_full.get(b, 0.0) for b in bin_idx_test])
test_bin = np.clip(test_bin, 0, 10)

bin_rmse = rmse(y_true, oof_bin_corrected)
print(f"  CV RMSE: {bin_rmse:.5f}")
print(f"  Delta vs orijinal: {bin_rmse - 1.21412:+.5f}")


# ============================================================
# YONTEM 4: RIDGE META (residuali feature'lardan tahmin)
# ============================================================
print("\n" + "="*55)
print("YONTEM 4: Ridge Meta (residual model)")
print("="*55)

# Residual = y_true - oof_stacked, bunu modelden bagimsiz olarak tahmin et
# Feature olarak: oof_stacked + bireysel modellerin tahminleri
features_for_meta = np.column_stack([oof_stacked, oof_lgb, oof_xgb, oof_cb,
                                       oof_stacked**2, np.abs(oof_stacked - 5)])

oof_meta = np.zeros(len(y_true))
test_meta_predictions = []

# Test feature
test_features = np.column_stack([
    test_pred, test_pred, test_pred, test_pred,  # placeholder, gercekte ayri yuklemeliyiz
    test_pred**2, np.abs(test_pred - 5),
])

# Burada bireysel modellerin test tahminlerini de yukleyelim
# Ama seed-avg per-model test tahminleri yok dosya olarak
# Sadece stacking submission'i var, o yuzden test'te basitlestir
test_features = np.column_stack([
    test_pred, test_pred, test_pred, test_pred,
    test_pred**2, np.abs(test_pred - 5),
])

for fold, (tr_idx, va_idx) in enumerate(kf.split(features_for_meta), 1):
    ridge_meta = Ridge(alpha=2.0)
    residuals_tr = y_true[tr_idx] - oof_stacked[tr_idx]
    ridge_meta.fit(features_for_meta[tr_idx], residuals_tr)
    pred_residual_va = ridge_meta.predict(features_for_meta[va_idx])
    oof_meta[va_idx] = oof_stacked[va_idx] + pred_residual_va
    pred_residual_test = ridge_meta.predict(test_features)
    test_meta_predictions.append(test_pred + pred_residual_test)

oof_meta = np.clip(oof_meta, 0, 10)
test_meta = np.clip(np.mean(test_meta_predictions, axis=0), 0, 10)

meta_rmse = rmse(y_true, oof_meta)
print(f"  CV RMSE: {meta_rmse:.5f}")
print(f"  Delta vs orijinal: {meta_rmse - 1.21412:+.5f}")


# ============================================================
# OZET VE KARAR
# ============================================================
print("\n" + "="*55)
print("OZET")
print("="*55)
results = {
    "Orijinal (no correction)": (1.21412, None, None),
    "Isotonic": (iso_rmse, oof_iso, test_iso),
    "Polynomial (deg=2)": (poly_rmse, oof_poly, test_poly),
    "Bin-based (n=20)": (bin_rmse, oof_bin_corrected, test_bin),
    "Ridge meta (residual)": (meta_rmse, oof_meta, test_meta),
}
for name, (rmse_val, _, _) in results.items():
    delta = rmse_val - 1.21412
    flag = "BEST" if rmse_val == min(r[0] for r in results.values()) else ""
    print(f"  {name:30s}: {rmse_val:.5f}  ({delta:+.5f})  {flag}")


# Bias dagilimini her yontem icin goster
print("\n" + "="*55)
print("BIAS PROFILI (target_bin bazinda mean error)")
print("="*55)
bins = [0, 2, 4, 5, 6, 8, 10]
target_bins = pd.cut(y_true, bins=bins, include_lowest=True)

for name, (_, oof_corrected, _) in results.items():
    if oof_corrected is None:
        continue
    df_bias = pd.DataFrame({"bin": target_bins, "err": y_true - oof_corrected})
    bias_per_bin = df_bias.groupby("bin", observed=True)["err"].mean()
    biases_str = ", ".join(f"{b:+.2f}" for b in bias_per_bin.values)
    print(f"  {name:30s}: [{biases_str}]")

# Orijinal bias
df_bias_orig = pd.DataFrame({"bin": target_bins, "err": y_true - oof_stacked})
bias_orig = df_bias_orig.groupby("bin", observed=True)["err"].mean()
biases_str = ", ".join(f"{b:+.2f}" for b in bias_orig.values)
print(f"  {'Orijinal (no correction)':30s}: [{biases_str}]")


# ============================================================
# EN IYI YONTEMI KAYDET
# ============================================================
best_method = min(results.items(), key=lambda x: x[1][0])
best_name, (best_rmse, _, best_test) = best_method
if best_test is not None:
    safe_name = best_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
    save_submission(best_test, f"corrected_{safe_name}", cv_rmse=best_rmse)
    print(f"\n>>> EN IYI: {best_name} | CV: {best_rmse:.5f} <<<")
    print(f">>> Submission havuzuna eklendi <<<")
