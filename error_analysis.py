"""
Hata analizi: CB tuned modelin OOF tahminlerini kullanarak
modelin nerede yanildigini bul.

Cikti: detayli rapor + figurler
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import OOF_DIR, TARGET, ID_COL
from src.cv import rmse
from src.data import load_and_preprocess
from src.features import make_features_v1

# Klasor olustur
fig_dir = Path("outputs/error_analysis")
fig_dir.mkdir(parents=True, exist_ok=True)

print("Veri ve OOF tahminleri yukleniyor...")
train, _, _ = load_and_preprocess()
train_fe = make_features_v1(train)
y_true = train[TARGET].values

# CB tuned OOF
oof_cb = np.load(OOF_DIR / "oof_cb_tuned_v1.npy")
errors = y_true - oof_cb
abs_errors = np.abs(errors)

print(f"\nGENEL ISTATISTIKLER:")
print(f"  CV RMSE: {rmse(y_true, oof_cb):.5f}")
print(f"  MAE: {abs_errors.mean():.5f}")
print(f"  Median |error|: {np.median(abs_errors):.5f}")
print(f"  Max |error|: {abs_errors.max():.5f}")
print(f"  Std error: {errors.std():.5f}")
print(f"  Mean error (bias): {errors.mean():+.5f}")

# 1. Hata buyuklugu dagilimi
print(f"\n--- HATA BUYUKLUGU DAGILIMI ---")
quantiles = [0.5, 0.75, 0.90, 0.95, 0.99]
for q in quantiles:
    print(f"  %{int(q*100)} altinda: |error| < {np.quantile(abs_errors, q):.4f}")

# Buyuk hatali yuzdesi
big_error_threshold = 2.0
n_big = (abs_errors > big_error_threshold).sum()
print(f"  |error| > {big_error_threshold}: {n_big} satir ({100*n_big/len(y_true):.1f}%)")

# 2. Hatay target ile karsilastir
print(f"\n--- HATA vs TARGET ARALIGI ---")
target_bins = [0, 2, 4, 5, 6, 8, 10]
train_fe = train_fe.copy()
train_fe["abs_error"] = abs_errors
train_fe["error"] = errors
train_fe["target_bin"] = pd.cut(y_true, bins=target_bins, include_lowest=True)
bin_stats = train_fe.groupby("target_bin", observed=True)["abs_error"].agg(["mean", "median", "count"])
print(bin_stats.round(3))

# 3. En kotu 200 satiri analiz et
print(f"\n--- EN KOTU 200 SATIR (worst predictions) ---")
worst_idx = np.argsort(abs_errors)[-200:]
worst = train_fe.iloc[worst_idx].copy()

print(f"\nWorst 200 vs tum data:")
features_to_check = [
    "yas", "stres_skoru", "rem_yuzdesi", "derin_uyku_yuzdesi",
    "gunluk_calisma_saati", "vucut_kitle_indeksi", "gecelik_uyanma_sayisi",
    "uyku_oncesi_kafein_mg", "uyku_oncesi_ekran_suresi_dk",
    "gunluk_adim_sayisi", "sekerleme_suresi_dk", "dinlenik_nabiz_bpm",
    "oda_sicakligi_celsius", "hafta_sonu_uyku_farki_saat",
]
for feat in features_to_check:
    if feat in train_fe.columns:
        all_mean = train_fe[feat].mean()
        worst_mean = worst[feat].mean()
        diff_pct = 100 * (worst_mean - all_mean) / (abs(all_mean) + 1e-6)
        if abs(diff_pct) > 5:  # En az %5 fark varsa goster
            print(f"  {feat}: tum={all_mean:.2f} | worst200={worst_mean:.2f} | fark={diff_pct:+.1f}%")

# 4. Kategorik degiskenlerde hata orani
print(f"\n--- KATEGORIK DEGISKENLERDE HATA ---")
cat_cols = ["cinsiyet", "meslek", "ulke", "kronotip", "ruh_sagligi_durumu",
            "mevsim", "gun_tipi"]
for col in cat_cols:
    if col not in train_fe.columns:
        continue
    grp = train_fe.groupby(col, observed=True)["abs_error"].agg(["mean", "count"])
    grp = grp.sort_values("mean", ascending=False)
    if len(grp) <= 1:
        continue
    spread = grp["mean"].max() - grp["mean"].min()
    if spread > 0.05:  # Anlamli spread varsa goster
        print(f"\n  {col} (spread={spread:.3f}):")
        print(grp.round(3).to_string())

# 5. Eksik degerlerin etkisi
print(f"\n--- EKSIK DEGERLERIN HATAYA ETKISI ---")
cols_with_missing = [
    "meslek", "vucut_kitle_indeksi", "uyku_oncesi_kafein_mg",
    "stres_skoru", "kronotip", "ruh_sagligi_durumu",
]
for col in cols_with_missing:
    is_missing = train[col].isna()
    if is_missing.sum() == 0:
        continue
    err_missing = abs_errors[is_missing].mean()
    err_present = abs_errors[~is_missing].mean()
    diff = err_missing - err_present
    print(f"  {col}: NaN |err|={err_missing:.4f} vs not-NaN |err|={err_present:.4f}  (fark={diff:+.4f})")

# 6. Bias analizi: model fazla mi yuksek/dusuk tahmin ediyor?
print(f"\n--- BIAS ANALIZI (target_bin bazinda) ---")
bias_stats = train_fe.groupby("target_bin", observed=True)["error"].agg(["mean", "median"])
print(bias_stats.round(3))
print("\n  Pozitif = model dusuk tahmin etmis (under-predict)")
print("  Negatif = model yuksek tahmin etmis (over-predict)")

# 7. Gorsel ciktilar
print(f"\nGorsel ciktilar uretiliyor: {fig_dir}")

# Figur 1: Predicted vs True scatter
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.scatter(y_true, oof_cb, alpha=0.1, s=5)
ax.plot([0, 10], [0, 10], "r--", label="ideal")
ax.set_xlabel("Gercek")
ax.set_ylabel("Tahmin")
ax.set_title("Predicted vs True (CB tuned)")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(errors, bins=80, edgecolor="white", alpha=0.7)
ax.axvline(0, color="red", linestyle="--")
ax.set_xlabel("Error (gercek - tahmin)")
ax.set_ylabel("Frekans")
ax.set_title("Hata dagilimi")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / "fig1_predicted_vs_true.png", dpi=100, bbox_inches="tight")
plt.close()

# Figur 2: Target bin'lere gore hata
fig, ax = plt.subplots(figsize=(10, 5))
train_fe.boxplot(column="abs_error", by="target_bin", ax=ax, showfliers=False)
ax.set_xlabel("Target Bin")
ax.set_ylabel("|Hata|")
ax.set_title("Target aralik bazinda mutlak hata dagilimi")
plt.suptitle("")
plt.tight_layout()
plt.savefig(fig_dir / "fig2_error_by_target.png", dpi=100, bbox_inches="tight")
plt.close()

# Figur 3: Eksik vs dolu hata kiyasi
fig, ax = plt.subplots(figsize=(10, 5))
data_for_plot = []
for col in cols_with_missing:
    is_missing = train[col].isna()
    if is_missing.sum() == 0:
        continue
    data_for_plot.append({"col": col, "type": "NaN", "err": abs_errors[is_missing].mean()})
    data_for_plot.append({"col": col, "type": "var", "err": abs_errors[~is_missing].mean()})
df_plot = pd.DataFrame(data_for_plot)
sns.barplot(data=df_plot, x="col", y="err", hue="type", ax=ax)
ax.set_xlabel("Sutun")
ax.set_ylabel("Ortalama |hata|")
ax.set_title("Eksik vs dolu degerlerde ortalama hata")
ax.tick_params(axis="x", rotation=20)
plt.tight_layout()
plt.savefig(fig_dir / "fig3_missing_impact.png", dpi=100, bbox_inches="tight")
plt.close()

print(f"\nTamamlandi. Figurler: {fig_dir}")
