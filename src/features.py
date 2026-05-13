"""
Feature engineering.

Faz mimarisi:
- v0: hiçbir FE yapılmaz (baseline'da kullanılır)
- v1: domain bilgisi tabanlı temel feature'lar
- v2: ileri etkileşimler, oranlar, target encoding (sonraki fazlarda)

Her fonksiyon idempotent olmalı: aynı input → aynı output.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ============================================================
# v1: TEMEL FEATURE ENGINEERING
# ============================================================

def add_sleep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Uyku kalitesi türev değişkenleri.

    Fikir: uyku ile ilgili 5+ ham değişken var; bunları
    "verimlilik" gibi yoğun göstergelere indirgemek modele yardımcı olur.
    """
    df = df.copy()

    # REM + Derin uyku toplam yüzdesi (yüksek = kaliteli uyku)
    df["fe_kaliteli_uyku_yuzdesi"] = (
        df["rem_yuzdesi"].fillna(0) + df["derin_uyku_yuzdesi"].fillna(0)
    )

    # REM / Derin uyku oranı (uyku mimarisi göstergesi)
    df["fe_rem_derin_orani"] = (
        df["rem_yuzdesi"] / df["derin_uyku_yuzdesi"].replace(0, np.nan)
    )

    # Uykuya dalma süresi + gecelik uyanma → "uyku bozulması" göstergesi
    df["fe_uyku_bozulmasi"] = (
        df["uykuya_dalma_suresi_dk"].fillna(0) * 0.5
        + df["gecelik_uyanma_sayisi"].fillna(0) * 5  # her uyanma ~5dk değer
    )

    # Sosyal jetlag göstergesi: |hafta_sonu_uyku_farki|
    # mutlak değer, çünkü hem fazla hem az fark zararlı
    df["fe_sosyal_jetlag"] = df["hafta_sonu_uyku_farki_saat"].abs()

    return df


def add_lifestyle_features(df: pd.DataFrame) -> pd.DataFrame:
    """Yaşam tarzı türev değişkenleri."""
    df = df.copy()

    # Stres × çalışma saati (tükenmişlik proksisi)
    df["fe_stres_x_calisma"] = (
        df["stres_skoru"].fillna(df["stres_skoru"].median())
        * df["gunluk_calisma_saati"].fillna(df["gunluk_calisma_saati"].median())
    )

    # Aktivite skoru: adım sayısı log dönüşümlü (genelde sağa çarpık)
    df["fe_log_adim"] = np.log1p(df["gunluk_adim_sayisi"].fillna(0))

    # Ekran süresi × kafein (uyku öncesi uyarıcı yükü)
    df["fe_uyku_oncesi_uyarim"] = (
        df["uyku_oncesi_ekran_suresi_dk"].fillna(0) / 60.0
        + df["uyku_oncesi_kafein_mg"].fillna(0) / 100.0
    )

    return df


def add_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Demografik türev değişkenleri."""
    df = df.copy()

    # Yaş grubu (ordinal, ağaç tabanlı modeller bunu zaten yakalar
    # ama ek feature olarak da bulunsun)
    df["fe_yas_grubu"] = pd.cut(
        df["yas"],
        bins=[0, 25, 35, 45, 55, 100],
        labels=[0, 1, 2, 3, 4],
    ).astype("float")  # NaN olabilir diye float

    # VKİ kategorisi (WHO sınıfları)
    df["fe_vki_kategori"] = pd.cut(
        df["vucut_kitle_indeksi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3],
    ).astype("float")

    # Nabız × yaş (kardiyovasküler proksi)
    df["fe_nabiz_x_yas"] = df["dinlenik_nabiz_bpm"] * df["yas"] / 100.0

    return df


def make_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """v1 paketi: tüm temel FE'leri uygular.
    Train ve test'e ayrı ayrı uygulanmalı.
    """
    df = add_sleep_features(df)
    df = add_lifestyle_features(df)
    df = add_demographic_features(df)
    return df


# ============================================================
# v2: GELİŞMİŞ FEATURE ENGINEERING
# Etkileşimler, polinomlar, group stats, missing indicators, ratios
# Target encoding ayrı fonksiyonda (KFold-aware, leakage-safe)
# ============================================================

def add_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Her eksik içeren sütun için binary 'is_missing' feature.
    EDA'da 6 sütunda eksik vardı; eksiklik kalıbı kendisi bilgi olabilir."""
    df = df.copy()
    cols_with_missing = [
        "meslek", "vucut_kitle_indeksi", "uyku_oncesi_kafein_mg",
        "stres_skoru", "kronotip", "ruh_sagligi_durumu",
    ]
    for col in cols_with_missing:
        df[f"fe_missing_{col}"] = df[col].isna().astype("int8")
    # Toplam eksik sayısı (kişinin "eksik veri yoğunluğu")
    df["fe_total_missing"] = df[
        [f"fe_missing_{c}" for c in cols_with_missing]
    ].sum(axis=1).astype("int8")
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain bilgisinden gelen ikili etkileşimler.

    Hipotezler:
    - Yüksek stres × gece insanı → en kötü performans kombinasyonu
    - Yaş arttıkça çalışma saatinin etkisi büyür
    - VKİ × düşük adım → metabolik risk
    """
    df = df.copy()

    # Stres × ekran süresi (uyku öncesi yorgunluk yükü)
    df["fe_stres_x_ekran"] = (
        df["stres_skoru"].fillna(df["stres_skoru"].median())
        * df["uyku_oncesi_ekran_suresi_dk"].fillna(0) / 60.0
    )

    # Yaş × çalışma saati (yaşa göre çalışma yükü etkisi)
    df["fe_yas_x_calisma"] = df["yas"] * df["gunluk_calisma_saati"].fillna(0)

    # VKİ × Adım (metabolik denge proxy)
    df["fe_vki_x_log_adim"] = (
        df["vucut_kitle_indeksi"].fillna(df["vucut_kitle_indeksi"].median())
        * np.log1p(df["gunluk_adim_sayisi"].fillna(0))
    )

    # Gecelik uyanma × yaş (yaşla birlikte uyanmaların etkisi büyür)
    df["fe_uyanma_x_yas"] = df["gecelik_uyanma_sayisi"].fillna(0) * df["yas"]

    # Şekerleme × gecelik uyanma (gündüz uykusu telafi mi yoksa zarar mı?)
    df["fe_sekerleme_x_uyanma"] = (
        df["sekerleme_suresi_dk"].fillna(0)
        * df["gecelik_uyanma_sayisi"].fillna(0)
    )

    return df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hedefle güçlü doğrusal korelasyonu olan değişkenlerin
    karelerini ekle. Ağaç modelleri bu eğriyi yakalar ama bazen
    explicit feature daha hızlı ulaşır."""
    df = df.copy()
    df["fe_stres_kare"] = df["stres_skoru"].fillna(df["stres_skoru"].median()) ** 2
    df["fe_rem_kare"] = df["rem_yuzdesi"].fillna(df["rem_yuzdesi"].median()) ** 2
    df["fe_derin_uyku_kare"] = df["derin_uyku_yuzdesi"].fillna(df["derin_uyku_yuzdesi"].median()) ** 2
    df["fe_calisma_kare"] = df["gunluk_calisma_saati"].fillna(df["gunluk_calisma_saati"].median()) ** 2
    return df


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Anlamlı oranlar."""
    df = df.copy()
    eps = 1e-3

    # Kafein / VKİ — vücut kütlesine göre kafein yoğunluğu
    df["fe_kafein_per_vki"] = (
        df["uyku_oncesi_kafein_mg"].fillna(0)
        / (df["vucut_kitle_indeksi"].fillna(df["vucut_kitle_indeksi"].median()) + eps)
    )

    # Ekran / uyku dalma süresi (dijital alışkanlık → uyku gecikmesi)
    df["fe_ekran_per_uyku_dalma"] = (
        df["uyku_oncesi_ekran_suresi_dk"].fillna(0)
        / (df["uykuya_dalma_suresi_dk"].fillna(0) + eps)
    )

    # Adım / Çalışma saati (oturarak vs aktif yaşam)
    df["fe_adim_per_calisma"] = (
        df["gunluk_adim_sayisi"].fillna(0)
        / (df["gunluk_calisma_saati"].fillna(0) + eps)
    )

    # Stres / Adım (yüksek stres düşük aktivite — endişeli sedanter)
    df["fe_stres_per_log_adim"] = (
        df["stres_skoru"].fillna(df["stres_skoru"].median())
        / (np.log1p(df["gunluk_adim_sayisi"].fillna(0)) + eps)
    )

    return df


def add_group_statistics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    group_cols: list[str] | None = None,
    agg_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train üzerinden hesaplanan grup istatistikleri (mean/std)
    train ve test'e map'lenir.

    NOT: Bu, target değil sadece feature istatistikleri kullandığı için
    leakage YOK. (Target encoding için ayrı KFold-aware fonksiyon var.)
    """
    if group_cols is None:
        group_cols = ["ulke", "meslek", "kronotip"]
    if agg_cols is None:
        agg_cols = ["yas", "stres_skoru", "gunluk_calisma_saati",
                    "rem_yuzdesi", "vucut_kitle_indeksi"]

    train = train.copy()
    test = test.copy()

    for gc in group_cols:
        if gc not in train.columns:
            continue
        for ac in agg_cols:
            if ac not in train.columns:
                continue
            stats = train.groupby(gc, observed=True)[ac].agg(["mean", "std"])
            stats.columns = [f"fe_grp_{gc}_{ac}_mean", f"fe_grp_{gc}_{ac}_std"]
            train = train.merge(stats, left_on=gc, right_index=True, how="left")
            test = test.merge(stats, left_on=gc, right_index=True, how="left")

            # Sapma feature'ı: bireyin grup ortalamasından farkı
            train[f"fe_grp_{gc}_{ac}_dev"] = (
                train[ac] - train[f"fe_grp_{gc}_{ac}_mean"]
            )
            test[f"fe_grp_{gc}_{ac}_dev"] = (
                test[ac] - test[f"fe_grp_{gc}_{ac}_mean"]
            )

    return train, test


def make_features_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v2 paketi: v1 + tüm gelişmiş feature'lar.

    Train ve test'i birlikte alır çünkü group statistics train'den
    hesaplanıp test'e uygulanıyor.
    """
    # v1: tek-tablo feature'lar
    train = make_features_v1(train)
    test = make_features_v1(test)

    # v2: tek-tablo feature'lar
    for fn in [add_missing_indicators, add_interaction_features,
               add_polynomial_features, add_ratio_features]:
        train = fn(train)
        test = fn(test)

    # v2: train-bağımlı (group stats)
    train, test = add_group_statistics(train, test)

    return train, test


# ============================================================
# TARGET ENCODING — KFold-aware (leakage-safe)
# ============================================================

def add_target_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
    cat_cols: list[str],
    n_splits: int = 5,
    smoothing: float = 10.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KFold-aware target encoding.

    Train'de her satır için, o satırın bulunduğu fold HARİÇ
    diğer foldların hedef ortalaması kullanılır. Bu, leakage'ı önler.
    Test için tüm train'in hedef ortalaması kullanılır.

    Smoothing formula:
        encoded = (n * group_mean + smoothing * global_mean) / (n + smoothing)

    Az sayıda örneği olan kategorilerde global ortalamaya çekilme yapılır.
    """
    from sklearn.model_selection import KFold

    train = train.copy()
    test = test.copy()
    global_mean = train[target_col].mean()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for col in cat_cols:
        new_col = f"fe_te_{col}"

        # --- Train için OOF target encoding ---
        oof = np.full(len(train), global_mean)
        for tr_idx, va_idx in kf.split(train):
            tr_part = train.iloc[tr_idx]
            stats = tr_part.groupby(col, observed=True)[target_col].agg(["mean", "count"])
            stats["smoothed"] = (
                stats["count"] * stats["mean"] + smoothing * global_mean
            ) / (stats["count"] + smoothing)
            mapping = stats["smoothed"].to_dict()
            oof[va_idx] = train.iloc[va_idx][col].map(mapping).fillna(global_mean).values
        train[new_col] = oof

        # --- Test için tüm train'den encoding ---
        stats = train.groupby(col, observed=True)[target_col].agg(["mean", "count"])
        stats["smoothed"] = (
            stats["count"] * stats["mean"] + smoothing * global_mean
        ) / (stats["count"] + smoothing)
        mapping = stats["smoothed"].to_dict()
        test[new_col] = test[col].map(mapping).fillna(global_mean)

    return train, test


# ============================================================
# Yardımcı: feature ekledikten sonra sütun listesi al
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
    id_col: str = "id",
    target_col: str = "bilissel_performans_skoru",
) -> list[str]:
    """ID ve TARGET hariç tüm sütunları döner."""
    return [c for c in df.columns if c not in (id_col, target_col)]


if __name__ == "__main__":
    # Hızlı sanity check
    from .data import load_and_preprocess
    train, test, _ = load_and_preprocess()
    train_fe = make_features_v1(train)
    test_fe = make_features_v1(test)
    new_cols = [c for c in train_fe.columns if c.startswith("fe_")]
    print(f"Eklenen FE sütunları ({len(new_cols)}):")
    for c in new_cols:
        print(f"  {c}: dtype={train_fe[c].dtype}, NaN={train_fe[c].isna().sum()}")
