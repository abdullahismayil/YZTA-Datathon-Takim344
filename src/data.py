"""
Veri yükleme ve temel ön işleme.

Buradaki preprocess SADECE veri kalitesi düzeltmeleri içerir
(ülke normalize, tip dönüşümü). Eksik değer imputation YAPMAYIZ —
LightGBM/XGBoost/CatBoost eksikleri kendileri ele alıyor; baseline
için en temiz yol bu.
"""
from __future__ import annotations
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

from .config import (
    TRAIN_PATH, TEST_PATH, SAMPLE_SUB_PATH,
    ID_COL, TARGET, CATEGORICAL_COLS, SEED,
)


def seed_everything(seed: int = SEED) -> None:
    """Tüm rastgelelik kaynaklarını sabit seed'e bağlar."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---- Veri kalitesi: ülke normalize ----
# EDA bulgusu: 3 ülke iki kez kodlanmış, birleştiriyoruz.
# (Spain/Ispanya, South Korea/Guney Kore, Sweden/Isvec)
COUNTRY_MAPPING = {
    "Spain": "Ispanya",
    "South Korea": "Guney Kore",
    "Sweden": "Isvec",
}


def normalize_country(s: pd.Series) -> pd.Series:
    """Ülke isimlerinde Türkçe/İngilizce dublikasyonları gider."""
    return s.map(lambda x: COUNTRY_MAPPING.get(x, x) if pd.notna(x) else x)


def load_raw() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train, test ve sample submission'ı ham haliyle okur."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    return train, test, sample_sub


def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Train ve test'e uygulanacak temel temizlik.

    - Ülke isimlerini normalize et
    - Kategorik sütunları string'e çevir (eksikler NaN kalır)
    """
    df = df.copy()
    df["ulke"] = normalize_country(df["ulke"])
    return df


def to_categorical(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cat_cols: list[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Kategorik sütunları pandas 'category' tipine çevirir.

    Test'teki kategorileri train kategorilerine göre hizalar — train'de
    görülmemiş bir kategori test'te varsa NaN olur (LightGBM bunu
    eksik değer gibi ele alabilir).
    """
    if cat_cols is None:
        cat_cols = CATEGORICAL_COLS
    train = train.copy()
    test = test.copy()
    for col in cat_cols:
        train[col] = train[col].astype("category")
        test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)
    return train, test


def load_and_preprocess() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tek çağrıda ham veriyi yükler + preprocess eder."""
    train, test, sample_sub = load_raw()
    train = basic_preprocess(train)
    test = basic_preprocess(test)
    return train, test, sample_sub


if __name__ == "__main__":
    # Hızlı sanity check
    seed_everything()
    train, test, sample_sub = load_and_preprocess()
    print(f"Train: {train.shape}, Test: {test.shape}")
    print(f"Hedef ortalaması: {train[TARGET].mean():.3f}")
    print(f"Ülke unique sayısı (train): {train['ulke'].nunique()}")
    print(f"Ülke unique sayısı (test): {test['ulke'].nunique()}")
