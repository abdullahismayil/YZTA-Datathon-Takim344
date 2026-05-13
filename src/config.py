"""
Proje sabitleri: yollar, seed, sütun adları.
Bütün diğer modüller buradan import eder.
"""
from __future__ import annotations
from pathlib import Path

# ---- Yollar ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SUBMISSIONS_DIR = OUTPUTS_DIR / "submissions"
OOF_DIR = OUTPUTS_DIR / "oof"
LOGS_DIR = OUTPUTS_DIR / "logs"

for d in [SUBMISSIONS_DIR, OOF_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_x.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

# ---- Sütun adları ----
ID_COL = "id"
TARGET = "bilissel_performans_skoru"

# Veri setindeki kategorik sütunlar (EDA ile teyit edildi)
CATEGORICAL_COLS = [
    "cinsiyet",
    "meslek",
    "ulke",
    "kronotip",
    "ruh_sagligi_durumu",
    "mevsim",
    "gun_tipi",
]

# Sayısal sütunlar (id ve hedef hariç)
NUMERIC_COLS = [
    "yas",
    "vucut_kitle_indeksi",
    "rem_yuzdesi",
    "derin_uyku_yuzdesi",
    "uykuya_dalma_suresi_dk",
    "gecelik_uyanma_sayisi",
    "uyku_oncesi_kafein_mg",
    "uyku_oncesi_ekran_suresi_dk",
    "gunluk_adim_sayisi",
    "sekerleme_suresi_dk",
    "stres_skoru",
    "gunluk_calisma_saati",
    "dinlenik_nabiz_bpm",
    "oda_sicakligi_celsius",
    "hafta_sonu_uyku_farki_saat",
]

# ---- Reproducibility ----
SEED = 42
N_FOLDS = 5
