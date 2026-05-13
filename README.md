# YZTA Datathon 2026 — Bilişsel Performans Skoru Tahmini

5 günlük datathon yarışması için modüler ML pipeline.

**Takım:** Datathon Takım [X]
**Hedef:** `bilissel_performans_skoru` (regresyon, RMSE)
**Public/Private split:** %80 / %20

## Proje Yapısı

```
yzta_datathon/
├── data/                          # train.csv, test_x.csv, sample_submission.csv
├── src/
│   ├── config.py                  # yollar, sabitler, seed
│   ├── data.py                    # load + preprocess + ülke normalize
│   ├── features.py                # FE fonksiyonları (v1, v2 ...)
│   ├── models.py                  # LGB / XGB / CB train fonksiyonları
│   ├── cv.py                      # CV runner, OOF/submission/log kaydı
│   └── ensemble.py                # average + Ridge stacking
├── notebooks/
│   └── 01_eda.ipynb               # keşifsel analiz, görselleştirmeler
├── outputs/
│   ├── submissions/               # sub_*.csv (CV skorlu adlandırma)
│   ├── oof/                       # OOF tahminleri (.npy)
│   └── logs/experiments.jsonl     # her deneyin kaydı
├── run_baseline.py                # F1: LGBM baseline
├── run_fe_v1.py                   # F2: FE v1 + 3 model
├── run_ensemble.py                # F3: ensemble
└── requirements.txt
```

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Veri dosyalarını `data/` altına yerleştir (train.csv, test_x.csv, sample_submission.csv).

## Faz Akışı

| Faz | Komut | Çıktı |
|---|---|---|
| F1 — Baseline | `python run_baseline.py` | `sub_lgb_baseline_cv*.csv` |
| F2 — FE v1 + 3 model | `python run_fe_v1.py` | `sub_lgb_fe_v1_cv*.csv`, xgb, cb |
| F3 — Ensemble | `python run_ensemble.py` | `sub_ensemble_*_cv*.csv` |

Her faz idempotent: aynı seed → aynı sonuç. Submission dosya adı CV RMSE'yi
içerir, böylece havuzdan en iyiyi kolayca seçersin.

## Submission Stratejisi

**Submission havuzu yaklaşımı:**
- Her zaman ≥2 farklı kaliteli `sub_*.csv` dosyası hazır olsun.
- Submission fırsatı çıkınca havuzdan en yüksek lokal CV'li dosya seçilir.
- Final için 2 dosya: bir "robust ensemble" + bir "best CV single".

**Kritik kural:** Submission yapmadan önce lokal CV'de iyileşme görmek şart.
Submission'lar deneme aracı değil, kalibrasyon noktasıdır.

## Reproducibility

- Tüm rastgelelik kaynakları `SEED=42`'ye bağlı.
- 5-fold KFold sabit (`shuffle=True, random_state=42`).
- Tüm deneyler `outputs/logs/experiments.jsonl` dosyasına kaydedilir.

## Yarışma Kuralları (kısa hatırlatma)

- Tek Kaggle hesabı.
- Profesyonel destek alınmaz; takım kendi çabasıyla çalışır.
- Akademinin verdiği takım ismi kullanılır.
- Günde max 3 submission (5 kişilik takım → ben max 1, fırsata göre artar).
- Final için 2 submission seçilir.
- İlk 10 takım kod paylaşmak zorunda → kod kalitesi kritik.
