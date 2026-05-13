"""
B: Stres-NaN i\u00e7in alt-model yakla\u015f\u0131m\u0131.

\u0130ki strateji:
  1. Iterative imputation: NaN stres_skoru'nu di\u011fer feature'lardan tahmin et
  2. NaN-flag + interaction: NaN olma durumunu agressifce feature olarak kullan

CB tuned ile bu iki yakla\u015f\u0131m\u0131 dene, CV'leri kar\u015f\u0131la\u015ft\u0131r.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import load_and_preprocess, to_categorical, seed_everything
from src.features import make_features_v1, get_feature_columns
from src.models import train_one_fold_cb
from src.cv import run_cv, save_oof, save_submission, rmse
from src.config import ID_COL, TARGET, SEED, LOGS_DIR


def load_cb_best_params():
    with open(LOGS_DIR / "best_params_catboost.json") as f:
        return json.load(f)


def main():
    seed_everything()
    cb_params = load_cb_best_params()

    print("Veri + FE v1 yukleniyor...")
    train, test, _ = load_and_preprocess()

    # FE v1 (NaN'lar bozulmadan)
    train_fe = make_features_v1(train)
    test_fe = make_features_v1(test)

    print(f"Stres NaN: train={train['stres_skoru'].isna().sum()}, test={test['stres_skoru'].isna().sum()}")

    # ============================================================
    # YAKLA\u015eIM 1: Iterative imputation
    # ============================================================
    print("\n" + "="*55)
    print("YAKLASIM 1: Iterative Imputation")
    print("="*55)

    # Imputation icin sadece sayisal feature'lar (kategorikler kategorik kalmali)
    numeric_for_imp = [c for c in train_fe.columns
                       if c not in (ID_COL, TARGET) and
                       train_fe[c].dtype in [np.float64, np.int64, np.int32]]
    print(f"Imputation feature sayisi: {len(numeric_for_imp)}")

    train_imp = train_fe.copy()
    test_imp = test_fe.copy()

    # IterativeImputer (HGB tabanli, hizli)
    imputer = IterativeImputer(
        estimator=HistGradientBoostingRegressor(max_iter=100, random_state=SEED),
        max_iter=5,
        random_state=SEED,
        verbose=0,
    )

    print("Iterative imputer fit ediliyor (train+test birlikte)...")
    combined = pd.concat([train_imp[numeric_for_imp], test_imp[numeric_for_imp]],
                         axis=0, ignore_index=True)
    combined_imputed = imputer.fit_transform(combined)
    n_train = len(train_imp)
    train_imp[numeric_for_imp] = combined_imputed[:n_train]
    test_imp[numeric_for_imp] = combined_imputed[n_train:]

    # Stres_skoru ve digerleri artik dolu
    print(f"Sonra stres NaN: train={train_imp['stres_skoru'].isna().sum()}, test={test_imp['stres_skoru'].isna().sum()}")

    train_imp, test_imp = to_categorical(train_imp, test_imp)
    FEATURES_IMP = get_feature_columns(train_imp, ID_COL, TARGET)
    X_imp = train_imp[FEATURES_IMP]
    y = train_imp[TARGET]
    X_test_imp = test_imp[FEATURES_IMP]

    print(f"Imputed CB tuned egitiliyor ({len(FEATURES_IMP)} features)...")
    fold_kwargs = {"params": {**cb_params, "loss_function": "RMSE",
                              "verbose": 0, "allow_writing_files": False,
                              "random_seed": SEED},
                   "iterations": 5000, "early_stopping": 200}
    result_imp = run_cv(X_imp, y, X_test_imp,
                       train_one_fold=train_one_fold_cb,
                       experiment_name="cb_imputed",
                       fold_kwargs=fold_kwargs)

    cv_imp = result_imp["cv_rmse"]
    save_oof(result_imp["oof"], "cb_imputed")
    save_submission(result_imp["test_preds"], "cb_imputed", cv_rmse=cv_imp)


    # ============================================================
    # YAKLA\u015eIM 2: NaN-flag + interaction features
    # ============================================================
    print("\n" + "="*55)
    print("YAKLASIM 2: NaN-flag + interaction features")
    print("="*55)

    train_fe2 = train_fe.copy()
    test_fe2 = test_fe.copy()

    # NaN-flag'leri ekle
    for col in ["stres_skoru", "ruh_sagligi_durumu", "meslek",
                "vucut_kitle_indeksi", "uyku_oncesi_kafein_mg", "kronotip"]:
        train_fe2[f"fe_isna_{col}"] = train_fe2[col].isna().astype("int8")
        test_fe2[f"fe_isna_{col}"] = test_fe2[col].isna().astype("int8")

    # Toplam NaN sayisi
    nan_cols = [f"fe_isna_{c}" for c in ["stres_skoru", "ruh_sagligi_durumu",
                                          "meslek", "vucut_kitle_indeksi",
                                          "uyku_oncesi_kafein_mg", "kronotip"]]
    train_fe2["fe_total_nan"] = train_fe2[nan_cols].sum(axis=1)
    test_fe2["fe_total_nan"] = test_fe2[nan_cols].sum(axis=1)

    # Etkilesim: stres_NaN × diger feature'lar
    train_fe2["fe_stresNaN_x_yas"] = train_fe2["fe_isna_stres_skoru"] * train_fe2["yas"]
    test_fe2["fe_stresNaN_x_yas"] = test_fe2["fe_isna_stres_skoru"] * test_fe2["yas"]

    train_fe2["fe_stresNaN_x_calisma"] = (
        train_fe2["fe_isna_stres_skoru"] *
        train_fe2["gunluk_calisma_saati"].fillna(0)
    )
    test_fe2["fe_stresNaN_x_calisma"] = (
        test_fe2["fe_isna_stres_skoru"] *
        test_fe2["gunluk_calisma_saati"].fillna(0)
    )

    train_fe2["fe_stresNaN_x_uyanma"] = (
        train_fe2["fe_isna_stres_skoru"] *
        train_fe2["gecelik_uyanma_sayisi"].fillna(0)
    )
    test_fe2["fe_stresNaN_x_uyanma"] = (
        test_fe2["fe_isna_stres_skoru"] *
        test_fe2["gecelik_uyanma_sayisi"].fillna(0)
    )

    train_fe2, test_fe2 = to_categorical(train_fe2, test_fe2)
    FEATURES_FLAG = get_feature_columns(train_fe2, ID_COL, TARGET)
    X_flag = train_fe2[FEATURES_FLAG]
    X_test_flag = test_fe2[FEATURES_FLAG]

    print(f"NaN-flag CB tuned egitiliyor ({len(FEATURES_FLAG)} features)...")
    result_flag = run_cv(X_flag, y, X_test_flag,
                        train_one_fold=train_one_fold_cb,
                        experiment_name="cb_nanflag",
                        fold_kwargs=fold_kwargs)

    cv_flag = result_flag["cv_rmse"]
    save_oof(result_flag["oof"], "cb_nanflag")
    save_submission(result_flag["test_preds"], "cb_nanflag", cv_rmse=cv_flag)


    # ============================================================
    # OZET
    # ============================================================
    print("\n" + "="*55)
    print("OZET")
    print("="*55)
    print(f"  Orijinal CB tuned (ref):       1.21600")
    print(f"  Yaklasim 1 (Iterative impute): {cv_imp:.5f}  ({cv_imp - 1.21600:+.5f})")
    print(f"  Yaklasim 2 (NaN-flag + intr):  {cv_flag:.5f}  ({cv_flag - 1.21600:+.5f})")

    # En iyi yaklasimla seed-avg stacking yapmaya deger mi?
    best = min([("imputed", cv_imp), ("nanflag", cv_flag)],
               key=lambda x: x[1])
    print(f"\n  En iyi yaklasim: {best[0]} ({best[1]:.5f})")
    if best[1] < 1.21600:
        print(f"  + KAZANC: orijinal CB'den {1.21600 - best[1]:.5f} iyi")
        print(f"  Bu yeni CB modelini eski LGB+XGB ile ensemble yapabiliriz.")
    else:
        print(f"  - KAZANC YOK: NaN-aware yontemler i\u015fe yaramadi.")


if __name__ == "__main__":
    main()
