"""
Siniflandirma + Regresyon Hibrit Modeli (Classification + Residual Regression)
Teori: Target = Tamsayi_Sinifi + Gurultu
"""
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from src.data import load_and_preprocess, to_categorical
from src.features import make_features_v1, get_feature_columns
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Veri yukleniyor...")
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    train, test = to_categorical(train, test)
    FEATURES = get_feature_columns(train, 'id', 'bilissel_performans_skoru')
    X = train[FEATURES]
    y = train['bilissel_performans_skoru']
    X_test = test[FEATURES]
    y_class = np.clip(np.round(y).astype(int), 0, 10)
    print("\n--- ASAMA 1: SINIFLANDIRMA (Multi-Class) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_class_probs = np.zeros((len(X), 11))
    test_class_probs = np.zeros((len(X_test), 11))
    clf_params = {
        'objective': 'multiclass', 'num_class': 11, 'metric': 'multi_logloss',
        'random_state': 42, 'learning_rate': 0.05, 'n_estimators': 1000, 'verbosity': -1
    }
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y_class)):
        X_tr, y_c_tr = X.iloc[tr_idx], y_class.iloc[tr_idx]
        X_va, y_c_va = X.iloc[va_idx], y_class.iloc[va_idx]
        clf = LGBMClassifier(**clf_params)
        clf.fit(X_tr, y_c_tr, eval_set=[(X_va, y_c_va)], callbacks=[])
        oof_class_probs[va_idx] = clf.predict_proba(X_va)
        test_class_probs += clf.predict_proba(X_test) / 5
        print(f"  Fold {fold+1} Classifier egitildi.")
    classes = np.arange(11)
    oof_ev = np.sum(oof_class_probs * classes, axis=1)
    test_ev = np.sum(test_class_probs * classes, axis=1)
    print("\n--- ASAMA 2: REGRESYON (Residual Tahmini) ---")
    y_residual = y - oof_ev
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_res = np.zeros(len(X))
    test_res = np.zeros(len(X_test))
    reg_params = {
        'objective': 'regression', 'metric': 'rmse', 'random_state': 42,
        'learning_rate': 0.05, 'n_estimators': 1000, 'verbosity': -1
    }
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, y_r_tr = X.iloc[tr_idx], y_residual.iloc[tr_idx]
        X_va, y_r_va = X.iloc[va_idx], y_residual.iloc[va_idx]
        X_tr_aug = X_tr.copy()
        X_tr_aug['expected_value'] = oof_ev[tr_idx]
        X_va_aug = X_va.copy()
        X_va_aug['expected_value'] = oof_ev[va_idx]
        reg = LGBMRegressor(**reg_params)
        reg.fit(X_tr_aug, y_r_tr, eval_set=[(X_va_aug, y_r_va)], callbacks=[])
        oof_res[va_idx] = reg.predict(X_va_aug)
        X_test_aug = X_test.copy()
        X_test_aug['expected_value'] = test_ev
        test_res += reg.predict(X_test_aug) / 5
        print(f"  Fold {fold+1} Regressor egitildi.")
    print("\n--- ASAMA 3: FINAL BIRLESTIRME ---")
    final_oof = np.clip(oof_ev + oof_res, 0.0, 10.0)
    final_test = np.clip(test_ev + test_res, 0.0, 10.0)
    cv_rmse = np.sqrt(mean_squared_error(y, final_oof))
    print("="*50)
    print(f"HYBRID MODEL CV RMSE: {cv_rmse:.5f}")
    print("="*50)
    df_sub = pd.DataFrame({'id': test['id'], 'bilissel_performans_skoru': final_test})
    out_name = f"outputs/submissions/sub_hybrid_cv{cv_rmse:.5f}.csv"
    df_sub.to_csv(out_name, index=False)
    print(f"Dosya kaydedildi: {out_name}")

if __name__ == "__main__":
    main()
