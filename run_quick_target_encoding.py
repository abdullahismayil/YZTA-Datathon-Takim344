import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from src.data import load_and_preprocess, to_categorical
from src.features import make_features_v1, get_feature_columns

def kfold_target_encoding(train_df, test_df, cat_cols, target_col, smoothing=20):
    train_encoded = np.zeros(len(train_df))
    train_df['mega_cat'] = train_df[cat_cols].fillna('MISSING').astype(str).apply(lambda row: '_'.join(row), axis=1)
    test_df['mega_cat'] = test_df[cat_cols].fillna('MISSING').astype(str).apply(lambda row: '_'.join(row), axis=1)
    global_mean = train_df[target_col].mean()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in kf.split(train_df):
        X_tr, X_va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        stats = X_tr.groupby('mega_cat')[target_col].agg(['mean', 'count'])
        smoothed_means = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
        train_encoded[va_idx] = X_va['mega_cat'].map(smoothed_means).fillna(global_mean)
    stats_full = train_df.groupby('mega_cat')[target_col].agg(['mean', 'count'])
    smoothed_means_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
    test_encoded = test_df['mega_cat'].map(smoothed_means_full).fillna(global_mean)
    return train_encoded, test_encoded.values

def main():
    print("Veri yukleniyor ve F1 (Baseline) FE yapiliyor...")
    train, test, _ = load_and_preprocess()
    train = make_features_v1(train)
    test = make_features_v1(test)
    cat_cols_to_combine = ['cinsiyet', 'meslek', 'ulke', 'ruh_sagligi_durumu', 'mevsim', 'gun_tipi']
    print("KFold-aware Target Encoding uygulaniyor (Smoothing=20)...")
    tr_enc, te_enc = kfold_target_encoding(train, test, cat_cols_to_combine, 'bilissel_performans_skoru')
    train['mega_cat_encoded'] = tr_enc
    test['mega_cat_encoded'] = te_enc
    # mega_cat string sutununu sil
    train.drop(columns=['mega_cat'], inplace=True)
    test.drop(columns=['mega_cat'], inplace=True)
    train, test = to_categorical(train, test)
    FEATURES = get_feature_columns(train, 'id', 'bilissel_performans_skoru')
    X = train[FEATURES]
    y = train['bilissel_performans_skoru']
    print(f"Toplam Feature Sayisi: {len(FEATURES)}. Hizli LightGBM CV basliyor...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))
    params = {
        'objective': 'regression', 'metric': 'rmse', 'random_state': 42,
        'learning_rate': 0.05, 'n_estimators': 1500, 'verbosity': -1
    }
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds[va_idx] = model.predict(X_va)
    cv_rmse = np.sqrt(mean_squared_error(y, oof_preds))
    print("\n" + "="*50)
    print(f"YENI TARGET ENCODING CV RMSE: {cv_rmse:.5f}")
    print("="*50)
    if cv_rmse < 1.21352:
        print(">>> KAZANC VAR!")
    else:
        print(">>> Kazanc Yok. Mevcut en iyi (CV 1.21352) korunacak.")

if __name__ == "__main__":
    main()
