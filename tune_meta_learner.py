"""
Strateji C: Non-linear stacking with LightGBM meta-learner.

3 tuned modelin OOF tahminlerini feature olarak kullan,
LightGBM meta-learner ile non-linear kombinasyon kur.

Cikti: yeni submission + cv kiyaslama.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

from src.config import OOF_DIR, SUBMISSIONS_DIR, TARGET, ID_COL, SEED, N_FOLDS
from src.cv import rmse, save_submission
from src.data import load_and_preprocess


# 1) Veri
train, _, _ = load_and_preprocess()
y = train[TARGET].values

# 2) OOF ve test tahminlerini yukle
model_names = ["lgb_tuned_v1", "xgb_tuned_v1", "cb_tuned_v1"]
oof_list, test_list = [], []
for name in model_names:
    oof_list.append(np.load(OOF_DIR / f"oof_{name}.npy"))
    sub_files = sorted(SUBMISSIONS_DIR.glob(f"sub_{name}_cv*.csv"))
    df = pd.read_csv(sub_files[-1])
    target_col = [c for c in df.columns if c != ID_COL][0]
    test_list.append(df[target_col].values)

# Feature matrix
X_meta_train = np.column_stack(oof_list)  # (56000, 3)
X_meta_test = np.column_stack(test_list)  # (24000, 3)

print(f"Meta features: {X_meta_train.shape}")
print(f"Bireysel CV:")
for n, oof in zip(model_names, oof_list):
    print(f"  {n}: {rmse(y, oof):.5f}")

# 3) LightGBM meta-learner — cok sig + yuksek regularizasyon
meta_params = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 7,           # Cok sig
    "max_depth": 3,            # Cok sig
    "min_data_in_leaf": 200,   # Yuksek regularizasyon
    "feature_fraction": 1.0,   # Sadece 3 feature, hepsini kullan
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbosity": -1,
    "seed": SEED,
}

# 4) 5-fold CV ile meta-learner'i egit
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_meta = np.zeros(len(y))
test_meta = np.zeros(len(X_meta_test))

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_meta_train), 1):
    X_tr, X_va = X_meta_train[tr_idx], X_meta_train[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_set = lgb.Dataset(X_tr, y_tr)
    valid_set = lgb.Dataset(X_va, y_va, reference=train_set)

    model = lgb.train(
        meta_params, train_set, num_boost_round=2000,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred = model.predict(X_va, num_iteration=model.best_iteration)
    oof_meta[va_idx] = pred
    test_meta += model.predict(X_meta_test, num_iteration=model.best_iteration) / N_FOLDS

    fold_rmse = rmse(y_va, pred)
    print(f"  Fold {fold}: RMSE={fold_rmse:.5f} (best_iter={model.best_iteration})")

cv_meta = rmse(y, oof_meta)
print(f"\n>>> LGBM meta-learner CV: {cv_meta:.5f} <<<")
print(f">>> Ridge stacking referans: 1.21520 <<<")
print(f">>> Delta: {cv_meta - 1.21520:+.5f} <<<")

# 5) Kaydet
save_submission(test_meta, "ensemble_lgbm_meta", cv_rmse=cv_meta)

# 6) Karar
if cv_meta < 1.21520:
    print(f"\n+ KAZANC: LGBM meta {cv_meta:.5f} < Ridge 1.21520")
    print("Yeni submission kullanilabilir.")
else:
    print(f"\n- KAZANC YOK: LGBM meta {cv_meta:.5f} >= Ridge 1.21520")
    print("Ridge stacking daha iyi.")
