"""
Sentetik Formul Avcisi - Sembolik Regresyon (gplearn)
Eger veri gizli bir matematiksel formulle uretildiyse, bu betik o formulu bulmaya calisir.
"""
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from sklearn.metrics import mean_squared_error, r2_score
from src.data import load_and_preprocess
from src.features import get_feature_columns

def main():
    print("Veri yukleniyor...")
    train, _, _ = load_and_preprocess()
    
    # Sadece sayisal sutunlari al (kategoriklerle matematiksel islem zor)
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['id', 'bilissel_performans_skoru']]
    
    # Hizi artirmak icin rastgele 5000 satir sec (formul heryerde aynidir)
    train_sample = train.sample(n=5000, random_state=42).dropna(subset=numeric_cols)
    
    X = train_sample[numeric_cols]
    y = train_sample['bilissel_performans_skoru']
    
    print(f"X boyutu: {X.shape}. Sembolik Regresyon basliyor (bu islem 1-2 dakika surebilir)...")
    
    # Genetik algoritma ile matematiksel fonksiyonlari turet
    est_gp = SymbolicRegressor(population_size=2000,
                               generations=20,
                               stopping_criteria=0.01,
                               p_crossover=0.7,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.01,
                               random_state=42,
                               feature_names=numeric_cols)
    
    est_gp.fit(X, y)
    
    preds = est_gp.predict(X)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    
    print("\n" + "="*55)
    print(">>> KESFEDILEN MATEMATIKSEL FORMUL <<<")
    print("="*55)
    print(est_gp._program)
    print("="*55)
    print(f"Formulun R^2 Skoru  : {r2:.5f}")
    print(f"Formulun RMSE Skoru : {rmse:.5f}")
    
    if r2 > 0.85:
        print("\n[DIKKAT!] Inanilmaz yuksek R2! Veri kesinlikle bu formulle (veya cok benzeriyle) uretilmis!")
    else:
        print("\n[BILGI] R2 cok yuksek degil. Basit bir sentetik formul yok veya A Ihtimali (Sizinti/Bug) gecerli.")

if __name__ == "__main__":
    main()
