"""
Final Asisi (Golden Blend) - Son Kursun
Stacking pürüzsüzlüğüne %10 oranında tekil CatBoost keskinliği ekler.
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    SUBMISSIONS_DIR = Path("outputs/submissions")
    
    # 1. En iyi Pseudo'lu Stacking
    pseudo_files = sorted(SUBMISSIONS_DIR.glob("*seedavg_pseudo*.csv"))
    # 2. En iyi Pseudo'suz Stacking
    seedavg_files = sorted(SUBMISSIONS_DIR.glob("*seedavg_cv*.csv"))
    # 3. En iyi tekil CatBoost dosyasini bul (Stacking veya Ensemble olmayan)
    cb_files = [f for f in SUBMISSIONS_DIR.glob("*cb*.csv") if "ensemble" not in f.name and "stacking" not in f.name]
    
    if not pseudo_files or not seedavg_files:
        print("[HATA] Gerekli ana harman dosyalari bulunamadi!")
        return
        
    f_pseudo = pseudo_files[-1]
    f_seedavg = seedavg_files[-1]
    
    print(f"Ana Dosya 1 (%45): {f_pseudo.name}")
    print(f"Ana Dosya 2 (%45): {f_seedavg.name}")
    
    df_pseudo = pd.read_csv(f_pseudo)
    df_seedavg = pd.read_csv(f_seedavg)
    
    # Hedef sutun adi
    target_col = [col for col in df_pseudo.columns if col != 'id'][0]
    
    # CatBoost asisi
    if cb_files:
        # Isme gore siralayip sonuncuyu al (genelde en iyi CV sondadir)
        f_cb = sorted(cb_files)[-1]
        print(f"CatBoost Asisi (%10): {f_cb.name}")
        df_cb = pd.read_csv(f_cb)
        
        final_preds = (df_pseudo[target_col] * 0.45) + \
                      (df_seedavg[target_col] * 0.45) + \
                      (df_cb[target_col] * 0.10)
    else:
        print("Uygun tekil CatBoost dosyasi bulunamadi. Ağırlıklar 60/40 olarak güncelleniyor...")
        final_preds = (df_pseudo[target_col] * 0.60) + \
                      (df_seedavg[target_col] * 0.40)
                      
    # Kirpma (Guvenlik)
    final_preds = np.clip(final_preds, 0.0, 10.0)
    
    df_submit = pd.DataFrame({
        'id': df_pseudo['id'],
        target_col: final_preds
    })
    
    out_name = SUBMISSIONS_DIR / "sub_FINAL_GOLDEN_INJECTION.csv"
    df_submit.to_csv(out_name, index=False)
    
    print(f"\n[BASARILI] Son kursun hazirlandi: {out_name.name}")
    print(">> Kaggle'a hemen yukle ve sonuca bakalim!")

if __name__ == "__main__":
    main()
