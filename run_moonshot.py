import pandas as pd
import numpy as np
from pathlib import Path

def main():
    SUBMISSIONS_DIR = Path("outputs/submissions")
    
    # 1. En iyi Pseudo Stacking (1.20321)
    f1 = SUBMISSIONS_DIR / "sub_FINAL_BLEND_50_50_AUTO.csv"
    # 2. En iyi Safe Stacking (1.20328)
    f2 = SUBMISSIONS_DIR / "sub_ensemble_stacking_seedavg_cv1.21412.csv"
    # 3. En iyi tekil CatBoost (Varyans katmak icin)
    f3_list = sorted(SUBMISSIONS_DIR.glob("*cb_pseudo*.csv"))
    
    if not f1.exists() or not f2.exists() or not f3_list:
        print("[HATA] Gerekli dosyalar bulunamadi!")
        return
        
    f3 = f3_list[-1]
    
    print(f"Bilesen 1 (%40): {f1.name}")
    print(f"Bilesen 2 (%40): {f2.name}")
    print(f"Bilesen 3 (%20): {f3.name}")
    
    df1 = pd.read_csv(f1)
    df2 = pd.read_csv(f2)
    df3 = pd.read_csv(f3)
    
    target_col = [col for col in df1.columns if col != 'id'][0]
    
    # TRIPLE BLEND FORMULU
    final_preds = (df1[target_col] * 0.40) + \
                  (df2[target_col] * 0.40) + \
                  (df3[target_col] * 0.20)
    
    final_preds = np.clip(final_preds, 0.0, 10.0)
    
    df_out = df1.copy()
    df_out[target_col] = final_preds
    
    out_file = SUBMISSIONS_DIR / "sub_FINAL_MOONSHOT_TRIPLE.csv"
    df_out.to_csv(out_file, index=False)
    
    print(f"\n[BASARILI] Son kursun 'sub_FINAL_MOONSHOT_TRIPLE.csv' olarak hazirlandi!")
    print(">> Kaggle'a yukle ve sonucu bekle.")

if __name__ == "__main__":
    main()
