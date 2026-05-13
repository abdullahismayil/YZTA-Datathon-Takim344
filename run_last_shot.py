import pandas as pd
import numpy as np
from pathlib import Path

def main():
    SUBMISSIONS_DIR = Path("outputs/submissions")
    
    # Dosyalari bul (Isimler degisiklik gosterebilir, o yuzden en iyi ihtimalleri ariyoruz)
    best_agresif = SUBMISSIONS_DIR / "sub_FINAL_BLEND_50_50_AUTO.csv"
    best_defansif = SUBMISSIONS_DIR / "sub_ensemble_stacking_seedavg_cv1.21412.csv"
    
    if not best_agresif.exists():
        best_agresif = SUBMISSIONS_DIR / "sub_ensemble_stacking_seedavg_pseudo_cv1.21352.csv"
        
    print(f"Bilesen 1 (%60 Agresif) : {best_agresif.name}")
    print(f"Bilesen 2 (%40 Defansif): {best_defansif.name}")
    
    df1 = pd.read_csv(best_agresif)
    df2 = pd.read_csv(best_defansif)
    
    target_col = [col for col in df1.columns if col != 'id'][0]
    
    # %60 - %40 Altin Oran
    final_preds = (df1[target_col] * 0.60) + (df2[target_col] * 0.40)
    
    # Sınırlandırma
    final_preds = np.clip(final_preds, 0.0, 10.0)
    
    df_out = df1.copy()
    df_out[target_col] = final_preds
    
    out_file = SUBMISSIONS_DIR / "sub_FINAL_LAST_SHOT_60_40.csv"
    df_out.to_csv(out_file, index=False)
    
    print(f"\n[BASARILI] Son kursun hazirlandi: {out_file.name}")

if __name__ == "__main__":
    main()
