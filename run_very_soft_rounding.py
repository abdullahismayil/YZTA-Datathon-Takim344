import pandas as pd
import numpy as np
from pathlib import Path

def main():
    SUBMISSIONS_DIR = Path("outputs/submissions")
    
    # En iyi skorumuz (1.20321) olan dosyayi bul
    input_file = SUBMISSIONS_DIR / "sub_FINAL_BLEND_50_50_AUTO.csv"
    
    if not input_file.exists():
        input_file = SUBMISSIONS_DIR / "sub_ensemble_stacking_seedavg_pseudo_cv1.21352.csv"
        
    output_file = SUBMISSIONS_DIR / "sub_very_soft_round_a15.csv"
    alpha = 0.15

    print(f"Okunuyor: {input_file.name}")
    df = pd.read_csv(input_file)
    target_col = [col for col in df.columns if col != 'id'][0]
    preds = df[target_col].values
    
    print(f"Very Soft Rounding (Alpha={alpha}) uygulaniyor...")
    # Orijinal tahmini %85 koru, %15 tamsayiya dogru cek
    soft_rounded = (preds * (1 - alpha)) + (np.round(preds) * alpha)
    final_preds = np.clip(soft_rounded, 0.0, 10.0)
    
    df[target_col] = final_preds
    df.to_csv(output_file, index=False)
    
    print(f"[BASARILI] Kaydedildi: {output_file.name}")

if __name__ == "__main__":
    main()
