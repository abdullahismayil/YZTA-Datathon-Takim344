"""
Otomatik Mean Shrinkage (Ortalamaya Cekme) Betigi
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    SUBMISSIONS_DIR = Path("outputs/submissions")
    
    # 1.20321 alan en iyi dosyamiz
    input_file = SUBMISSIONS_DIR / "sub_FINAL_BLEND_50_50_AUTO.csv"
    
    # Eger klasorde yoksa (isim degistiysa vs.), en iyi CV'li pseudo dosyasini yedek olarak al
    if not input_file.exists():
        input_file = SUBMISSIONS_DIR / "sub_ensemble_stacking_seedavg_pseudo_cv1.21352.csv"
        
    if not input_file.exists():
        print("[HATA] Kaynak dosya bulunamadi! 'outputs/submissions' icinde uygun dosya yok.")
        return

    output_file = SUBMISSIONS_DIR / "sub_mean_shrinkage_095_AUTO.csv"
    shrinkage_factor = 0.95

    print(f"[{input_file.name}] okunuyor...")
    df = pd.read_csv(input_file)
    
    # id haricindeki hedef sutunu bul
    target_col = [col for col in df.columns if col != 'id'][0]
    preds = df[target_col].values
    
    mean_val = np.mean(preds)
    print(f"Orijinal Ortalama: {mean_val:.4f}, Orijinal Std: {np.std(preds):.4f}")
    
    print(f"Shrinkage Factor ({shrinkage_factor}) uygulaniyor...")
    # Formül: (Tahmin - Ortalama) * Faktor + Ortalama
    shrunk_preds = (preds - mean_val) * shrinkage_factor + mean_val
    
    # 0-10 arasina kirp (guvenlik)
    final_preds = np.clip(shrunk_preds, 0.0, 10.0)
    
    print(f"Yeni Ortalama: {np.mean(final_preds):.4f}, Yeni Std: {np.std(final_preds):.4f}")
    
    df[target_col] = final_preds
    df.to_csv(output_file, index=False)
    print(f"\n[BASARILI] Yeni dosya kaydedildi: {output_file.name}")
    print(">> Hic beklemeden Kaggle'a yukleyebilirsin!")

if __name__ == "__main__":
    main()
