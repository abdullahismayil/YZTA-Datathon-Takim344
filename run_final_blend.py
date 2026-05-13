"""
Otomatik Final Blend (Harmanlama) + Kirpma (Post-Processing)
"""
import pandas as pd
import numpy as np
from pathlib import Path

def main():
    # Dizin ayarlari
    SUBMISSIONS_DIR = Path("outputs/submissions")
    
    # 1. En iyi LB (1.20328) dosyamiz:
    best_lb_path = SUBMISSIONS_DIR / "sub_ensemble_stacking_seedavg_cv1.21412.csv"
    
    # 2. En iyi CV (1.21352) dosyamiz:
    best_cv_path = SUBMISSIONS_DIR / "sub_ensemble_stacking_seedavg_pseudo_cv1.21352.csv"

    print("Dosyalar yukleniyor...")
    if not best_lb_path.exists():
        print(f"[HATA] {best_lb_path} bulunamadi!")
        return
    if not best_cv_path.exists():
        print(f"[HATA] {best_cv_path} bulunamadi!")
        return

    df_lb = pd.read_csv(best_lb_path)
    df_cv = pd.read_csv(best_cv_path)

    assert all(df_lb['id'] == df_cv['id']), "ID'ler eslesmiyor!"

    print("Blend yapiliyor (Agirliklar: %50 En Iyi LB, %50 En Iyi CV)...")
    blended_preds = (df_lb['bilissel_performans_skoru'] * 0.50) + \
                    (df_cv['bilissel_performans_skoru'] * 0.50)

    print("Post-processing (0.0 - 10.0 arasi clipping) uygulaniyor...")
    final_preds = np.clip(blended_preds, 0.0, 10.0)
    out_of_bounds_count = sum((blended_preds < 0.0) | (blended_preds > 10.0))

    df_submit = pd.DataFrame({
        'id': df_lb['id'],
        'bilissel_performans_skoru': final_preds
    })

    out_name = SUBMISSIONS_DIR / "sub_FINAL_BLEND_50_50_AUTO.csv"
    df_submit.to_csv(out_name, index=False)

    print(f"\n{'='*55}\nOZET\n{'='*55}")
    print(f"  En iyi LB dosyasi: {best_lb_path.name}")
    print(f"  En iyi CV dosyasi: {best_cv_path.name}")
    print(f"  Kirpilan (clipped) satir sayisi: {out_of_bounds_count}")
    print(f"  >> KAYDEDILDI: {out_name}")
    print("\n  >>> HIC BEKLEMEDEN KAGGLE'A YUKLE! <<<")

if __name__ == "__main__":
    main()
