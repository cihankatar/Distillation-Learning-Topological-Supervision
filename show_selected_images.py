"""
Bu script, train_ssl_pretrained.py ve data_loader_ssl_pretrained.py kodlarının
belirli seed'ler (100, 200, 300 vb.) ve oranlar için hangi görselleri seçtiğini listeler.
"""

import os
import random
from glob import glob
from pathlib import Path

# Veri seti yolu
data_root = os.environ.get("ML_DATA_ROOT", "/Users/input/data/ckatar/")
dataset_name = "isic_2018_1"
train_im_dir = os.path.join(data_root, dataset_name, "train/images")
train_mask_dir = os.path.join(data_root, dataset_name, "train/masks")

train_images = sorted(glob(os.path.join(train_im_dir, "*.jpg")))
train_masks = sorted(glob(os.path.join(train_mask_dir, "*.png")))

if not train_images:
    print(f"HATA: '{train_im_dir}' dizininde görsel bulunamadı!")
    print("Lütfen veri setinin doğru konumda olduğundan emin olun.")
else:
    print("=" * 70)
    print(f"Veri Seti: {dataset_name} | Toplam Eğitim Görseli: {len(train_images)}")
    print("=" * 70)

    # 0.0025 oranı için 5 görsel
    ratio = 0.0025
    budget = 5

    for seed in [100, 200, 300]:
        combined = list(zip(train_images, train_masks))
        random.seed(seed)
        random.shuffle(combined)
        selected = combined[:budget]

        print(f"\n>>> SEED {seed} için Seçilen {budget} Görsel (ratio={ratio}):")
        print("-" * 55)
        for idx, (img_path, mask_path) in enumerate(selected, 1):
            img_name = os.path.basename(img_path)
            mask_name = os.path.basename(mask_path)
            print(f"  {idx}. {img_name:<18} (Maske: {mask_name})")
    print("\n" + "=" * 70)
