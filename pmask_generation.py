import os
import numpy as np
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from skimage.transform import resize

# Kendi importların
from utils.Dullrazor import dullrazor
from utils.padding import adaptive_pad
from utils.color_fields import build_gray_variants, lesion_is_high
from cubical_complex_main import cubical_complex_segmentation

def generate_and_save_cubical_masks(
    base_dir,
    splits=("train", "val", "test"),
    size=(256, 256),
    variants=("robust_delta_e_s_v",),
):
    """
    Belirtilen veri setindeki tüm resimler için Cubical Complex ile pseudo-mask 
    üretir ve ``pmasks_h0_spatial_cleanup`` klasörüne kaydeder. Varsayılan
    olarak yalnız makalede kullanılan nihai robust DeltaE/S/inverse-V alanı
    çalıştırılır.
    """
    to_tensor = T.ToTensor()

    for split in splits:
        print(f"\n--- İşleniyor: {split.upper()} ---")
        
        # Klasör yolları: Örn: /Users/input/data/ckatar/isic_2018_1/train/images
        image_dir = os.path.join(base_dir, split, "images")
        pmask_dir = os.path.join(base_dir, split, "pmasks_h0_spatial_cleanup")
        
        # pmasks klasörü yoksa oluştur
        os.makedirs(pmask_dir, exist_ok=True)
        
        all_images = [f for f in sorted(os.listdir(image_dir)) if f.endswith((".jpg", ".png", ".jpeg"))]
        
        with tqdm(total=len(all_images), desc=f"Generating {split} Cubical masks", unit="img") as pbar:
            for img_name in all_images:
                image_path = os.path.join(image_dir, img_name)
                
                # 1. Görüntüyü Yükle ve Tensor'a Çevir
                pil_img = Image.open(image_path).convert("RGB").resize(
                    size,
                    Image.Resampling.BILINEAR,
                )
                img_tensor = to_tensor(pil_img)
                
                # 2. Dullrazor (Kıl temizleme)
                clean_img = dullrazor(img_tensor)
                clean_img = clean_img.permute(1, 2, 0).cpu().numpy()
                
                # 3. Build and process every requested gray-level variant.
                gray_variants = build_gray_variants(clean_img)
                selected_variants = variants or ("robust_delta_e_s_v",)
                unknown = set(selected_variants) - set(gray_variants)
                if unknown:
                    raise ValueError(f"Unknown gray variants: {sorted(unknown)}")

                base_name = os.path.splitext(img_name)[0]
                save_name = f"{base_name}_pmasks.png"
                for variant_name in selected_variants:
                    gray = gray_variants[variant_name]
                    pad_size = 2
                    padded = adaptive_pad(gray, pad_size, 5)
                    input_image = resize(
                        padded,
                        (256, 256),
                        order=1,
                        preserve_range=True,
                        anti_aliasing=True,
                    ).astype(np.float32)

                    pseudo_mask, _, _, _, _ = cubical_complex_segmentation(
                        input_image,
                        h0_image=gray,
                        alpha=0.20,
                        lesion_high=lesion_is_high(variant_name),
                    )
                    pseudo_mask = (pseudo_mask > 0).astype(np.uint8) * 255

                    variant_dir = os.path.join(pmask_dir, variant_name)
                    os.makedirs(variant_dir, exist_ok=True)
                    Image.fromarray(pseudo_mask).save(
                        os.path.join(variant_dir, save_name)
                    )
                
                pbar.update(1)

if __name__ == "__main__":
    dataset = os.environ.get("PSEUDO_MASK_DATASET", "isic_2018_1")
    
    # Kendi dizin yapın
    root_path = os.environ.get("ML_DATA_ROOT", "/Users/input/data/ckatar/")
    base = os.path.join(root_path, dataset)
    
    # Train, test ve val için çalıştır
    generate_and_save_cubical_masks(base, splits=['train', 'val', 'test'])
