import argparse
import os
import sys
import time
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import color, morphology as morph
from skimage.filters import threshold_otsu
from skimage.measure import euler_number, label
from skimage.transform import resize
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

# Custom Utils imports
from utils.Dullrazor import dullrazor
from utils.iou_dice import iou_and_dice
from utils.padding import adaptive_pad
from utils.color_fields import build_gray_variants, lesion_is_high
from utils.Algorithms import (
    morphological_chan_vese_segmentation,
    random_walker_pseudo_mask,
    chc_otsu_pseudo_mask,
    grabcut_pseudo_mask,
    watershed_pseudo_mask,
)
from cubical_complex_main import cubical_complex_segmentation, resolve_ml_data_root

DATASET_ALIASES = {
    "isic2018": "isic_2018_1",
    "isic_2018": "isic_2018_1",
    "isic_2018_1": "isic_2018_1",
    "ph2": "PH2Dataset",
    "ph2dataset": "PH2Dataset",
    "isic2016": "isic_2016_1",
    
    "isic_2016": "isic_2016_1",
    "isic_2016_1": "isic_2016_1",
}

METRIC_NAMES = (
    "iou",
    "dice",
    "beta0_error",
    "beta1_error",
    "betti_error",
    "betti_correct",
)


def empty_metric_store():
    return {metric_name: [] for metric_name in METRIC_NAMES}


def to_binary_mask(mask):
    """Convert bool, 0/1, probability, or 0/255 masks to a 2-D bool array."""
    mask = np.asarray(mask)
    mask = np.squeeze(mask)

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {mask.shape}.")

    if mask.dtype == np.bool_:
        return mask

    finite_values = mask[np.isfinite(mask)]
    if finite_values.size == 0:
        return np.zeros(mask.shape, dtype=bool)

    threshold = 0.5 if finite_values.max() <= 1.0 else 127.5
    return mask > threshold


def to_shared_uint8_scalar(field):
    """Convert the common [0, 255] scalar field once for every baseline."""
    scalar = np.asarray(field, dtype=np.float32)
    if scalar.ndim != 2:
        raise ValueError(f"Expected a 2-D scalar field, got {scalar.shape}.")
    return np.rint(np.clip(scalar, 0.0, 255.0)).astype(np.uint8)


def betti_numbers_2d(mask, connectivity=2):
    """
    Return (beta_0, beta_1) for a 2-D binary mask.
    connectivity=2 uses 8-connectivity for the foreground and complementary
    4-connectivity convention for the background.
    """
    binary_mask = to_binary_mask(mask)
    _, beta_0 = label(
        binary_mask.astype(np.uint8),
        background=0,
        connectivity=connectivity,
        return_num=True,
    )
    euler_characteristic = euler_number(
        binary_mask,
        connectivity=connectivity,
    )
    beta_1 = int(beta_0 - euler_characteristic)
    return int(beta_0), beta_1


def calculate_mask_metrics(pred_mask, real_mask, real_betti=None):
    """Calculate overlap and Betti-number metrics for one prediction."""
    pred_binary = to_binary_mask(pred_mask)
    real_binary = to_binary_mask(real_mask)

    iou, dice = iou_and_dice(pred_binary, real_binary)
    pred_beta_0, pred_beta_1 = betti_numbers_2d(pred_binary)

    if real_betti is None:
        real_betti = betti_numbers_2d(real_binary)
    real_beta_0, real_beta_1 = real_betti

    beta0_error = abs(pred_beta_0 - real_beta_0)
    beta1_error = abs(pred_beta_1 - real_beta_1)
    betti_error = beta0_error + beta1_error

    return {
        "iou": float(iou),
        "dice": float(dice),
        "beta0_error": int(beta0_error),
        "beta1_error": int(beta1_error),
        "betti_error": int(betti_error),
        "betti_correct": int(betti_error == 0),
    }


def append_metrics(results, algorithm_name, metrics):
    for metric_name in METRIC_NAMES:
        results[algorithm_name][metric_name].append(metrics[metric_name])


def evaluate_dataset(
    image_dir,
    mask_dir,
    size=(256, 256),
    sample_size=None,
    low_score_threshold=0.20,
    variant="robust_delta_e_s_v",
    evaluate_baselines=True,
    save_topological_masks=False,
    report_path=None,
):
    """
    Evaluates pseudo-mask extraction methods on the given dataset split.
    Every classical baseline receives the same selected scalar field.  The only
    topological setting is the final alpha=0.20 pipeline.  Random Walker,
    Morphological Chan--Vese, CHC-Otsu, GrabCut, and Watershed are evaluated as
    baselines.  When ``save_topological_masks`` is True, only the alpha=0.20
    topological masks are saved; baseline masks are never written to disk.
    """
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.exists():
        print(f"[UYARI] Görüntü dizini bulunamadı: {image_dir}")
        return None

    to_tensor = T.ToTensor()
    image_exts = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_images = sorted([f for f in os.listdir(image_dir) if Path(f).suffix.lower() in image_exts])
    all_masks = sorted([f for f in os.listdir(mask_dir) if Path(f).suffix.lower() in image_exts]) if mask_dir.exists() else []

    if not all_images:
        print(f"[UYARI] {image_dir} içinde geçerli görsel bulunamadı.")
        return None

    # Hedef pmasks klasörü
    base_split_dir = image_dir.parent
    # Example: <ML_DATA_ROOT>/isic_2018_1/train/pmasks
    pmask_dir = base_split_dir / "pmasks"
    if save_topological_masks:
        os.makedirs(pmask_dir, exist_ok=True)

    if sample_size is not None and len(all_images) > sample_size:
        indices = sorted(random.sample(range(len(all_images)), sample_size))
    else:
        indices = list(range(len(all_images)))

    images_list = [all_images[i] for i in indices]

    # Metot anahtarları: ortak skaler alanlı klasik yöntemler + nihai topoloji.
    method_keys = []
    if evaluate_baselines:
        method_keys.extend([
            "random_walker",
            "morphological",
            "chc_otsu",
            "grabcut",
            "watershed",
        ])
    method_keys.append("topological_alpha_0.20")

    results = {m: empty_metric_store() for m in method_keys}
    timers = {m: 0.0 for m in method_keys}

    low_score_images = []
    method_failures = []
    missing_ground_truth_images = []
    ground_truth_betti_distribution = Counter()
    evaluated_ground_truth_count = 0

    def record_method_failure(method_name, image_name, exception):
        failure = (
            f"{method_name} | {image_name} | "
            f"{type(exception).__name__}: {exception}"
        )
        method_failures.append(failure)
        print(f"[UYARI] {failure}")

    split_name = base_split_dir.name.upper()
    print(f"\n================================================================================")
    print(f"BAŞLATILIYOR: [{split_name}] Fazı | Toplam {len(images_list)} Görsel İncelenecek")
    print(f"Konfigürasyon: Tüm yöntemlerde ortak skaler alan={variant} | Topoloji Alfa=0.20")
    print(f"Yalnız topolojik maske kaydı: {save_topological_masks} | Hedef={pmask_dir}")
    print(f"================================================================================\n")

    # Continuous tqdm timing is disabled. Progress and average runtimes are
    # printed only every 50 images and once at the end of the split.
    with tqdm(
        total=len(images_list),
        desc=f"Evaluating [{split_name}]",
        unit="img",
        disable=True,
    ) as pbar:
        for i, (orig_idx, img_name) in enumerate(zip(indices, images_list)):
            image_path = image_dir / img_name

            # Maske eşleştirmesi
            stem = Path(img_name).stem
            mask_candidate = mask_dir / img_name
            if not mask_candidate.exists():
                for sfx in ["_segmentation.png", "_mask.png", "_pmasks.png"]:
                    cand = mask_dir / f"{stem}{sfx}"
                    if cand.exists():
                        mask_candidate = cand
                        break

            has_ground_truth = mask_candidate.exists()
            if has_ground_truth:
                evaluated_ground_truth_count += 1
                real_mask_pil = Image.open(mask_candidate).convert("L").resize(size, Image.Resampling.NEAREST)
                real_mask_binary = to_binary_mask(np.array(real_mask_pil))
                real_betti = betti_numbers_2d(real_mask_binary)
                ground_truth_betti_distribution[real_betti] += 1
            else:
                real_mask_binary = None
                real_betti = None
                missing_ground_truth_images.append(img_name)

            # Görseli oku ve DullRazor ile kılları temizle
            pil_img = Image.open(image_path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
            img_tensor = to_tensor(pil_img)
            clean_img = dullrazor(img_tensor).permute(1, 2, 0).cpu().numpy()

            # Skalar alan oluşturma (robust_delta_e_s_v)
            gray_variants = build_gray_variants(clean_img)
            gray = gray_variants[variant]
            shared_scalar = to_shared_uint8_scalar(gray)
            shared_lesion_high = lesion_is_high(variant)

            # Adaptive padding ve 256x256 yeniden boyutlandırma
            pad_size = 2
            padded = adaptive_pad(gray, pad_size, 5)
            h1_input = resize(padded, size, order=1, preserve_range=True, anti_aliasing=True).astype(gray.dtype)

            image_results = []

            # ==========================================================
            # 1. BASİT TEMEL METODLAR (Karşılaştırmalar)
            # ==========================================================
            if evaluate_baselines and has_ground_truth:
                # Random Walker
                try:
                    t0 = time.time()
                    rw_mask = random_walker_pseudo_mask(
                        shared_scalar,
                        lesion_high=shared_lesion_high,
                    )
                    timers["random_walker"] += time.time() - t0
                    rw_m = calculate_mask_metrics(rw_mask, real_mask_binary, real_betti)
                    append_metrics(results, "random_walker", rw_m)
                    image_results.append(("Random Walker", rw_m))
                except Exception as exc:
                    record_method_failure("Random Walker", img_name, exc)

                # Morphological Chan-Vese
                try:
                    t0 = time.time()
                    mcv_mask, _ = morphological_chan_vese_segmentation(shared_scalar)
                    timers["morphological"] += time.time() - t0
                    mcv_m = calculate_mask_metrics(mcv_mask, real_mask_binary, real_betti)
                    append_metrics(results, "morphological", mcv_m)
                    image_results.append(("Morph. Chan-Vese", mcv_m))
                except Exception as exc:
                    record_method_failure("Morphological Chan-Vese", img_name, exc)

                # CHC-Otsu
                try:
                    t0 = time.time()
                    chc_mask, _ = chc_otsu_pseudo_mask(shared_scalar)
                    timers["chc_otsu"] += time.time() - t0
                    chc_m = calculate_mask_metrics(chc_mask, real_mask_binary, real_betti)
                    append_metrics(results, "chc_otsu", chc_m)
                    image_results.append(("CHC-Otsu", chc_m))
                except Exception as exc:
                    record_method_failure("CHC-Otsu", img_name, exc)

                # GrabCut
                try:
                    t0 = time.time()
                    gc_mask = grabcut_pseudo_mask(shared_scalar)
                    timers["grabcut"] += time.time() - t0
                    gc_m = calculate_mask_metrics(gc_mask, real_mask_binary, real_betti)
                    append_metrics(results, "grabcut", gc_m)
                    image_results.append(("GrabCut", gc_m))
                except Exception as exc:
                    record_method_failure("GrabCut", img_name, exc)

                # Marker-controlled Watershed
                try:
                    t0 = time.time()
                    watershed_mask = watershed_pseudo_mask(
                        shared_scalar,
                        lesion_high=shared_lesion_high,
                    )
                    timers["watershed"] += time.time() - t0
                    watershed_m = calculate_mask_metrics(
                        watershed_mask,
                        real_mask_binary,
                        real_betti,
                    )
                    append_metrics(results, "watershed", watershed_m)
                    image_results.append(("Watershed", watershed_m))
                except Exception as exc:
                    record_method_failure("Watershed", img_name, exc)

            # ==========================================================
            # 2. NİHAİ TOPOLOJİK SEGMENTASYON (ALFA=0.20)
            # ==========================================================
            alpha_val = 0.20
            key_name = "topological_alpha_0.20"
            try:
                t0 = time.time()
                cc_res = cubical_complex_segmentation(
                    h1_input,
                    h0_image=gray,
                    persistence_threshold=None,
                    alpha=alpha_val,
                    lesion_high=shared_lesion_high,
                    return_debug=False,
                )
                final_mask = cc_res[0]
                timers[key_name] += time.time() - t0

                if save_topological_masks:
                    save_mask = (final_mask > 0).astype(np.uint8) * 255
                    base_name = Path(img_name).stem
                    for sfx in ["_segmentation", "_mask", "_pmasks"]:
                        if base_name.endswith(sfx):
                            base_name = base_name[:-len(sfx)]
                    save_path = pmask_dir / f"{base_name}_pmasks.png"
                    Image.fromarray(save_mask).save(save_path)

                if has_ground_truth:
                    t_metrics = calculate_mask_metrics(
                        final_mask,
                        real_mask_binary,
                        real_betti,
                    )
                    append_metrics(results, key_name, t_metrics)
                    if t_metrics["iou"] < low_score_threshold:
                        low_score_images.append(
                            (
                                orig_idx,
                                img_name,
                                t_metrics["iou"],
                                t_metrics["dice"],
                            )
                        )
            except Exception as exc:
                record_method_failure("Topological alpha=0.20", img_name, exc)

            # ----------------------------------------------------------
            # HER 50 GÖRSELDE BİR ARA RAPOR
            # ----------------------------------------------------------
            if (i + 1) % 50 == 0:
                print("\n" + "=" * 80)
                print(f"ARA RAPOR: {i + 1}/{len(images_list)} Görsel İşlendi - [{split_name}]")
                print("=" * 80)
                for alg_key in method_keys:
                    m_list = results[alg_key]
                    if len(m_list["iou"]) > 0:
                        cur_iou = np.mean(m_list["iou"])
                        cur_dice = np.mean(m_list["dice"])
                        cur_b0_err = np.mean(m_list["beta0_error"])
                        cur_b1_err = np.mean(m_list["beta1_error"])
                        cur_corr = 100.0 * np.mean(m_list["betti_correct"])
                        success_count = len(m_list["iou"])
                        avg_t = timers[alg_key] / max(success_count, 1)
                        print(
                            f"  {alg_key:<26} | "
                            f"mIoU: {cur_iou:.4f} | mDice: {cur_dice:.4f} | "
                            f"b0: {cur_b0_err:.3f} | b1: {cur_b1_err:.3f} | "
                            f"TopoCorr: {cur_corr:5.1f}% | Süre: {avg_t:.3f} s/img"
                        )
                print("=" * 80 + "\n")

            pbar.update(1)

    # ==========================================================
    # SPLIT SONU GENEL RAPORLAMA VE TABLO 1 ÇIKTISI
    # ==========================================================
    summary_lines = []
    for alg_key in method_keys:
        m_store = results[alg_key]
        success_count = len(m_store["iou"])
        if success_count == 0:
            summary_lines.append(
                f"{alg_key:<26} | Başarılı ölçüm yok"
            )
            continue

        m_iou = np.mean(m_store["iou"])
        m_dice = np.mean(m_store["dice"])
        m_b0 = np.mean(m_store["beta0_error"])
        m_b1 = np.mean(m_store["beta1_error"])
        correct_pct = 100.0 * np.mean(m_store["betti_correct"])
        avg_time = timers[alg_key] / success_count
        summary_lines.append(
            f"{alg_key:<26} | n: {success_count:4d} | "
            f"mIoU: {m_iou:.4f} | mDice: {m_dice:.4f} | "
            f"b0: {m_b0:.3f} | b1: {m_b1:.3f} | "
            f"TopoCorr: {correct_pct:5.1f}% | Süre: {avg_time:.3f} s/img"
        )

    latex_rows = [
        ("random_walker", "Random Walker\\citep{grady2006random}"),
        ("morphological", "Morphological Chan--Vese\\citep{marquez2014morphological}"),
        ("chc_otsu", "CHC-Otsu\\citep{joseph2022preprocessing}"),
        ("grabcut", "GrabCut\\citep{rother2004grabcut}"),
        ("watershed", "Watershed\\citep{vincent1991watersheds}"),
        ("hdashline", "\\hdashline"),
        ("topological_alpha_0.20", "Topological Segmentation ($\\alpha = 0.20$)"),
    ]
    latex_output_lines = []
    for key_code, tex_label in latex_rows:
        if key_code == "hdashline":
            latex_output_lines.append("\\hdashline")
            continue
        m_store = results.get(key_code)
        if m_store is None or len(m_store["iou"]) == 0:
            continue
        success_count = len(m_store["iou"])
        avg_t = timers[key_code] / success_count
        latex_output_lines.append(
            f"{tex_label} & "
            f"{np.mean(m_store['iou']):.4f} & "
            f"{np.mean(m_store['dice']):.4f} & "
            f"{np.mean(m_store['beta0_error']):.4f} & "
            f"{np.mean(m_store['beta1_error']):.4f} & "
            f"{100.0 * np.mean(m_store['betti_correct']):.1f}\\% & "
            f"{avg_t:.3f} \\\\"
        )

    print("\n" + "#" * 90)
    print(
        f"[{split_name}] SONUÇLARI "
        f"(Görsel: {len(images_list)}, GT: {evaluated_ground_truth_count})"
    )
    print("#" * 90)
    for line in summary_lines:
        print(line)
    print(
        f"Başarısız yöntem çalışması: {len(method_failures)} | "
        f"Düşük topolojik skor: {len(low_score_images)} | "
        f"Eksik gerçek maske: {len(missing_ground_truth_images)}"
    )

    report_lines = [
        "=" * 110,
        f"SPLIT: {split_name}",
        f"Görüntü dizini: {image_dir}",
        f"Toplam görüntü: {len(images_list)}",
        f"Gerçek maskesi bulunan: {evaluated_ground_truth_count}",
        f"Skaler alan: {variant}",
        f"Topolojik alpha: 0.20",
        f"Topolojik maske kaydı: {save_topological_masks}",
        f"Topolojik maske klasörü: {pmask_dir}",
        "",
        "NİHAİ YÖNTEM SONUÇLARI",
        "-" * 110,
        *summary_lines,
        "",
        "GERÇEK MASKE BETTI DAĞILIMI (beta0, beta1 -> adet)",
        "-" * 110,
    ]
    if ground_truth_betti_distribution:
        report_lines.extend(
            f"{betti_pair} -> {count}"
            for betti_pair, count in sorted(
                ground_truth_betti_distribution.items()
            )
        )
    else:
        report_lines.append("Gerçek maske bulunmadı.")

    report_lines.extend([
        "",
        f"BAŞARISIZ YÖNTEM ÇALIŞMALARI ({len(method_failures)})",
        "-" * 110,
    ])
    report_lines.extend(method_failures or ["Yok."])

    report_lines.extend([
        "",
        (
            f"DÜŞÜK SKORLU TOPOLOJİK MASKELER "
            f"(IoU < {low_score_threshold:.2f}) ({len(low_score_images)})"
        ),
        "-" * 110,
    ])
    if low_score_images:
        report_lines.extend(
            f"[{idx_val:04d}] {name_val} | "
            f"IoU={iou_val:.4f} | Dice={dice_val:.4f}"
            for idx_val, name_val, iou_val, dice_val in low_score_images
        )
    else:
        report_lines.append("Yok.")

    report_lines.extend([
        "",
        f"EKSİK GERÇEK MASKELER ({len(missing_ground_truth_images)})",
        "-" * 110,
        *(missing_ground_truth_images or ["Yok."]),
        "",
        "TABLO 1 İÇİN LATEX SATIRLARI",
        "-" * 110,
        *(latex_output_lines or ["Ölçülebilir sonuç yok."]),
        "",
    ])

    if report_path is not None:
        paths_to_write = report_path if isinstance(report_path, (list, tuple)) else [report_path]
        for p in paths_to_write:
            p = Path(p)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as report_file:
                report_file.write("\n".join(report_lines) + "\n")
            print(f"Rapor güncellendi: {p}")

    return results, low_score_images, timers


def ask_dataset_name():
    """Ask which dataset should be evaluated and receive pseudo masks."""
    raw_name = input(
        "Veri setini girin [isic2018 / ph2 / isic2016] "
        "(varsayılan: isic2018): "
    ).strip()
    if not raw_name:
        raw_name = "isic2018"

    dataset_key = raw_name.lower()
    if dataset_key not in DATASET_ALIASES:
        raise ValueError(
            f"Bilinmeyen veri seti: {raw_name!r}. "
            "Geçerli seçenekler: isic2018, ph2, isic2016."
        )
    return DATASET_ALIASES[dataset_key]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pseudo-mask methods and optionally save only the final "
            "alpha=0.20 topological masks."
        )
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="isic2018, ph2, isic2016, or the corresponding directory name",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Dataset parent directory (defaults to ML_DATA_ROOT)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Evaluate a random subset instead of the full split",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=0.20,
        help="IoU threshold used to list failed/low-quality masks",
    )
    parser.add_argument(
        "--save-topological-masks",
        action="store_true",
        help="Write final masks to <dataset>/<split>/pmasks",
    )
    parser.add_argument(
        "--topology-only",
        action="store_true",
        help="Skip the five classical baselines (useful for mask generation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=932,
        help="Seed used only when --sample-size is set",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    if args.dataset is None:
        dataset = ask_dataset_name()
    else:
        dataset_key = args.dataset.lower()
        if dataset_key not in DATASET_ALIASES:
            raise ValueError(
                f"Bilinmeyen veri seti: {args.dataset!r}. "
                "Geçerli seçenekler: isic2018, ph2, isic2016."
            )
        dataset = DATASET_ALIASES[dataset_key]

    data_root = args.data_root or Path(resolve_ml_data_root())
    base = os.path.join(data_root, dataset)
    if not Path(base).is_dir():
        raise FileNotFoundError(f"Veri seti klasörü bulunamadı: {base}")

    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    file_ts = time.strftime("%Y%m%d_%H%M%S")

    # 1. Ana kümülatif rapor dosyası (ESKİ RAPORLAR ASLA SİLİNMEZ, 'a' modunda eklenir)
    main_report_path = Path(base) / "test_new_results.txt"
    # 2. Bu çalıştırmaya özel bağımsız rapor dosyası (Zaman damgalı, daima korunur)
    session_report_path = Path(base) / f"test_new_results_{file_ts}.txt"

    header = (
        f"\n{'='*110}\n"
        f"PSEUDO-MASK DEĞERLENDİRME RAPORU - [{timestamp_str}]\n"
        f"Veri seti: {dataset}\n"
        f"Kök klasör: {base}\n"
        "Ortak skaler alan: robust_delta_e_s_v\n"
        "Topolojik alpha: 0.20\n"
        f"{'='*110}\n\n"
    )

    for p in [main_report_path, session_report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(header)

    print(f"Seçilen veri seti: {dataset}")
    print(f"Veri seti yolu: {base}")
    print(f"Ana rapor (Append):       {main_report_path}")
    print(f"Oturum raporu (Müstakil): {session_report_path}")
    
    # Her split ayrı rapor bölümü olarak yazılır.
    for split in args.splits:
        print(f"\n==================== STARTING PHASE: {split.upper()} ====================")
        im_path = os.path.join(base, f"{split}/images")
        masks_path = os.path.join(base, f"{split}/masks")

        evaluate_dataset(
            im_path,
            masks_path,
            sample_size=args.sample_size,
            low_score_threshold=args.low_score_threshold,
            evaluate_baselines=not args.topology_only,
            save_topological_masks=args.save_topological_masks,
            report_path=[main_report_path, session_report_path],
        )
        print(f"==================== COMPLETED PHASE: {split.upper()} ====================\n")


if __name__ == "__main__":
    main()
