"""
Evaluate TopoDistill downstream checkpoints and build paper-ready reports.
Tüm ayarlar doğrudan dosya başındaki KULLANICI AYARLARI bölümünden kontrol edilebilir.
VS Code ile doğrudan 'Çalıştır' (Run / Play) butonuna basarak çalıştırabilirsiniz.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.data_loader_ssl_pretrained import loader
from models.Model import ATTNext


# ==============================================================================
#                      KULLANICI AYARLARI (VS CODE İÇİN)
# ==============================================================================
# 1) Hepsini birden çalıştırmak için:
#    RUN_ALL = True yapın. (Aşağıdaki RATIOS ve SEEDS listesindeki modelleri arar)
#
# 2) Belirli oran veya seed'leri çalıştırmak için:
#    RUN_ALL = True iken aşağıdaki RATIOS ve SEEDS listesini istediğiniz gibi düzenleyin.
#
# 3) Belirli spesifik checkpoint dosyalarını çalıştırmak için:
#    RUN_ALL = False yapıp SPECIFIC_CHECKPOINTS listesine tam dosya yollarını ekleyin.
# ==============================================================================

# ÇALIŞTIRMA MODU:
RUN_ALL = True  # True: RATIOS ve SEEDS listesini tara | False: SPECIFIC_CHECKPOINTS listesini çalıştır

# Test edilecek veri oranları (split ratio):
# 0.0025 -> 5 labels
# 0.005  -> 10 labels
# 0.01   -> 20 labels
# 0.05   -> 104 labels
# 0.10   -> 208 labels
# 0.50   -> 1040 labels
RATIOS = [
    0.0025,
    0.005,
    0.01,
    0.05,
    0.10,
]

# Test edilecek seed'ler:
SEEDS = [100, 200, 300]

# RUN_ALL = False ise sadece bu listedeki modeller test edilir:
SPECIFIC_CHECKPOINTS = [
    # "/Users/output/ckatar/output/isic_1/downstream/ATTNext[op=train mode=ssl_pretrained sslmode_modelname=Dino imnetpr=True bsize=8 epochs=503 imsize=256 lrate=0.0001 aug=True shuffle=True sratio=0.1 workers=2 cutoutpr=0.5 cutoutbox=25 cutmixpr=0 (1).5 noclasses=1]_seed_200",
]

# Eksik model varsa durmadan mevcut olanlarla devam etsin:
ALLOW_MISSING = True

# Bir model hata verirse diğerlerine devam etsin:
CONTINUE_ON_ERROR = True

# Dizin Yolları:
CHECKPOINT_DIR = "/Users/output/ckatar/output/isic_1/downstream"
DATA_ROOT = "/Users/input/data/ckatar/"
DATASET = "isic_2018_1"
TEST_DATASET = "isic_2018_1"

# Rapor Klasörü: Topo klasörü altına doğrudan kaydedilecek
TOPO_DIR = Path("/Users/cihankatar/Desktop/PhD/Topo")
REPORT_DIR = TOPO_DIR / "reports"

# Topo/main.tex içindeki Table 3 otomatik güncellensin mi?
UPDATE_MAIN_TEX = True
MAIN_TEX_PATH = TOPO_DIR / "main.tex"

# Model & Donanım Parametreleri:
DEVICE = "auto"          # 'auto' (varsa CUDA, yoksa CPU), 'cuda', 'cpu', 'mps'
BATCH_SIZE = 8
WORKERS = 2
IMAGE_SIZE = 256
THRESHOLD = 0.5
METRIC_REDUCTION = "legacy-batch"  # Table 3 uyumlu: 'legacy-batch' veya 'global'
MAX_BATCHES = None       # Hızlı deneme için sayı verebilirsiniz (örn: 3), tam test için None
# ==============================================================================


METRICS = ("iou", "dice", "recall", "precision", "accuracy")
LABEL_BUDGETS = {
    0.0025: 5,
    0.005: 10,
    0.01: 20,
    0.05: 104,
    0.10: 208,
    0.50: 1040,
}
OUTPUT_FOLDERS = {
    "isic_2018_1": "isic_1",
    "kvasir_1": "kvasir_1",
    "ham_1": "ham_1",
    "PH2Dataset": "PH2Dataset",
    "isic_2016_1": "isic_2016_1",
}
CHECKPOINT_PATTERN = re.compile(
    r"sratio=(?P<ratio>[0-9]+(?:\.[0-9]+)?).*?_seed_(?P<seed>[0-9]+)"
)
DOWNSTREAM_SIGNATURE = (
    "op=train",
    "mode=ssl_pretrained",
    "sslmode_modelname=Dino",
    "imnetpr=True",
    "epochs=503",
    "imsize=256",
    "noclasses=1",
)


def resolve_topo_dir():
    """Topo dizinini otomatik tespit eder."""
    if TOPO_DIR.is_dir():
        return TOPO_DIR.resolve()
    script_path = Path(__file__).resolve()
    candidate = script_path.parents[2] / "Topo"
    if candidate.is_dir():
        return candidate.resolve()
    return Path("Topo").resolve()


def parse_args():
    """
    Hem VS Code üzerinden direkt çalıştırmayı (argüman yokken kod başındaki
    değişkenleri kullanarak) hem de terminal/slurm argümanlarını destekler.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate TopoDistill downstream checkpoints with Mean & Variance."
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Explicit checkpoint path(s).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(CHECKPOINT_DIR),
        help="Directory searched for ATTNext ssl_pretrained checkpoints.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=RATIOS,
        help="Ratios to discover.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS,
        help="Seeds to discover and aggregate.",
    )
    parser.add_argument(
        "--dataset",
        default=DATASET,
        choices=sorted(OUTPUT_FOLDERS),
        help="Dataset on which the checkpoints were trained.",
    )
    parser.add_argument(
        "--test-dataset",
        default=TEST_DATASET,
        choices=sorted(OUTPUT_FOLDERS),
        help="Dataset evaluated by the report.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(DATA_ROOT),
        help="Path to data root directory containing datasets.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=REPORT_DIR,
        help="Directory where output reports will be saved.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--metric-reduction",
        choices=("legacy-batch", "global"),
        default=METRIC_REDUCTION,
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=DEVICE,
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        default=ALLOW_MISSING,
        help="Create report even if some ratio/seed checkpoints are missing.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        default=CONTINUE_ON_ERROR,
        help="Continue evaluation if an individual checkpoint fails.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=MAX_BATCHES,
        help="Evaluate only the first N batches (for quick smoke tests).",
    )
    parser.add_argument(
        "--update-main-tex",
        action="store_true",
        default=UPDATE_MAIN_TEX,
        help="Automatically update Table 3 in Topo/main.tex with results.",
    )
    parser.add_argument(
        "--main-tex",
        type=Path,
        default=MAIN_TEX_PATH,
        help="Path to main.tex file.",
    )
    args = parser.parse_args()

    # Kullanıcı kodun başında RUN_ALL = False yapmışsa ve SPECIFIC_CHECKPOINTS vermişse:
    if not RUN_ALL and SPECIFIC_CHECKPOINTS and not args.checkpoint:
        args.checkpoint = SPECIFIC_CHECKPOINTS

    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers cannot be negative")
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")
    return args


def ratio_key(value):
    return round(float(value), 8)


def label_budget(ratio):
    key = ratio_key(ratio)
    for known_ratio, budget in LABEL_BUDGETS.items():
        if math.isclose(key, known_ratio, rel_tol=0.0, abs_tol=1e-8):
            return budget
    return None


def parse_checkpoint_name(path):
    match = CHECKPOINT_PATTERN.search(path.name)
    if not match:
        raise ValueError(
            f"Cannot read sratio and seed from checkpoint name: {path.name}"
        )
    return ratio_key(match.group("ratio")), int(match.group("seed"))


def default_checkpoint_dir(dataset):
    base = (
        os.environ.get("ML_DATA_OUTPUT")
        or os.environ.get("ML_DATA_OUTPUT_LOCAL")
        or "/Users/output/ckatar/output"
    )
    return Path(base) / OUTPUT_FOLDERS.get(dataset, dataset)


def resolve_ambiguous_matches(matches):
    """
    Eğer aynı (ratio, seed) için birden fazla dosya varsa:
    1. macOS kopya etiketlerini (' (1)', '(1)', ' (2)' vb.) içermeyen temiz orijinal dosyayı seç.
    2. Eğer hala birden fazlaysa, en son değiştirilen (mtime) dosyayı seç.
    """
    if len(matches) == 1:
        return matches[0]

    clean_matches = [p for p in matches if not re.search(r"\(\d+\)", p.name)]
    if len(clean_matches) == 1:
        return clean_matches[0]
    candidates = clean_matches if clean_matches else matches

    try:
        candidates_sorted = sorted(
            candidates, key=lambda p: p.stat().st_mtime, reverse=True
        )
        return candidates_sorted[0]
    except Exception:
        return candidates[0]


def discover_checkpoints(args):
    if args.checkpoint:
        paths = [Path(value).expanduser() for value in args.checkpoint]
        absent = [str(path) for path in paths if not path.is_file()]
        if absent:
            raise FileNotFoundError(
                "Explicit checkpoint(s) not found:\n  " + "\n  ".join(absent)
            )
        records = []
        for path in paths:
            ratio, seed = parse_checkpoint_name(path)
            records.append({"path": path.resolve(), "ratio": ratio, "seed": seed})
        return sorted(records, key=lambda item: (-item["ratio"], item["seed"])), []

    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir(args.dataset)
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()

    if not checkpoint_dir.is_dir():
        # downstream alt klasörünü veya parent'ı dene
        if (checkpoint_dir / "downstream").is_dir():
            checkpoint_dir = checkpoint_dir / "downstream"
        elif (checkpoint_dir.parent / "downstream").is_dir():
            checkpoint_dir = checkpoint_dir.parent / "downstream"
        else:
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    search_dirs = [checkpoint_dir]
    downstream_subdir = checkpoint_dir / "downstream"
    if downstream_subdir.is_dir() and downstream_subdir not in search_dirs:
        search_dirs.append(downstream_subdir)

    by_experiment = {}
    seen_paths = set()
    for sdir in search_dirs:
        for path in sdir.iterdir():
            if not path.is_file() or not path.name.startswith("ATTNext["):
                continue
            if path.resolve() in seen_paths:
                continue
            if not all(fragment in path.name for fragment in DOWNSTREAM_SIGNATURE):
                continue
            try:
                ratio, seed = parse_checkpoint_name(path)
            except ValueError:
                continue
            seen_paths.add(path.resolve())
            by_experiment.setdefault((ratio, seed), []).append(path)

    records = []
    missing = []
    resolved_duplicates = []
    for ratio in [ratio_key(value) for value in args.ratios]:
        for seed in args.seeds:
            matches = by_experiment.get((ratio, seed), [])
            if not matches:
                missing.append({"ratio": ratio, "seed": seed})
            elif len(matches) == 1:
                records.append(
                    {"path": matches[0].resolve(), "ratio": ratio, "seed": seed}
                )
            else:
                chosen = resolve_ambiguous_matches(matches)
                resolved_duplicates.append((ratio, seed, chosen, matches))
                records.append(
                    {"path": chosen.resolve(), "ratio": ratio, "seed": seed}
                )

    if resolved_duplicates:
        print("\n[Otomatik Kopya Ayıklama - macOS Duplicate Filter]")
        for ratio, seed, chosen, all_m in resolved_duplicates:
            print(
                f"  ratio={ratio:g}, seed={seed}: {len(all_m)} dosya bulundu,"
                f" kopya olmayan asıl dosya seçildi -> {chosen.name}"
            )
        print()

    if missing and not args.allow_missing:
        details = ", ".join(
            f"ratio={item['ratio']:g}/seed={item['seed']}" for item in missing
        )
        raise FileNotFoundError(
            f"Missing {len(missing)} expected checkpoint(s) in {checkpoint_dir}: "
            f"{details}. Set ALLOW_MISSING = True to run with available files."
        )
    if not records:
        raise FileNotFoundError(f"No matching checkpoints found in {checkpoint_dir}")
    return sorted(records, key=lambda item: (-item["ratio"], item["seed"])), missing


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps was requested, but MPS is unavailable")
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Device: {device.type}")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict(path):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                payload = candidate
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload in {path}")
    for prefix in ("module.", "model."):
        if payload and all(str(key).startswith(prefix) for key in payload):
            payload = {str(key)[len(prefix):]: value for key, value in payload.items()}
    return payload


def counts_to_metrics(tp, fp, fn, tn):
    def safe_divide(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    return {
        "iou": safe_divide(tp, tp + fp + fn),
        "dice": safe_divide(2 * tp, 2 * tp + fp + fn),
        "recall": safe_divide(tp, tp + fn),
        "precision": safe_divide(tp, tp + fp),
        "accuracy": safe_divide(tp + tn, tp + fp + fn + tn),
    }


def binary_confusion(target, prediction):
    target = target.bool()
    prediction = prediction.bool()
    tp = torch.logical_and(target, prediction).sum().item()
    fp = torch.logical_and(torch.logical_not(target), prediction).sum().item()
    fn = torch.logical_and(target, torch.logical_not(prediction)).sum().item()
    tn = torch.logical_and(
        torch.logical_not(target), torch.logical_not(prediction)
    ).sum().item()
    return tp, fp, fn, tn


def evaluate(model, test_loader, device, threshold, reduction, max_batches=None):
    model.eval()
    global_counts = np.zeros(4, dtype=np.int64)
    batch_metric_sums = {metric: 0.0 for metric in METRICS}
    batches = 0
    images_seen = 0
    with torch.inference_mode():
        progress = tqdm(test_loader, desc="test", leave=False)
        for batch_index, (images, labels) in enumerate(progress):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True) > 0.5
            predictions = torch.sigmoid(model(images)) > threshold
            counts = binary_confusion(labels, predictions)
            global_counts += np.asarray(counts, dtype=np.int64)
            batch_metrics = counts_to_metrics(*counts)
            for metric in METRICS:
                batch_metric_sums[metric] += batch_metrics[metric]
            batches += 1
            images_seen += images.shape[0]

    if batches == 0:
        raise RuntimeError("The test loader produced no batches")
    if reduction == "legacy-batch":
        metrics = {
            metric: batch_metric_sums[metric] / batches for metric in METRICS
        }
    else:
        metrics = counts_to_metrics(*global_counts.tolist())
    return metrics, images_seen, batches


def summarize(results):
    groups = {}
    for result in results:
        groups.setdefault(ratio_key(result["ratio"]), []).append(result)
    summary = []
    for ratio, rows in groups.items():
        row = {
            "split_ratio": ratio,
            "label_budget": label_budget(ratio),
            "num_seeds": len(rows),
            "seeds": sorted(item["seed"] for item in rows),
        }
        for metric in METRICS:
            values = [item[metric] for item in rows]
            m = statistics.mean(values)
            s = statistics.stdev(values) if len(values) > 1 else None
            v = statistics.variance(values) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = m
            row[f"{metric}_std"] = s
            row[f"{metric}_variance"] = v
        summary.append(row)
    return sorted(
        summary,
        key=lambda item: (
            -(item["label_budget"] if item["label_budget"] is not None else -1),
            -item["split_ratio"],
        ),
    )


def display_metric(mean, std):
    return f"{mean:.3f}" if std is None else f"{mean:.3f} +/- {std:.3f}"


def display_metric_full(mean, std, variance):
    if std is None:
        return f"{mean:.4f} (var: {variance:.6f})"
    return f"{mean:.4f} +/- {std:.4f} (var: {variance:.6f})"


def latex_metric(mean, std):
    return f"{mean:.3f}" if std is None else f"${mean:.3f}\\pm{std:.3f}$"


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def update_main_tex(main_tex_path, summary):
    target = Path(main_tex_path).resolve()
    if not target.is_file():
        print(f"[LaTeX Update] Warning: main.tex not found at {target}, skipping.")
        return False

    content = target.read_text(encoding="utf-8")

    latex_lines = [f"\\multirow{{{len(summary)}}}{{*}}{{TopoDistill}}"]
    for item in summary:
        label = (
            item["label_budget"]
            if item["label_budget"] is not None
            else item["split_ratio"]
        )
        values = [
            latex_metric(item[f"{metric}_mean"], item[f"{metric}_std"])
            for metric in METRICS
        ]
        latex_lines.append(
            f"& {label} & {values[0]} & {values[1]} & - & - & "
            f"{values[2]} & {values[3]} & {values[4]} \\\\"
        )
    replacement = "\n".join(latex_lines)

    # Search for \multirow{...}{*}{TopoDistill} block up to \hline in Table 3
    pattern = re.compile(
        r"\\multirow\{\d+\}\{\*\}\{TopoDistill\}[\s\S]*?(?=\\hline)", re.MULTILINE
    )
    match = pattern.search(content)
    if not match:
        print("[LaTeX Update] Warning: Could not locate TopoDistill block in Table 3 in main.tex.")
        return False

    new_content = pattern.sub(replacement + "\n", content, count=1)

    backup_path = target.with_suffix(".tex.bak")
    backup_path.write_text(content, encoding="utf-8")
    target.write_text(new_content, encoding="utf-8")
    print(f"[LaTeX Update] Table 3 in {target} has been successfully updated!")
    print(f"[LaTeX Update] Backup of original file saved to: {backup_path}")
    return True


def write_reports(args, results, missing, failures, elapsed_seconds):
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)

    # 1. per_checkpoint.csv
    checkpoint_fields = [
        "label_budget",
        "split_ratio",
        "seed",
        "test_dataset",
        "num_images",
        "num_batches",
        "metric_reduction",
        *METRICS,
        "duration_seconds",
        "checkpoint",
    ]
    checkpoint_rows = [
        {key: result[key] for key in checkpoint_fields} for result in results
    ]
    write_csv(report_dir / "per_checkpoint.csv", checkpoint_fields, checkpoint_rows)

    # 2. summary.csv (ortalama, std ve varyans dahil)
    summary_fields = ["label_budget", "split_ratio", "num_seeds", "seeds"]
    for metric in METRICS:
        summary_fields.extend(
            (f"{metric}_mean", f"{metric}_std", f"{metric}_variance")
        )
    summary_rows = []
    for item in summary:
        row = dict(item)
        row["seeds"] = ";".join(str(seed) for seed in item["seeds"])
        summary_rows.append(row)
    write_csv(report_dir / "summary.csv", summary_fields, summary_rows)

    # 3. results.json
    machine_report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_dataset": args.dataset,
        "test_dataset": args.test_dataset,
        "threshold": args.threshold,
        "metric_reduction": args.metric_reduction,
        "max_batches": args.max_batches,
        "elapsed_seconds": elapsed_seconds,
        "results": results,
        "summary": summary,
        "missing": missing,
        "failures": failures,
    }
    with (report_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(machine_report, handle, indent=2, ensure_ascii=False)

    # 4. table3_topodistill.tex
    latex_lines = [
        "% Auto-generated by test.py; paste into the TopoDistill block of Table 3.",
        f"\\multirow{{{len(summary)}}}{{*}}{{TopoDistill}}",
    ]
    for item in summary:
        label = (
            item["label_budget"]
            if item["label_budget"] is not None
            else item["split_ratio"]
        )
        values = [
            latex_metric(item[f"{metric}_mean"], item[f"{metric}_std"])
            for metric in METRICS
        ]
        latex_lines.append(
            f"& {label} & {values[0]} & {values[1]} & - & - & "
            f"{values[2]} & {values[3]} & {values[4]} \\\\"
        )
    with (report_dir / "table3_topodistill.tex").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write("\n".join(latex_lines) + "\n")

    # 5. report.md
    markdown = [
        "# TopoDistill Downstream Test Raporu",
        "",
        f"- **Eğitim Veri Seti**: `{args.dataset}`",
        f"- **Test Veri Seti**: `{args.test_dataset}`",
        f"- **Görüntü Sayısı (Her run)**: `{results[0]['num_images'] if results else 0}`",
        f"- **Threshold**: `{args.threshold}`",
        f"- **Metric Reduction**: `{args.metric_reduction}`",
        f"- **Test Edilen Toplam Checkpoint**: `{len(results)}`",
        f"- **Rapor Oluşturulma Zamanı**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]
    if args.max_batches is not None:
        markdown.extend(
            [
                "> **UYARI**: `--max-batches` parametresi kullanıldı. Bunlar hızlı deneme sonuçlarıdır.",
                "",
            ]
        )
    markdown.extend(
        [
            "## Özet Tablo (Ortalama, Standart Sapma ve Varyans)",
            "",
            "Aşağıdaki varyans ve standart sapma değerleri farklı seed'ler üzerinden hesaplanmıştır.",
            "",
            "| Etiket (Labels) | Oran (Ratio) | Seed'ler | IoU (Ortalama ± Std, Varyans) | Dice (Ortalama ± Std, Varyans) | Recall (Ortalama ± Std, Varyans) | Precision (Ortalama ± Std, Varyans) | Accuracy (Ortalama ± Std, Varyans) |",
            "|---:|---:|:---|:---|:---|:---|:---|:---|",
        ]
    )
    for item in summary:
        values = [
            display_metric_full(
                item[f"{metric}_mean"],
                item[f"{metric}_std"],
                item[f"{metric}_variance"],
            )
            for metric in METRICS
        ]
        label = item["label_budget"] if item["label_budget"] is not None else "n/a"
        seeds = ", ".join(str(seed) for seed in item["seeds"])
        markdown.append(
            f"| {label} | {item['split_ratio']:g} | {seeds} | "
            + " | ".join(values)
            + " |"
        )

    markdown.extend(
        [
            "",
            "## Makale Tablosu Formatı (Table 3 Uyumlu: Mean ± Std)",
            "",
            "| Labels | Ratio | Seeds | IoU | Dice | Recall | Precision | Accuracy |",
            "|---:|---:|:---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summary:
        values = [
            display_metric(item[f"{metric}_mean"], item[f"{metric}_std"])
            for metric in METRICS
        ]
        label = item["label_budget"] if item["label_budget"] is not None else "n/a"
        seeds = ", ".join(str(seed) for seed in item["seeds"])
        markdown.append(
            f"| {label} | {item['split_ratio']:g} | {seeds} | "
            + " | ".join(values)
            + " |"
        )

    markdown.extend(
        [
            "",
            "## Checkpoint Başına Ham Değerler",
            "",
            "| Labels | Ratio | Seed | IoU | Dice | Recall | Precision | Accuracy | Süre (sn) |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sorted(results, key=lambda item: (-item["ratio"], item["seed"])):
        values = [f"{result[metric]:.6f}" for metric in METRICS]
        markdown.append(
            f"| {result['label_budget']} | {result['ratio']:g} | "
            f"{result['seed']} | " + " | ".join(values) + f" | {result['duration_seconds']} |"
        )
    if missing:
        markdown.extend(["", "## Eksik Checkpoint'ler", ""])
        markdown.extend(
            f"- ratio={item['ratio']:g}, seed={item['seed']}" for item in missing
        )
    if failures:
        markdown.extend(["", "## Hata Veren Checkpoint'ler", ""])
        markdown.extend(
            f"- `{item['checkpoint']}`: {item['error']}" for item in failures
        )
    with (report_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")

    # Ayrıca doğrudan Topo klasörünün kök dizinine de kolay erişim için kopyala:
    topo_dir = resolve_topo_dir()
    if topo_dir.is_dir() and topo_dir != report_dir:
        try:
            shutil.copy2(report_dir / "report.md", topo_dir / "report.md")
            shutil.copy2(report_dir / "summary.csv", topo_dir / "summary.csv")
            shutil.copy2(report_dir / "table3_topodistill.tex", topo_dir / "table3_topodistill.tex")
            print(f"[Rapor] Rapor kopyaları doğrudan {topo_dir} dizinine de kaydedildi.")
        except Exception as e:
            print(f"[Rapor Uyarı] Topo ana dizinine kopyalama yapılamadı: {e}")

    return report_dir, summary


def print_ascii_summary(summary):
    """Terminalde ortalama, standart sapma ve varyansı tablo olarak basar."""
    print("\n" + "=" * 110)
    print("           Topodistill Downstream Sonuç Özeti (Ortalama, Std Dev ve Varyans)")
    print("=" * 110)
    header = f"{'Labels':<8} | {'Ratio':<7} | {'Seeds':<15} | {'Metrik':<10} | {'Ortalama':<12} | {'Std Dev':<12} | {'Varyans':<14}"
    print(header)
    print("-" * 110)
    for item in summary:
        label = str(item["label_budget"]) if item["label_budget"] is not None else "n/a"
        ratio = f"{item['split_ratio']:g}"
        seeds = ",".join(str(s) for s in item["seeds"])
        for idx, metric in enumerate(METRICS):
            m = item[f"{metric}_mean"]
            s = item[f"{metric}_std"]
            v = item[f"{metric}_variance"]
            s_str = f"{s:.4f}" if s is not None else "n/a"
            v_str = f"{v:.6f}" if v is not None else "n/a"
            m_str = f"{m:.4f}"
            if idx == 0:
                print(f"{label:<8} | {ratio:<7} | {seeds:<15} | {metric:<10} | {m_str:<12} | {s_str:<12} | {v_str:<14}")
            else:
                print(f"{'':<8} | {'':<7} | {'':<15} | {metric:<10} | {m_str:<12} | {s_str:<12} | {v_str:<14}")
        print("-" * 110)
    print("=" * 110 + "\n")


def main():
    args = parse_args()
    if args.data_root:
        os.environ["ML_DATA_ROOT"] = str(args.data_root.expanduser().resolve())

    print("=" * 70)
    print("Topodistill Test ve Raporlama Başlatılıyor")
    print("=" * 70)
    print(f"Mod                : {'Tüm modeller (RUN_ALL)' if RUN_ALL else 'Belirli modeller (SPECIFIC)'}")
    print(f"Checkpoint Klasörü : {args.checkpoint_dir}")
    print(f"Veri Seti Kökü     : {os.environ.get('ML_DATA_ROOT', args.data_root)}")
    print(f"Hedef Rapor Dizini : {args.report_dir}")
    print(f"Oranlar (Ratios)   : {args.ratios}")
    print(f"Seed'ler           : {args.seeds}")
    print("=" * 70)

    records, missing = discover_checkpoints(args)
    device = select_device(args.device)
    seed_everything(0)

    test_loader = loader(
        "test",
        "supervised",
        "Dino",
        args.batch_size,
        args.workers,
        args.image_size,
        0.0,
        None,
        False,
        records[0]["ratio"],
        args.test_dataset,
        0,
    )
    print(f"Test Veri Seti     : {args.test_dataset} ({len(test_loader.dataset)} görsel)")
    print(f"Seçilen Modeller   : {len(records)} adet")
    print(f"Metrik Hesaplama   : {args.metric_reduction}\n")

    model = ATTNext("supervised").to(device)
    results = []
    failures = []
    run_started = time.time()
    for index, record in enumerate(records, start=1):
        checkpoint = record["path"]
        print(
            f"[{index}/{len(records)}] ratio={record['ratio']:g}, "
            f"seed={record['seed']}: {checkpoint.name}"
        )
        started = time.time()
        try:
            model.load_state_dict(load_state_dict(checkpoint), strict=True)
            metrics, num_images, num_batches = evaluate(
                model,
                test_loader,
                device,
                args.threshold,
                args.metric_reduction,
                args.max_batches,
            )
            result = {
                "label_budget": label_budget(record["ratio"]),
                "split_ratio": record["ratio"],
                "ratio": record["ratio"],
                "seed": record["seed"],
                "test_dataset": args.test_dataset,
                "num_images": num_images,
                "num_batches": num_batches,
                "metric_reduction": args.metric_reduction,
                **metrics,
                "duration_seconds": round(time.time() - started, 3),
                "checkpoint": str(checkpoint),
            }
            results.append(result)
            print(
                "  -> "
                + " | ".join(
                    f"{metric}={metrics[metric]:.4f}" for metric in METRICS
                )
            )
        except Exception as error:
            failures.append({"checkpoint": str(checkpoint), "error": str(error)})
            print(f"  FAILED: {error}")
            if not args.continue_on_error:
                if results:
                    write_reports(
                        args,
                        results,
                        missing,
                        failures,
                        time.time() - run_started,
                    )
                raise

        # Her checkpoint sonrası güncel raporu diske yaz (kesilmelere karşı koruma)
        if results:
            write_reports(
                args,
                results,
                missing,
                failures,
                time.time() - run_started,
            )

    if not results:
        raise RuntimeError("Hiçbir checkpoint başarıyla test edilemedi.")

    report_dir, summary = write_reports(
        args,
        results,
        missing,
        failures,
        time.time() - run_started,
    )

    print_ascii_summary(summary)
    print(f"Tüm raporlar kaydedildi: {report_dir}")
    print(f"Topo ana klasörüne kopyalandı: {resolve_topo_dir() / 'report.md'}")

    # LaTeX main.tex otomatik güncelleme
    if args.update_main_tex:
        main_tex_target = args.main_tex if args.main_tex else (resolve_topo_dir() / "main.tex")
        if main_tex_target and Path(main_tex_target).is_file():
            update_main_tex(main_tex_target, summary)
        else:
            print(f"[LaTeX Update] main.tex dosyası bulunamadı: {main_tex_target}")

    if failures:
        print(f"\nUYARI: {len(failures)} adet checkpoint hata verdi. Detaylar report.md dosyasında.")


if __name__ == "__main__":
    main()
