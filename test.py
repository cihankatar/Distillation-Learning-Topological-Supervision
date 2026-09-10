"""Evaluate downstream checkpoints and build paper-ready reports."""

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.data_loader_ssl_pretrained import loader
from models.Model import ATTNext


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
    r"sratio=(?P<ratio>[0-9]+(?:\.[0-9]+)?)"
    r".*_seed_(?P<seed>[0-9]+)(?:\.[^.]+)?$"
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate TopoDistill downstream checkpoints and aggregate "
            "mean +/- standard deviation across seeds."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Explicit checkpoint path. Repeat to evaluate several files.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory searched for ATTNext ssl_pretrained checkpoints.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[0.0025, 0.005, 0.01, 0.05, 0.1],
        help="Ratios to discover (default: 5, 10, 20, 104, 208 labels).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[100, 200, 300],
        help="Seeds to discover and aggregate.",
    )
    parser.add_argument(
        "--dataset",
        default="isic_2018_1",
        choices=sorted(OUTPUT_FOLDERS),
        help="Dataset on which the checkpoints were trained.",
    )
    parser.add_argument(
        "--test-dataset",
        default="isic_2018_1",
        choices=sorted(OUTPUT_FOLDERS),
        help="Dataset evaluated by the report (default: ISIC 2018 test).",
    )
    parser.add_argument("--data-root", type=Path, help="Overrides ML_DATA_ROOT.")
    parser.add_argument(
        "--report-dir", type=Path, default=Path("reports/downstream_test")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--metric-reduction",
        choices=("legacy-batch", "global"),
        default="legacy-batch",
        help=(
            "legacy-batch reproduces the historical Table 3 test.py; global "
            "uses one confusion matrix over the complete test set."
        ),
    )
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Create a partial report if an expected ratio/seed file is absent.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record a failed checkpoint and continue with remaining files.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        help="Evaluate only the first N batches (smoke tests only).",
    )
    args = parser.parse_args()
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
    return Path(base) / OUTPUT_FOLDERS[dataset]


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
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    by_experiment = {}
    for path in checkpoint_dir.iterdir():
        if not path.is_file() or not path.name.startswith("ATTNext["):
            continue
        if not all(fragment in path.name for fragment in DOWNSTREAM_SIGNATURE):
            continue
        try:
            ratio, seed = parse_checkpoint_name(path)
        except ValueError:
            continue
        by_experiment.setdefault((ratio, seed), []).append(path)

    records = []
    missing = []
    ambiguous = []
    for ratio in [ratio_key(value) for value in args.ratios]:
        for seed in args.seeds:
            matches = by_experiment.get((ratio, seed), [])
            if not matches:
                missing.append({"ratio": ratio, "seed": seed})
            elif len(matches) > 1:
                ambiguous.append((ratio, seed, matches))
            else:
                records.append(
                    {"path": matches[0].resolve(), "ratio": ratio, "seed": seed}
                )

    if ambiguous:
        details = []
        for ratio, seed, paths in ambiguous:
            details.append(
                f"ratio={ratio:g}, seed={seed}: "
                + ", ".join(str(path) for path in paths)
            )
        raise RuntimeError(
            "Multiple checkpoints match one experiment; pass explicit "
            "--checkpoint paths:\n  " + "\n  ".join(details)
        )
    if missing and not args.allow_missing:
        details = ", ".join(
            f"ratio={item['ratio']:g}/seed={item['seed']}" for item in missing
        )
        raise FileNotFoundError(
            f"Missing {len(missing)} expected checkpoint(s) in {checkpoint_dir}: "
            f"{details}. Use --allow-missing only for an intentional partial report."
        )
    if not records:
        raise FileNotFoundError(f"No matching checkpoints found in {checkpoint_dir}")
    return sorted(records, key=lambda item: (-item["ratio"], item["seed"])), missing


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type == "cuda":
        print(f"Device: cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: cpu")
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
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else None
            )
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


def latex_metric(mean, std):
    return f"{mean:.3f}" if std is None else f"${mean:.3f}\\pm{std:.3f}$"


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(args, results, missing, failures, elapsed_seconds):
    report_dir = args.report_dir.resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(results)

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

    summary_fields = ["label_budget", "split_ratio", "num_seeds", "seeds"]
    for metric in METRICS:
        summary_fields.extend((f"{metric}_mean", f"{metric}_std"))
    summary_rows = []
    for item in summary:
        row = dict(item)
        row["seeds"] = ";".join(str(seed) for seed in item["seeds"])
        summary_rows.append(row)
    write_csv(report_dir / "summary.csv", summary_fields, summary_rows)

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

    markdown = [
        "# TopoDistill downstream test report",
        "",
        f"- Training dataset: `{args.dataset}`",
        f"- Test dataset: `{args.test_dataset}`",
        f"- Evaluated images per run: `{results[0]['num_images'] if results else 0}`",
        f"- Prediction threshold: `{args.threshold}`",
        f"- Metric reduction: `{args.metric_reduction}`",
        f"- Evaluated checkpoints: `{len(results)}`",
        "",
    ]
    if args.max_batches is not None:
        markdown.extend(
            [
                "> WARNING: `--max-batches` was used. These are smoke-test "
                "results, not paper results.",
                "",
            ]
        )
    markdown.extend(
        [
            "The standard deviations below are sample standard deviations across "
            "independently trained seeds, not across test images.",
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
            "## Per-checkpoint values",
            "",
            "| Labels | Ratio | Seed | IoU | Dice | Recall | Precision | Accuracy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in sorted(results, key=lambda item: (-item["ratio"], item["seed"])):
        values = [f"{result[metric]:.6f}" for metric in METRICS]
        markdown.append(
            f"| {result['label_budget']} | {result['ratio']:g} | "
            f"{result['seed']} | " + " | ".join(values) + " |"
        )
    if missing:
        markdown.extend(["", "## Missing checkpoints", ""])
        markdown.extend(
            f"- ratio={item['ratio']:g}, seed={item['seed']}" for item in missing
        )
    if failures:
        markdown.extend(["", "## Failed checkpoints", ""])
        markdown.extend(
            f"- `{item['checkpoint']}`: {item['error']}" for item in failures
        )
    with (report_dir / "report.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown) + "\n")
    return report_dir, summary


def main():
    args = parse_args()
    if args.data_root:
        os.environ["ML_DATA_ROOT"] = str(args.data_root.expanduser().resolve())
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
    print(f"Test dataset: {args.test_dataset} ({len(test_loader.dataset)} images)")
    print(f"Checkpoints selected: {len(records)}")
    print(f"Metric reduction: {args.metric_reduction}")

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
                "  "
                + " | ".join(
                    f"{metric}={metrics[metric]:.6f}" for metric in METRICS
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
        if results:
            write_reports(
                args,
                results,
                missing,
                failures,
                time.time() - run_started,
            )

    if not results:
        raise RuntimeError("No checkpoint was evaluated successfully")
    report_dir, summary = write_reports(
        args,
        results,
        missing,
        failures,
        time.time() - run_started,
    )
    print("\nMean +/- sample standard deviation across seeds")
    for item in summary:
        label = item["label_budget"] if item["label_budget"] is not None else "n/a"
        values = " | ".join(
            f"{metric}="
            f"{display_metric(item[f'{metric}_mean'], item[f'{metric}_std'])}"
            for metric in METRICS
        )
        print(f"labels={label}, n={item['num_seeds']} | {values}")
    print(f"\nReports written to: {report_dir}")
    if failures:
        raise RuntimeError(f"{len(failures)} checkpoint(s) failed; see report.md")


if __name__ == "__main__":
    main()
