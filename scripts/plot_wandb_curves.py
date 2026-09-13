"""Export journal-quality mean ± SD curves from completed W&B runs.

This script only reads run histories; it never starts or resumes training.
Example:

    python scripts/plot_wandb_curves.py \
      --metric validation/gt_monitor_iou \
      --group 'TopoDistill=entity/project/run1,entity/project/run2' \
      --group 'Self-Distillation=entity/project/run3,entity/project/run4' \
      --output figures/monitor_iou
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb


def parse_group(value: str) -> tuple[str, list[str]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=run/path,run/path")
    label, raw_paths = value.split("=", 1)
    paths = [item.strip() for item in raw_paths.split(",") if item.strip()]
    if not label.strip() or not paths:
        raise argparse.ArgumentTypeError("expected LABEL=run/path,run/path")
    return label.strip(), paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", action="append", type=parse_group, required=True)
    parser.add_argument("--metric", required=True, help="W&B history metric")
    parser.add_argument("--x-key", default="epoch")
    parser.add_argument("--output", type=Path, required=True, help="Path without suffix")
    parser.add_argument("--ylabel", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--smooth-window", type=int, default=1)
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(values, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def load_history(api: wandb.Api, run_path: str, x_key: str, metric: str):
    run = api.run(run_path)
    points = {}
    for row in run.scan_history(keys=[x_key, metric]):
        x_value = row.get(x_key)
        y_value = row.get(metric)
        if x_value is None or y_value is None:
            continue
        try:
            x_number = float(x_value)
            y_number = float(y_value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(x_number) and np.isfinite(y_number):
            points[x_number] = y_number
    if not points:
        raise RuntimeError(f"No finite {metric!r} history found in {run_path}")
    return points


def main() -> None:
    args = parse_args()
    if args.smooth_window < 1:
        raise ValueError("--smooth-window must be at least 1")

    api = wandb.Api()
    figure, axis = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    exported_rows = []

    for label, run_paths in args.group:
        histories = [
            load_history(api, run_path, args.x_key, args.metric)
            for run_path in run_paths
        ]
        # Use only epochs shared by every run so each mean/SD point has the
        # same sample count. This avoids visually misleading partial bands.
        all_steps = sorted(set.intersection(*(set(history) for history in histories)))
        if not all_steps:
            raise RuntimeError(f"No shared {args.x_key!r} values for {label}")
        step_values = defaultdict(list)
        for history in histories:
            for step, value in history.items():
                step_values[step].append(value)

        x_values = np.asarray(all_steps, dtype=np.float64)
        means = np.asarray(
            [np.mean(step_values[step]) for step in all_steps], dtype=np.float64
        )
        standard_deviations = np.asarray(
            [
                np.std(step_values[step], ddof=1)
                if len(step_values[step]) > 1
                else 0.0
                for step in all_steps
            ],
            dtype=np.float64,
        )
        means = moving_average(means, args.smooth_window)
        standard_deviations = moving_average(
            standard_deviations, args.smooth_window
        )

        line = axis.plot(x_values, means, linewidth=2.0, label=label)[0]
        if len(run_paths) > 1:
            axis.fill_between(
                x_values,
                means - standard_deviations,
                means + standard_deviations,
                color=line.get_color(),
                alpha=0.18,
                linewidth=0,
            )
        for step, mean, std in zip(x_values, means, standard_deviations):
            exported_rows.append(
                {
                    "method": label,
                    args.x_key: step,
                    "mean": mean,
                    "sample_sd": std,
                    "n": len(step_values[step]),
                    "metric": args.metric,
                }
            )

    axis.set_xlabel(args.x_key.replace("_", " ").title())
    axis.set_ylabel(args.ylabel or args.metric.split("/")[-1].replace("_", " ").title())
    if args.title:
        axis.set_title(args.title)
    axis.grid(axis="y", alpha=0.25)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)

    fieldnames = ["method", args.x_key, "mean", "sample_sd", "n", "metric"]
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exported_rows)


if __name__ == "__main__":
    main()
