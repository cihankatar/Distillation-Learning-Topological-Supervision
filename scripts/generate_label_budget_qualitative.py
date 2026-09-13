"""Generate Figure-13-style qualitative comparisons for several label budgets.

The script loads real ATTNext downstream checkpoints, runs inference on fixed
ISIC2018 test images, writes per-image IoU/Dice scores on the predictions, and
exports both 1440x300 row images and one stacked figure per label budget.

Column order matches the manuscript figure:
    Original | TopoDistill | Self-distillation | Supervised | Ground truth
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.Model import ATTNext


CHECKPOINT_ROOT = Path(
    os.environ.get("ML_DATA_OUTPUT", REPO_ROOT / "checkpoints")
) / "isic_1"
DATASET_ROOT = Path(
    os.environ.get("ML_DATA_ROOT", REPO_ROOT / "data")
) / "isic_2018_1" / "test"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "qualitative_label_budget_figures"
DEFAULT_TOPO_FIGURE_DIR = WORKSPACE_ROOT / "Topo" / "figures"

# These are the same five test images used in the manuscript's 20-label figure.
# Holding the images fixed makes changes across label budgets directly visible.
DEFAULT_IMAGE_IDS = (
    "ISIC_0000012",
    "ISIC_0000024",
    "ISIC_0000290",
    "ISIC_0000277",
    "ISIC_0000268",
)

IMAGE_SIZE = 256
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True)
class MethodSpec:
    display_name: str
    folder: str
    epoch: int
    preferred_seed: Optional[int]


@dataclass(frozen=True)
class BudgetSpec:
    labels: int
    ratios: Dict[str, float]


METHODS: Tuple[MethodSpec, ...] = (
    MethodSpec("TopoDistill", "downstream_topodistill", 503, 100),
    MethodSpec(
        "Self-distillation", "downstream_selfdistillationonly", 499, None
    ),
    MethodSpec("Supervised", "supervised", 450, None),
)

BUDGETS: Tuple[BudgetSpec, ...] = (
    # The older supervised/self-distillation runs use 0.002 in their filenames;
    # the TopoDistill run uses the corrected exact five-image ratio, 0.0025.
    BudgetSpec(
        5,
        {
            "TopoDistill": 0.0025,
            "Self-distillation": 0.002,
            "Supervised": 0.002,
        },
    ),
    BudgetSpec(
        10,
        {
            "TopoDistill": 0.005,
            "Self-distillation": 0.005,
            "Supervised": 0.005,
        },
    ),
    # The manuscript's fixed ISIC2018 split resolves sratio=0.05 to 104 images.
    BudgetSpec(
        104,
        {
            "TopoDistill": 0.05,
            "Self-distillation": 0.05,
            "Supervised": 0.05,
        },
    ),
)

RATIO_PATTERN = re.compile(r"sratio=(?P<ratio>[0-9]+(?:\.[0-9]+)?)")
SEED_PATTERN = re.compile(r"_seed_(?P<seed>[0-9]+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate real qualitative comparisons for 5, 10, and 104 labels."
    )
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--topo-figure-dir",
        type=Path,
        default=DEFAULT_TOPO_FIGURE_DIR,
        help="Also copy stacked figures here. Use --no-copy-to-topo to disable.",
    )
    parser.add_argument("--no-copy-to-topo", action="store_true")
    parser.add_argument(
        "--image-ids", nargs="+", default=list(DEFAULT_IMAGE_IDS)
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device(requested)


def checkpoint_ratio(path: Path) -> Optional[float]:
    match = RATIO_PATTERN.search(path.name)
    return float(match.group("ratio")) if match else None


def checkpoint_seed(path: Path) -> Optional[int]:
    match = SEED_PATTERN.search(path.name)
    return int(match.group("seed")) if match else None


def select_checkpoint(
    root: Path, method: MethodSpec, ratio: float
) -> Path:
    folder = root / method.folder
    if not folder.is_dir():
        raise FileNotFoundError(f"Checkpoint folder not found: {folder}")

    candidates = []
    for path in folder.iterdir():
        if not path.is_file() or not path.name.startswith("ATTNext["):
            continue
        if f"epochs={method.epoch}" not in path.name:
            continue
        parsed_ratio = checkpoint_ratio(path)
        if parsed_ratio is None or not np.isclose(parsed_ratio, ratio):
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No {method.display_name} checkpoint found for sratio={ratio:g} in {folder}"
        )

    # Prefer a clean filename over macOS duplicate names such as "(1)".
    clean = [path for path in candidates if not re.search(r"\(\d+\)", path.name)]
    if clean:
        candidates = clean

    if method.preferred_seed is not None:
        preferred = [
            path
            for path in candidates
            if checkpoint_seed(path) == method.preferred_seed
        ]
        if preferred:
            return sorted(preferred)[0]

    unseeded = [path for path in candidates if checkpoint_seed(path) is None]
    if unseeded:
        return sorted(unseeded)[0]

    return sorted(
        candidates,
        key=lambda path: (
            checkpoint_seed(path) if checkpoint_seed(path) is not None else -1,
            path.name,
        ),
    )[0]


def unwrap_state_dict(payload: object, path: Path) -> Dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload: {path}")

    state = payload
    for prefix in ("module.", "model."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix) :]: value for key, value in state.items()}
    return state


def load_model(path: Path, device: torch.device) -> ATTNext:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    model = ATTNext("supervised")
    model.load_state_dict(unwrap_state_dict(payload, path), strict=True)
    model.to(device)
    model.eval()
    return model


def load_examples(
    dataset_root: Path, image_ids: Sequence[str]
) -> Tuple[np.ndarray, torch.Tensor, np.ndarray]:
    originals: List[np.ndarray] = []
    normalized: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for image_id in image_ids:
        image_path = dataset_root / "images" / f"{image_id}.jpg"
        mask_path = dataset_root / "masks" / f"{image_id}_segmentation.png"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing image/mask pair: {image_path}, {mask_path}"
            )

        image = Image.open(image_path).convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        mask = Image.open(mask_path).convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
        )
        image_array = np.asarray(image, dtype=np.float32) / 255.0
        mask_array = np.asarray(mask, dtype=np.uint8) >= 128

        originals.append(image_array)
        normalized.append((image_array - IMAGENET_MEAN) / IMAGENET_STD)
        masks.append(mask_array)

    batch = torch.from_numpy(np.stack(normalized)).permute(0, 3, 1, 2).float()
    return np.stack(originals), batch, np.stack(masks)


def predict(
    model: ATTNext,
    batch: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    with torch.inference_mode():
        logits = model(batch.to(device, non_blocking=True))
        probabilities = torch.sigmoid(logits)
    return (probabilities[:, 0] >= threshold).cpu().numpy()


def overlap_scores(prediction: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(prediction, target).sum(dtype=np.int64)
    union = np.logical_or(prediction, target).sum(dtype=np.int64)
    total = prediction.sum(dtype=np.int64) + target.sum(dtype=np.int64)
    iou = 1.0 if union == 0 else float(intersection / union)
    dice = 1.0 if total == 0 else float((2 * intersection) / total)
    return iou, dice


def save_row(
    output_path: Path,
    original: np.ndarray,
    predictions: Dict[str, np.ndarray],
    target: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    scores = {
        name: overlap_scores(mask, target) for name, mask in predictions.items()
    }
    panels: Iterable[Tuple[Optional[str], np.ndarray]] = (
        (None, original),
        ("TopoDistill", predictions["TopoDistill"]),
        ("Self-distillation", predictions["Self-distillation"]),
        ("Supervised", predictions["Supervised"]),
        (None, target),
    )

    fig, axes = plt.subplots(1, 5, figsize=(14.4, 3.0), dpi=100)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.04, top=0.96, wspace=0.055)
    for axis, (method_name, panel) in zip(axes, panels):
        if panel.ndim == 3:
            axis.imshow(np.clip(panel, 0.0, 1.0), interpolation="bilinear")
        else:
            axis.imshow(panel, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        if method_name is not None:
            iou, dice = scores[method_name]
            axis.text(
                0.04,
                0.96,
                f"IoU: {iou:.2f}\nDice: {dice:.2f}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=16,
                fontweight="bold",
                color="white",
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.72,
                    "pad": 2.5,
                },
            )
        axis.set_axis_off()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, facecolor="white", transparent=False)
    plt.close(fig)
    return {
        method: {"iou": float(values[0]), "dice": float(values[1])}
        for method, values in scores.items()
    }


def stack_rows(row_paths: Sequence[Path], output_path: Path, gap: int = 4) -> None:
    rows = [Image.open(path).convert("RGB") for path in row_paths]
    if not rows:
        raise ValueError("No row images were supplied")
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        canvas.paste(row, ((width - row.width) // 2, y))
        y += row.height + gap
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=100)


def run(args: argparse.Namespace) -> None:
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    device = choose_device(args.device)
    output_dir = args.output_dir.resolve()
    originals, batch, targets = load_examples(
        args.dataset_root.resolve(), args.image_ids
    )
    print(f"Device: {device}")
    print("Images: " + ", ".join(args.image_ids))

    manifest = {
        "column_order": [
            "Original",
            "TopoDistill",
            "Self-distillation",
            "Supervised",
            "Ground truth",
        ],
        "threshold": args.threshold,
        "image_ids": list(args.image_ids),
        "budgets": {},
    }

    for budget in BUDGETS:
        print(f"\nGenerating {budget.labels}-label comparison")
        predictions: Dict[str, np.ndarray] = {}
        selected_checkpoints = {}
        for method in METHODS:
            ratio = budget.ratios[method.display_name]
            checkpoint = select_checkpoint(
                args.checkpoint_root.resolve(), method, ratio
            )
            selected_checkpoints[method.display_name] = {
                "path": str(checkpoint),
                "ratio": ratio,
                "seed": checkpoint_seed(checkpoint),
            }
            print(f"  {method.display_name}: {checkpoint.name}")
            model = load_model(checkpoint, device)
            predictions[method.display_name] = predict(
                model, batch, device, args.threshold
            )
            del model

        budget_dir = output_dir / f"{budget.labels}_labels"
        row_paths = []
        image_results = []
        for index, image_id in enumerate(args.image_ids):
            row_path = budget_dir / f"row_{index + 1}_{image_id}.png"
            per_image_predictions = {
                method.display_name: predictions[method.display_name][index]
                for method in METHODS
            }
            scores = save_row(
                row_path,
                originals[index],
                per_image_predictions,
                targets[index],
            )
            row_paths.append(row_path)
            image_results.append({"image_id": image_id, "scores": scores})

        composite_path = output_dir / f"qualitative_{budget.labels}_labels.png"
        stack_rows(row_paths, composite_path)
        manifest["budgets"][str(budget.labels)] = {
            "checkpoints": selected_checkpoints,
            "images": image_results,
            "rows": [str(path) for path in row_paths],
            "composite": str(composite_path),
        }
        print(f"  Saved: {composite_path}")

        if not args.no_copy_to_topo:
            args.topo_figure_dir.mkdir(parents=True, exist_ok=True)
            topo_path = (
                args.topo_figure_dir
                / f"qualitative_comparison_{budget.labels}_labels.png"
            )
            shutil.copy2(composite_path, topo_path)
            print(f"  Copied: {topo_path}")

    manifest_path = output_dir / "qualitative_scores.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    run(parse_args())
