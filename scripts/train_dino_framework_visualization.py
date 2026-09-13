"""Generate deterministic, real DINO framework assets from a saved encoder.

This is a figure-only derivative of ``train_dino.py``.  It deliberately does
not train or overwrite the checkpoint.  It recreates the two-global/four-local
view geometry used by the training loader, loads the recorded encoder weights,
and exports genuine stage activations for the manuscript framework figure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from models.enc import encoder_function


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

DEFAULT_IMAGE = Path(os.environ.get("ML_DATA_ROOT", REPO_ROOT / "data")) / (
    "isic_2018_1/train/images/ISIC_0000377.jpg"
)
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "TOPODISTILL_ENCODER_CHECKPOINT",
        REPO_ROOT / "checkpoints" / "encoder.pth",
    )
)
DEFAULT_OUTPUT = WORKSPACE_ROOT / "Topo" / "figures" / "framework_assets"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def save_rgb(image: Image.Image, path: Path, size: int) -> None:
    image.resize((size, size), Image.Resampling.LANCZOS).save(path)


def normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)


def colored_view(image: Image.Image, variant: int) -> Image.Image:
    """Deterministic colour-only views analogous to the DINO colour branch."""
    if variant == 0:
        result = ImageEnhance.Color(image).enhance(0.88)
        return ImageEnhance.Contrast(result).enhance(1.08)
    result = ImageEnhance.Brightness(image).enhance(1.05)
    result = ImageEnhance.Color(result).enhance(1.12)
    return result.filter(ImageFilter.GaussianBlur(radius=0.35))


def activation_image(feature: torch.Tensor, output_size: int = 256) -> Image.Image:
    activation = feature[0].float().pow(2).mean(dim=0).sqrt().cpu().numpy()
    lo, hi = np.percentile(activation, (2.0, 98.0))
    activation = np.clip((activation - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    rgb = (matplotlib.colormaps["viridis"](activation)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(rgb).resize(
        (output_size, output_size), Image.Resampling.NEAREST
    )


def crop_square(image: Image.Image, box: tuple[int, int, int, int], size: int) -> Image.Image:
    return image.crop(box).resize((size, size), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(args.image).convert("RGB").resize(
        (512, 512), Image.Resampling.LANCZOS
    )
    save_rgb(original, args.output_dir / "dino_input.png", 512)

    # Fixed boxes make the paper figure reproducible while retaining the exact
    # 256/128 global-local resolutions used by train_dino.py.
    global_boxes = ((0, 0, 512, 512), (64, 48, 464, 448))
    local_boxes = (
        (150, 150, 342, 342),
        (230, 150, 410, 330),
        (135, 225, 315, 405),
        (245, 235, 405, 395),
    )
    global_crops = [crop_square(original, box, 256) for box in global_boxes]
    local_crops = [crop_square(original, box, 128) for box in local_boxes]

    for index, crop in enumerate(global_crops, start=1):
        crop.save(args.output_dir / f"global_crop_{index}.png")
        colored_view(crop, 0).save(args.output_dir / f"teacher_global_{index}.png")
        colored_view(crop, 1).save(args.output_dir / f"student_global_{index}.png")
    for index, crop in enumerate(local_crops, start=1):
        crop.save(args.output_dir / f"local_crop_{index}.png")

    encoder = encoder_function()
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    encoder.load_state_dict(state_dict, strict=True)
    encoder.eval()

    shape_manifest: dict[str, list[list[int]]] = {}
    with torch.inference_mode():
        for index, crop in enumerate(global_crops, start=1):
            features = encoder(normalized_tensor(colored_view(crop, 1)))
            shape_manifest[f"global_{index}"] = [list(feature.shape) for feature in features]
            activation_image(features[-1]).save(
                args.output_dir / f"global_feature_{index}.png"
            )
            if index == 1:
                for stage, feature in enumerate(features, start=1):
                    activation_image(feature).save(
                        args.output_dir / f"encoder_stage_{stage}.png"
                    )

        for index, crop in enumerate(local_crops, start=1):
            features = encoder(normalized_tensor(colored_view(crop, 1)))
            shape_manifest[f"local_{index}"] = [list(feature.shape) for feature in features]
            activation_image(features[-1], output_size=128).save(
                args.output_dir / f"local_feature_{index}.png"
            )

    manifest = {
        "source_image": str(args.image),
        "checkpoint": str(args.checkpoint),
        "checkpoint_keys": len(state_dict),
        "feature_shapes": shape_manifest,
    }
    (args.output_dir / "dino_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
