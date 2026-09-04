"""Train ATTNext directly with topological pseudo masks as segmentation labels.

Examples
--------
python train_pseudo_supervised.py --dataset isic_2018_1
python train_pseudo_supervised.py --dataset PH2Dataset --epochs 300
python train_pseudo_supervised.py --dataset isic_2016_1 --validation-target pseudo

The dataset root is read from ML_DATA_ROOT. Pseudo masks are expected in
<dataset>/<split>/pmasks by default; override the name with
TOPODISTILL_PMASK_SUBDIR.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm import trange

from models.Model import ATTNext


DATASETS = {
    "isic_2018_1": ("isic_2018_1", ".jpg", ".png"),
    "isic_2016_1": ("isic_2016_1", ".jpg", ".png"),
    "PH2Dataset": ("PH2Dataset", ".jpeg", ".jpeg"),
    "kvasir_1": ("kvasir_1", ".jpg", ".jpg"),
    "ham_1": ("HAM10000_1", ".jpg", ".png"),
}


def canonical_key(path):
    key = Path(path).stem.lower()
    for suffix in ("_segmentation", "_mask", "_lesion"):
        if key.endswith(suffix):
            key = key[: -len(suffix)]
    return key


def indexed_files(directory, extension):
    files = sorted(Path(directory).glob(f"*{extension}"))
    result = {}
    for path in files:
        key = canonical_key(path)
        if key in result:
            raise RuntimeError(f"Duplicate sample key {key!r} in {directory}")
        result[key] = path
    return result


def aligned_split(dataset_root, split, image_extension, mask_extension, pseudo_subdir):
    images = indexed_files(dataset_root / split / "images", image_extension)
    masks = indexed_files(dataset_root / split / "masks", mask_extension)
    pseudo_masks = indexed_files(dataset_root / split / pseudo_subdir, ".png")
    common = sorted(set(images) & set(masks) & set(pseudo_masks))
    if not common:
        raise RuntimeError(
            f"No aligned image/mask/pseudo-mask triples found in {dataset_root / split}. "
            f"Expected pseudo masks in {pseudo_subdir!r}."
        )
    missing = {
        "image": sorted((set(masks) | set(pseudo_masks)) - set(images)),
        "mask": sorted((set(images) | set(pseudo_masks)) - set(masks)),
        "pseudo": sorted((set(images) | set(masks)) - set(pseudo_masks)),
    }
    if any(missing.values()):
        details = ", ".join(f"{name}={len(keys)} missing" for name, keys in missing.items())
        raise RuntimeError(f"Unaligned files in {dataset_root / split}: {details}")
    return [(images[key], masks[key], pseudo_masks[key]) for key in common]


class PseudoSupervisedDataset(Dataset):
    def __init__(self, samples, image_size, training, target_kind):
        self.samples = samples
        self.image_size = (image_size, image_size)
        self.training = training
        self.target_kind = target_kind

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def read_image(path):
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    @staticmethod
    def read_mask(path):
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return (torch.from_numpy(array).unsqueeze(0) > 0.5).float()

    def __getitem__(self, index):
        image_path, real_mask_path, pseudo_mask_path = self.samples[index]
        image = TF.resize(self.read_image(image_path), self.image_size, antialias=True)
        real_mask = TF.resize(
            self.read_mask(real_mask_path),
            self.image_size,
            interpolation=TF.InterpolationMode.NEAREST,
        )
        pseudo_mask = TF.resize(
            self.read_mask(pseudo_mask_path),
            self.image_size,
            interpolation=TF.InterpolationMode.NEAREST,
        )

        target = pseudo_mask if self.target_kind == "pseudo" else real_mask
        if self.training:
            if torch.rand(()) < 0.5:
                image = TF.hflip(image)
                target = TF.hflip(target)
                real_mask = TF.hflip(real_mask)
            if torch.rand(()) < 0.5:
                image = TF.vflip(image)
                target = TF.vflip(target)
                real_mask = TF.vflip(real_mask)
            rotations = int(torch.randint(0, 4, ()).item())
            if rotations:
                image = torch.rot90(image, rotations, dims=(-2, -1))
                target = torch.rot90(target, rotations, dims=(-2, -1))
                real_mask = torch.rot90(real_mask, rotations, dims=(-2, -1))

        return image, target, real_mask, str(image_path)


def reliable_pseudo_loss(logits, target, boundary_ignore_radius):
    target = (target > 0.5).type_as(logits)
    if boundary_ignore_radius > 0:
        kernel = 2 * boundary_ignore_radius + 1
        foreground_core = 1.0 - F.max_pool2d(
            1.0 - target, kernel, stride=1, padding=boundary_ignore_radius
        )
        background_core = 1.0 - F.max_pool2d(
            target, kernel, stride=1, padding=boundary_ignore_radius
        )
        valid = (foreground_core + background_core).clamp_(0.0, 1.0)
    else:
        valid = torch.ones_like(target)

    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * valid).sum() / valid.sum().clamp_min(1.0)
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice


def batch_metrics(logits, real_masks):
    prediction = (torch.sigmoid(logits) > 0.5).float().flatten(1)
    target = (real_masks > 0.5).float().flatten(1)
    intersection = (prediction * target).sum(dim=1)
    union = prediction.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2.0 * intersection + 1e-6) / (
        prediction.sum(dim=1) + target.sum(dim=1) + 1e-6
    )
    return iou.sum().item(), dice.sum().item()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default=os.environ.get("TOPODISTILL_DATASET", "isic_2018_1"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--boundary-ignore-radius", type=int, default=2)
    parser.add_argument(
        "--validation-target",
        choices=("real", "pseudo"),
        default="real",
        help="Target used for validation loss/checkpoint selection. Metrics are always against real masks.",
    )
    parser.add_argument("--seed", type=int, default=932)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data_root = os.environ.get("ML_DATA_ROOT")
    if not data_root:
        raise EnvironmentError("ML_DATA_ROOT must point to the directory containing datasets")
    output_root = os.environ.get(
        "ML_DATA_OUTPUT" if torch.cuda.is_available() else "ML_DATA_OUTPUT_LOCAL"
    ) or os.environ.get("ML_DATA_OUTPUT")
    if not output_root:
        raise EnvironmentError("ML_DATA_OUTPUT (or ML_DATA_OUTPUT_LOCAL) is required")

    folder, image_extension, mask_extension = DATASETS[args.dataset]
    dataset_root = Path(data_root) / folder
    pseudo_subdir = os.environ.get("TOPODISTILL_PMASK_SUBDIR", "pmasks")
    train_samples = aligned_split(
        dataset_root, "train", image_extension, mask_extension, pseudo_subdir
    )
    val_samples = aligned_split(
        dataset_root, "val", image_extension, mask_extension, pseudo_subdir
    )
    test_samples = aligned_split(
        dataset_root, "test", image_extension, mask_extension, pseudo_subdir
    )

    train_dataset = PseudoSupervisedDataset(
        train_samples, args.image_size, training=True, target_kind="pseudo"
    )
    val_dataset = PseudoSupervisedDataset(
        val_samples,
        args.image_size,
        training=False,
        target_kind=args.validation_target,
    )
    test_dataset = PseudoSupervisedDataset(
        test_samples, args.image_size, training=False, target_kind="real"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ATTNext("supervised").to(device)
    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate / 10.0
    )

    checkpoint_dir = Path(output_root) / folder
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "ATTNext_direct_pseudo_supervised.pt"

    wandb_directory = Path(os.environ.get("WANDB_DIR") or Path.cwd() / "wandb")
    wandb_directory.mkdir(parents=True, exist_ok=True)
    wandb.init(
        project="TopoDistill-Pseudo-Supervised",
        dir=str(wandb_directory),
        name=f"direct-pseudo-{args.dataset}",
        config=vars(args),
    )

    def run_epoch(data_loader, training):
        model.train(training)
        loss_sum = iou_sum = dice_sum = 0.0
        sample_count = 0
        with torch.set_grad_enabled(training):
            for images, targets, real_masks, _ in data_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                real_masks = real_masks.to(device, non_blocking=True)
                logits = model(images)
                loss = reliable_pseudo_loss(
                    logits, targets, max(0, args.boundary_ignore_radius)
                )
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                batch_size = images.shape[0]
                sample_count += batch_size
                loss_sum += loss.item() * batch_size
                batch_iou, batch_dice = batch_metrics(logits, real_masks)
                iou_sum += batch_iou
                dice_sum += batch_dice
        return {
            "loss": loss_sum / sample_count,
            "iou": iou_sum / sample_count,
            "dice": dice_sum / sample_count,
        }

    best_value = float("inf") if args.validation_target == "pseudo" else -float("inf")
    for epoch in trange(args.epochs, desc="Direct pseudo-supervised training"):
        train_metrics = run_epoch(train_loader, training=True)
        val_metrics = run_epoch(val_loader, training=False)
        scheduler.step()

        candidate = (
            val_metrics["loss"]
            if args.validation_target == "pseudo"
            else val_metrics["iou"]
        )
        improved = (
            candidate < best_value
            if args.validation_target == "pseudo"
            else candidate > best_value
        )
        if improved:
            best_value = candidate
            torch.save(model.state_dict(), checkpoint_path)

        wandb.log(
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train/loss": train_metrics["loss"],
                "train/iou_against_real": train_metrics["iou"],
                "train/dice_against_real": train_metrics["dice"],
                "validation/loss": val_metrics["loss"],
                "validation/iou_against_real": val_metrics["iou"],
                "validation/dice_against_real": val_metrics["dice"],
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | train loss={train_metrics['loss']:.4f} | "
            f"val IoU={val_metrics['iou']:.4f}, Dice={val_metrics['dice']:.4f}"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_metrics = run_epoch(test_loader, training=False)
    wandb.log(
        {
            "test/loss": test_metrics["loss"],
            "test/iou_against_real": test_metrics["iou"],
            "test/dice_against_real": test_metrics["dice"],
        }
    )
    print(
        f"Best checkpoint: {checkpoint_path}\n"
        f"Test against real masks: IoU={test_metrics['iou']:.4f}, "
        f"Dice={test_metrics['dice']:.4f}, loss={test_metrics['loss']:.4f}"
    )
    wandb.finish()


if __name__ == "__main__":
    main()
