import copy
import os

import torch
import torch.nn.functional as F
import wandb
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import trange

from data.data_loader import loader
from models.Model import ATTNext
from utils.Heads import (
    ProjectionHead,
    SegmentationMHead,
    SegmentationSHead,
    get_teacher_momentum,
    get_teacher_temp,
)
from utils.Loss_dino import DINOLoss
from wandb_init import parser_init, wandb_init


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def using_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using device: {device} ({torch.cuda.get_device_name(device)})")
    else:
        print(f"Using device: {device}")
    return device


def setup_paths(data):
    folder_mapping = {
        # Keep the historical checkpoint directory names so
        # train_ssl_pretrained.py can reconstruct the encoder path.
        "isic_2018_1": "isic_1",
        "kvasir_1": "kvasir_1",
        "ham_1": "ham_1",
        "PH2Dataset": "PH2Dataset",
        "isic_2016_1": "isic_2016_1",
    }
    if data not in folder_mapping:
        raise ValueError(f"Unsupported dataset: {data}")

    output_key = "ML_DATA_OUTPUT" if torch.cuda.is_available() else "ML_DATA_OUTPUT_LOCAL"
    base_path = os.environ.get(output_key) or os.environ.get("ML_DATA_OUTPUT")
    if not base_path:
        raise EnvironmentError(f"{output_key} must point to the checkpoint output directory")

    folder_path = os.path.join(base_path, folder_mapping[data])
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    for student_parameter, teacher_parameter in zip(student.parameters(), teacher.parameters()):
        teacher_parameter.data.mul_(momentum).add_(student_parameter.data, alpha=1.0 - momentum)


def project_global(feature_map, projection_head):
    """Project one encoder feature map to one DINO embedding per image."""
    return projection_head(feature_map.mean(dim=(2, 3)))


def pseudo_segmentation_loss(logits, target, boundary_ignore_radius=2):
    """BCE + soft Dice on reliable pseudo-mask pixels.

    Pseudo-mask boundaries are the least reliable part of masks with roughly
    0.62 IoU. A narrow, configurable boundary band is therefore ignored in
    both loss terms. Set the radius to zero to use every pixel.
    """
    target = (target > 0.5).type_as(logits)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")

    if boundary_ignore_radius > 0:
        kernel_size = 2 * boundary_ignore_radius + 1
        foreground_core = 1.0 - F.max_pool2d(
            1.0 - target,
            kernel_size=kernel_size,
            stride=1,
            padding=boundary_ignore_radius,
        )
        background_core = 1.0 - F.max_pool2d(
            target,
            kernel_size=kernel_size,
            stride=1,
            padding=boundary_ignore_radius,
        )
        valid = (foreground_core + background_core).clamp_(0.0, 1.0)
    else:
        valid = torch.ones_like(target)

    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * valid).sum() / valid.sum().clamp_min(1.0)

    probability = torch.sigmoid(logits) * valid
    reliable_target = target * valid
    dims = (1, 2, 3)
    intersection = (probability * reliable_target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + reliable_target.sum(dim=dims)
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce + dice_loss


def compute_batch_iou(pred_logits, target_masks):
    predictions = (torch.sigmoid(pred_logits) > 0.5).float().flatten(1)
    targets = (target_masks > 0.5).float().flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).sum().item()


def denormalize(image):
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0.0, 1.0)


def main():
    data = os.environ.get("TOPODISTILL_DATASET", "isic_2018_1")
    training_mode, operation, use_pseudo_supervision = "ssl", "train", True
    device = using_device()
    folder_path = setup_paths(data)

    args, result_parts = parser_init("segmentation task", operation, training_mode)
    result_name = "[" + " ".join(result_parts) + f"]_segloss_{use_pseudo_supervision}_{data}"
    config = wandb_init(
        os.environ.get("WANDB_API_KEY"),
        os.environ.get("WANDB_DIR"),
        args,
        data,
        use_pseudo_supervision,
    )

    def create_loader(loader_operation):
        return loader(
            loader_operation,
            args.mode,
            args.sslmode_modelname,
            args.bsize,
            args.workers,
            args.imsize,
            args.cutoutpr,
            args.cutoutbox,
            args.shuffle if loader_operation == "train" else False,
            args.sratio,
            data,
        )

    train_loader = create_loader("train")
    val_loader = create_loader("validation")

    model = ATTNext(args.mode).to(device)
    student = model.encoder
    teacher = copy.deepcopy(student).to(device)
    teacher.requires_grad_(False)

    student_head = ProjectionHead().to(device)
    teacher_head = copy.deepcopy(student_head).to(device)
    teacher_head.requires_grad_(False)
    segmentation_head = SegmentationSHead().to(device)

    # This head is a detached online probe. It never sends gradients into the
    # encoder and its ground-truth IoU is never used for checkpoint selection.
    enable_gt_monitor = env_bool("TOPODISTILL_ENABLE_GT_MONITOR", False)
    monitor_head = SegmentationMHead().to(device) if enable_gt_monitor else None

    dino_loss_fn = DINOLoss()
    trainable_parameters = (
        list(student.parameters())
        + list(student_head.parameters())
        + list(segmentation_head.parameters())
    )
    optimizer = AdamW(trainable_parameters, lr=config["learningrate"], weight_decay=0.05)
    scheduler = CosineAnnealingLR(
        optimizer,
        config["epochs"],
        eta_min=config["learningrate"] / 10,
    )
    monitor_optimizer = (
        AdamW(monitor_head.parameters(), lr=config["learningrate"])
        if monitor_head is not None
        else None
    )

    pseudo_weight_max = float(os.environ.get("TOPODISTILL_PSEUDO_WEIGHT", "1.0"))
    pseudo_warmup_epochs = max(
        1, int(os.environ.get("TOPODISTILL_PSEUDO_WARMUP_EPOCHS", "20"))
    )
    boundary_ignore_radius = max(
        0, int(os.environ.get("TOPODISTILL_BOUNDARY_IGNORE_RADIUS", "2"))
    )
    wandb_image_interval = max(
        0, int(os.environ.get("TOPODISTILL_WANDB_IMAGE_INTERVAL", "50"))
    )
    wandb_images_seen = 0

    checkpoint_path = os.path.join(folder_path, student.__class__.__name__ + result_name)
    print(
        f"Crops: teacher=2 global, student=2 global + 4 local. "
        f"Training images: {len(train_loader.dataset)}"
    )
    print(f"Best encoder checkpoint: {checkpoint_path}")
    print(
        f"Pseudo loss: BCE+Dice, boundary ignore radius={boundary_ignore_radius}, "
        f"maximum weight={pseudo_weight_max}, warm-up={pseudo_warmup_epochs} epochs"
    )

    def maybe_log_training_image(
        paths,
        original_images,
        student_views,
        pseudo_targets,
        segmentation_logits,
        previous_count,
        current_count,
    ):
        if wandb_image_interval == 0:
            return
        if previous_count // wandb_image_interval == current_count // wandb_image_interval:
            return

        sample_index = 0
        original = original_images[sample_index].permute(1, 2, 0).cpu().numpy()
        crop = (
            denormalize(student_views[0][sample_index].detach())
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        pseudo = pseudo_targets[0][sample_index].permute(1, 2, 0).detach().cpu().numpy()
        prediction = (
            torch.sigmoid(segmentation_logits[0][sample_index])
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )
        wandb.log(
            {
                "SSL sample/original": wandb.Image(original, caption=str(paths[sample_index])),
                "SSL sample/global crop": wandb.Image(crop),
                "SSL sample/pseudo target": wandb.Image(pseudo),
                "SSL sample/auxiliary probability": wandb.Image(prediction),
                "SSL sample/images seen": current_count,
            }
        )

    def run_epoch(data_loader, epoch_index, momentum, pseudo_weight, training):
        nonlocal wandb_images_seen
        student.train(training)
        student_head.train(training)
        segmentation_head.train(training)
        teacher.eval()
        teacher_head.eval()
        if monitor_head is not None:
            monitor_head.train(training)

        totals = {"combined": 0.0, "dino": 0.0, "pseudo": 0.0, "monitor": 0.0}
        monitor_iou_sum = 0.0
        sample_count = 0
        teacher_temperature = get_teacher_temp(epoch_index)

        with torch.set_grad_enabled(training):
            for (
                original_images,
                paths,
                cropped_real_masks,
                student_views,
                teacher_views,
                pseudo_masks,
            ) in data_loader:
                student_views = [view.to(device, non_blocking=True) for view in student_views]
                teacher_views = [view.to(device, non_blocking=True) for view in teacher_views]
                pseudo_targets = [mask.to(device, non_blocking=True) for mask in pseudo_masks]

                # Standard multi-crop DINO compares global image embeddings.
                # Spatial information is supervised separately by the aligned
                # pseudo masks on both student global views.
                student_features = [student(view)[3] for view in student_views]
                student_outputs = [project_global(feature, student_head) for feature in student_features]
                with torch.no_grad():
                    teacher_features = [teacher(view)[3] for view in teacher_views]
                    teacher_outputs = [project_global(feature, teacher_head) for feature in teacher_features]

                dino_loss = dino_loss_fn(
                    student_outputs,
                    teacher_outputs,
                    teacher_temperature,
                    exclude_matching_views=True,
                    update_center=training,
                )
                segmentation_logits = [
                    segmentation_head(student_features[index])
                    for index in range(len(pseudo_targets))
                ]
                pseudo_loss = torch.stack(
                    [
                        pseudo_segmentation_loss(logits, target, boundary_ignore_radius)
                        for logits, target in zip(segmentation_logits, pseudo_targets)
                    ]
                ).mean()
                combined_loss = dino_loss + pseudo_weight * pseudo_loss

                if training:
                    optimizer.zero_grad(set_to_none=True)
                    combined_loss.backward()
                    clip_grad_norm_(trainable_parameters, max_norm=3.0)
                    optimizer.step()
                    update_teacher(student, teacher, momentum)
                    update_teacher(student_head, teacher_head, momentum)

                batch_size = student_views[0].shape[0]
                sample_count += batch_size
                totals["combined"] += combined_loss.item() * batch_size
                totals["dino"] += dino_loss.item() * batch_size
                totals["pseudo"] += pseudo_loss.item() * batch_size

                if monitor_head is not None:
                    real_target = cropped_real_masks[0].to(device, non_blocking=True)
                    monitor_logits = monitor_head(student_features[0].detach())
                    monitor_loss = F.binary_cross_entropy_with_logits(monitor_logits, real_target)
                    if training:
                        monitor_optimizer.zero_grad(set_to_none=True)
                        monitor_loss.backward()
                        monitor_optimizer.step()
                    totals["monitor"] += monitor_loss.item() * batch_size
                    monitor_iou_sum += compute_batch_iou(monitor_logits, real_target)

                if training:
                    previous_count = wandb_images_seen
                    wandb_images_seen += batch_size
                    maybe_log_training_image(
                        paths,
                        original_images,
                        student_views,
                        pseudo_targets,
                        segmentation_logits,
                        previous_count,
                        wandb_images_seen,
                    )

        if sample_count == 0:
            raise RuntimeError("The data loader yielded no samples")
        metrics = {name: value / sample_count for name, value in totals.items()}
        metrics["monitor_iou"] = (
            monitor_iou_sum / sample_count if monitor_head is not None else float("nan")
        )
        return metrics

    best_selection_loss = float("inf")
    for epoch in trange(config["epochs"], desc="Epochs"):
        momentum = get_teacher_momentum(epoch, config["epochs"])
        pseudo_weight = pseudo_weight_max * min(1.0, (epoch + 1) / pseudo_warmup_epochs)

        train_metrics = run_epoch(
            train_loader,
            epoch,
            momentum,
            pseudo_weight,
            training=True,
        )
        val_metrics = run_epoch(
            val_loader,
            epoch,
            momentum,
            pseudo_weight,
            training=False,
        )
        scheduler.step()

        # Use a label-free criterion with the final pseudo-loss weight. This
        # avoids favouring early warm-up epochs and avoids ground-truth-based
        # encoder selection.
        selection_loss = val_metrics["dino"] + pseudo_weight_max * val_metrics["pseudo"]
        log_payload = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "teacher_momentum": momentum,
            "pseudo_weight": pseudo_weight,
            "train/combined_loss": train_metrics["combined"],
            "train/dino_loss": train_metrics["dino"],
            "train/pseudo_loss": train_metrics["pseudo"],
            "validation/combined_loss": val_metrics["combined"],
            "validation/dino_loss": val_metrics["dino"],
            "validation/pseudo_loss": val_metrics["pseudo"],
            "validation/selection_loss": selection_loss,
        }
        if monitor_head is not None:
            log_payload.update(
                {
                    "train/gt_monitor_loss": train_metrics["monitor"],
                    "validation/gt_monitor_loss": val_metrics["monitor"],
                    "validation/gt_monitor_iou": val_metrics["monitor_iou"],
                }
            )
        wandb.log(log_payload)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"train={train_metrics['combined']:.4f} "
            f"(DINO={train_metrics['dino']:.4f}, pseudo={train_metrics['pseudo']:.4f}) | "
            f"val_selection={selection_loss:.4f}"
        )
        if monitor_head is not None:
            print(f"Detached GT monitor IoU: {val_metrics['monitor_iou']:.4f}")

        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            torch.save(student.state_dict(), checkpoint_path)
            print(
                "Best encoder saved using label-free validation objective: "
                f"{best_selection_loss:.4f}"
            )

    wandb.finish()


if __name__ == "__main__":
    main()
