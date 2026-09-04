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


def compute_binary_mask_iou(pred_masks, target_masks):
    predictions = (pred_masks > 0.5).float().flatten(1)
    targets = (target_masks > 0.5).float().flatten(1)
    intersection = (predictions * targets).sum(dim=1)
    union = predictions.sum(dim=1) + targets.sum(dim=1) - intersection
    return ((intersection + 1e-6) / (union + 1e-6)).sum().item()


def compute_batch_iou(pred_logits, target_masks):
    return compute_binary_mask_iou(torch.sigmoid(pred_logits), target_masks)


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
    # All curves and qualitative snapshots use epoch as their x-axis.
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("validation/*", step_metric="epoch")
    wandb.define_metric("schedule/*", step_metric="epoch")
    wandb.define_metric("snapshot/*", step_metric="epoch")
    wandb.define_metric("samples/*", step_metric="epoch")

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
    wandb_visualization_epoch_interval = max(
        0, int(os.environ.get("TOPODISTILL_WANDB_VIS_EPOCH_INTERVAL", "25"))
    )

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

    def maybe_log_training_snapshot(
        paths,
        original_images,
        student_views,
        teacher_views,
        student_features,
        student_outputs,
        teacher_outputs,
        pseudo_targets,
        segmentation_logits,
        real_targets,
        scalar_metrics,
        epoch_index,
        batch_index,
    ):
        if wandb_visualization_epoch_interval == 0:
            return
        epoch_number = epoch_index + 1
        if epoch_number % wandb_visualization_epoch_interval != 0 or batch_index != 0:
            return

        sample_index = 0
        def crop_image(view, caption):
            image = (
                denormalize(view[sample_index].detach())
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            return wandb.Image(image, caption=caption)

        def mask_image(mask, caption):
            image = mask[sample_index].detach().squeeze(0).cpu().numpy()
            return wandb.Image(image, caption=caption)

        student_global = [
            crop_image(view, f"student global {index + 1}")
            for index, view in enumerate(student_views[:2])
        ]
        student_local = [
            crop_image(view, f"student local {index + 1}")
            for index, view in enumerate(student_views[2:])
        ]
        teacher_global = [
            crop_image(view, f"teacher global {index + 1}")
            for index, view in enumerate(teacher_views)
        ]
        pseudo_images = [
            mask_image(mask, f"pseudo target for global {index + 1}")
            for index, mask in enumerate(pseudo_targets)
        ]
        probability_images = [
            mask_image(torch.sigmoid(logits), f"aux probability for global {index + 1}")
            for index, logits in enumerate(segmentation_logits)
        ]
        binary_prediction_images = [
            mask_image(
                (torch.sigmoid(logits) > 0.5).float(),
                f"aux binary prediction for global {index + 1}",
            )
            for index, logits in enumerate(segmentation_logits)
        ]

        original = original_images[sample_index].permute(1, 2, 0).cpu().numpy()
        with torch.no_grad():
            student_probabilities = [
                F.softmax(output / dino_loss_fn.student_temp, dim=-1)
                for output in student_outputs
            ]
            teacher_probabilities = [
                F.softmax(
                    (output - dino_loss_fn.center.to(output.device))
                    / scalar_metrics["snapshot/teacher_temperature"],
                    dim=-1,
                )
                for output in teacher_outputs
            ]
            student_entropy = torch.stack(
                [
                    -(probability * probability.clamp_min(1e-12).log())
                    .sum(dim=-1)
                    .mean()
                    for probability in student_probabilities
                ]
            ).mean()
            teacher_entropy = torch.stack(
                [
                    -(probability * probability.clamp_min(1e-12).log())
                    .sum(dim=-1)
                    .mean()
                    for probability in teacher_probabilities
                ]
            ).mean()
            normalized_embedding = F.normalize(
                student_features[0].mean(dim=(2, 3)), dim=-1
            )
            embedding_std = normalized_embedding.std(dim=0, unbiased=False).mean()
            target_foreground = torch.stack(
                [target.float().mean() for target in pseudo_targets]
            ).mean()
            predicted_foreground = torch.stack(
                [
                    (torch.sigmoid(logits) > 0.5).float().mean()
                    for logits in segmentation_logits
                ]
            ).mean()

        scalar_metrics.update(
            {
                "snapshot/student_output_entropy": student_entropy.item(),
                "snapshot/teacher_output_entropy": teacher_entropy.item(),
                "snapshot/student_embedding_std": embedding_std.item(),
                "snapshot/dino_center_norm": dino_loss_fn.center.norm().item(),
                "snapshot/pseudo_foreground_fraction": target_foreground.item(),
                "snapshot/predicted_foreground_fraction": predicted_foreground.item(),
            }
        )
        log_payload = {
            "epoch": epoch_number,
            "samples/original": wandb.Image(
                original,
                caption=f"{paths[sample_index]} | epoch={epoch_number}",
            ),
            "samples/student_global_crops": student_global,
            "samples/student_local_crops": student_local,
            "samples/teacher_global_crops": teacher_global,
            "samples/pseudo_targets": pseudo_images,
            "samples/auxiliary_probabilities": probability_images,
            "samples/auxiliary_binary_predictions": binary_prediction_images,
        }
        if real_targets is not None:
            log_payload["samples/real_masks_diagnostic"] = [
                mask_image(mask, f"real mask for global {index + 1}")
                for index, mask in enumerate(real_targets)
            ]
        log_payload.update(scalar_metrics)
        wandb.log(log_payload)

    def run_epoch(data_loader, epoch_index, momentum, pseudo_weight, training):
        student.train(training)
        student_head.train(training)
        segmentation_head.train(training)
        teacher.eval()
        teacher_head.eval()
        if monitor_head is not None:
            monitor_head.train(training)

        totals = {
            "combined": 0.0,
            "dino": 0.0,
            "pseudo": 0.0,
            "pseudo_fit_iou": 0.0,
            "monitor": 0.0,
        }
        monitor_iou_sum = 0.0
        sample_count = 0
        teacher_temperature = get_teacher_temp(epoch_index)

        with torch.set_grad_enabled(training):
            for batch_index, (
                    original_images,
                    paths,
                    cropped_real_masks,
                    student_views,
                    teacher_views,
                    pseudo_masks,
            ) in enumerate(data_loader):
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

                gradient_norm = float("nan")
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    combined_loss.backward()
                    gradient_norm = float(
                        clip_grad_norm_(trainable_parameters, max_norm=3.0).item()
                    )
                    optimizer.step()
                    update_teacher(student, teacher, momentum)
                    update_teacher(student_head, teacher_head, momentum)

                batch_size = student_views[0].shape[0]
                sample_count += batch_size
                totals["combined"] += combined_loss.item() * batch_size
                totals["dino"] += dino_loss.item() * batch_size
                totals["pseudo"] += pseudo_loss.item() * batch_size

                pseudo_fit_iou = sum(
                    compute_batch_iou(logits, target)
                    for logits, target in zip(segmentation_logits, pseudo_targets)
                ) / (batch_size * len(pseudo_targets))
                totals["pseudo_fit_iou"] += pseudo_fit_iou * batch_size

                real_targets = None
                monitor_loss_value = float("nan")
                monitor_iou_value = float("nan")
                pseudo_real_iou = float("nan")
                if monitor_head is not None:
                    real_targets = [
                        mask.to(device, non_blocking=True) for mask in cropped_real_masks
                    ]
                    real_target = real_targets[0]
                    monitor_logits = monitor_head(student_features[0].detach())
                    monitor_loss = F.binary_cross_entropy_with_logits(monitor_logits, real_target)
                    monitor_loss_value = monitor_loss.item()
                    if training:
                        monitor_optimizer.zero_grad(set_to_none=True)
                        monitor_loss.backward()
                        monitor_optimizer.step()
                    totals["monitor"] += monitor_loss.item() * batch_size
                    monitor_iou_sum_batch = compute_batch_iou(monitor_logits, real_target)
                    monitor_iou_sum += monitor_iou_sum_batch
                    monitor_iou_value = monitor_iou_sum_batch / batch_size
                    pseudo_real_iou = sum(
                        compute_binary_mask_iou(pseudo, real)
                        for pseudo, real in zip(pseudo_targets, real_targets)
                    ) / (batch_size * len(pseudo_targets))

                if training:
                    running_denominator = max(sample_count, 1)
                    scalar_metrics = {
                        "snapshot/combined_loss": combined_loss.item(),
                        "snapshot/dino_loss": dino_loss.item(),
                        "snapshot/pseudo_loss": pseudo_loss.item(),
                        "snapshot/weighted_pseudo_loss": pseudo_weight
                        * pseudo_loss.item(),
                        "snapshot/pseudo_contribution_fraction": (
                            pseudo_weight * pseudo_loss.item()
                        )
                        / max(combined_loss.item(), 1e-12),
                        "snapshot/pseudo_fit_iou": pseudo_fit_iou,
                        "snapshot/running_combined_loss": totals["combined"]
                        / running_denominator,
                        "snapshot/running_dino_loss": totals["dino"]
                        / running_denominator,
                        "snapshot/running_pseudo_loss": totals["pseudo"]
                        / running_denominator,
                        "snapshot/pseudo_weight": pseudo_weight,
                        "snapshot/dino_weight": 1.0,
                        "snapshot/learning_rate": optimizer.param_groups[0]["lr"],
                        "snapshot/teacher_momentum": momentum,
                        "snapshot/teacher_temperature": teacher_temperature,
                        "snapshot/gradient_norm_before_clip": gradient_norm,
                    }
                    if monitor_head is not None:
                        scalar_metrics["snapshot/gt_monitor_loss"] = monitor_loss_value
                        scalar_metrics["snapshot/gt_monitor_iou"] = monitor_iou_value
                        scalar_metrics["snapshot/pseudo_vs_real_iou"] = pseudo_real_iou

                    maybe_log_training_snapshot(
                        paths,
                        original_images,
                        student_views,
                        teacher_views,
                        student_features,
                        student_outputs,
                        teacher_outputs,
                        pseudo_targets,
                        segmentation_logits,
                        real_targets,
                        scalar_metrics,
                        epoch_index,
                        batch_index,
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
            "schedule/learning_rate": optimizer.param_groups[0]["lr"],
            "schedule/teacher_momentum": momentum,
            "schedule/teacher_temperature": get_teacher_temp(epoch),
            "schedule/pseudo_weight": pseudo_weight,
            "schedule/dino_weight": 1.0,
            "train/combined_loss": train_metrics["combined"],
            "train/dino_loss": train_metrics["dino"],
            "train/pseudo_loss": train_metrics["pseudo"],
            "train/weighted_pseudo_loss": pseudo_weight * train_metrics["pseudo"],
            "train/pseudo_contribution_fraction": (
                pseudo_weight * train_metrics["pseudo"]
            )
            / max(train_metrics["combined"], 1e-12),
            "train/pseudo_fit_iou": train_metrics["pseudo_fit_iou"],
            "validation/combined_loss": val_metrics["combined"],
            "validation/dino_loss": val_metrics["dino"],
            "validation/pseudo_loss": val_metrics["pseudo"],
            "validation/weighted_pseudo_loss": pseudo_weight
            * val_metrics["pseudo"],
            "validation/pseudo_contribution_fraction": (
                pseudo_weight * val_metrics["pseudo"]
            )
            / max(val_metrics["combined"], 1e-12),
            "validation/pseudo_fit_iou": val_metrics["pseudo_fit_iou"],
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
