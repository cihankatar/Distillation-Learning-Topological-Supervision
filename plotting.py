import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
from utils.iou_dice import iou_and_dice
import torch


def plot(
    names,
    image,
    img_tensor,
    clean_img,
    kmeans_mask,
    grabcut_mask,
    watershed_mask,
    mask_h0,
    random_walker_mask,
    morphological_mask,
    otsu,
    cc_mask,
    real_mask,
    bests,
    pi,
    threshold,
    *,
    raw_h1_mask=None,
    h0_noise_mask=None,
    spatial_noise_mask=None,
    spatial_clean_mask=None,
    combined_mask=None,
    objects_removed_mask=None,
    morph_object_size_threshold=None,
    h1_anchor=None,
    h0_persistence_threshold=None,
    background_skin_rgb=None,
    show=False,
    max_persistence_points=5000,
):
    # --- Compute metrics ---
    masks = {
        "KMeans": kmeans_mask,
        "GrabCut": grabcut_mask,
        "Watershed": watershed_mask,
        "Random Walker": random_walker_mask,
        "Morphological": morphological_mask,
        "Topological": cc_mask,
        "Image": image,
        "Otsu": otsu,
        "Raw_H1": raw_h1_mask,
        "H0_Noise": h0_noise_mask,
        "Mask_H0": mask_h0,
        "Spatial_Noise": spatial_noise_mask,
        "Spatial_Clean": spatial_clean_mask,
        "Combined": combined_mask,
        "Objects_Removed": objects_removed_mask,
    }
    
    scores = {}
    for name, m in masks.items():
        # Maske veya gerçek maske None ise hesaplama yapmadan sıfır ata
        if m is None or real_mask is None:
            scores[name] = (0.0, 0.0)
            continue
        iou, dice = iou_and_dice(m, real_mask)
        scores[name] = (iou, dice)

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle(f"Results for: {names}", fontsize=16)
    axes = axes.ravel()

    # Görselleri None hatası almadan çizdirmek için yardımcı fonksiyon
    def safe_imshow(ax, img, title, is_tensor=False, cmap='gray'):
        if img is not None:
            if is_tensor:
                # Tensör ise permute işlemini yap (gerekirse .cpu().numpy() eklenebilir)
                ax.imshow(img.permute(1, 2, 0).cpu().numpy() if hasattr(img, 'cpu') else img.permute(1, 2, 0))
            else:
                ax.imshow(img, cmap=cmap)
        else:
            # Kullanılmayan yöntemler bu ablation görünümünde sessizce gizlenir.
            ax.set_visible(False)
            return
        
        ax.set_title(title)
        ax.axis("off")

    # 1) Orijinal
    safe_imshow(axes[0], img_tensor, "Original Image", is_tensor=True, cmap=None)
    if background_skin_rgb is not None and img_tensor is not None:
        height, width = img_tensor.shape[-2:]
        swatch_size = 32
        margin = 3
        axes[0].add_patch(
            Rectangle(
                (width - swatch_size - margin, height - swatch_size - margin),
                swatch_size,
                swatch_size,
                facecolor=np.clip(background_skin_rgb, 0.0, 1.0),
                edgecolor="white",
                linewidth=1.5,
            )
        )
        axes[0].text(
            width - swatch_size - margin,
            height - swatch_size - margin - 3,
            "skin",
            color="white",
            fontsize=8,
            ha="left",
            va="bottom",
        )

    # 2) Clean Image
    safe_imshow(axes[1], clean_img, "Clean Image")

    # 3) Only the selected scalar field is shown. The separate six-field
    # comparison figure was intentionally removed.
    safe_imshow(axes[2], image, "Selected scalar field")

    # 4) Random Walker
    iou, dice = scores.get("Random Walker", (0.0, 0.0))
    safe_imshow(axes[3], random_walker_mask, f"Random Walker\nIoU:{iou:.2f}, Dice:{dice:.2f}")

    # 5) Morphological
    iou, dice = scores.get("Morphological", (0.0, 0.0))
    safe_imshow(axes[4], morphological_mask, f"Morphological_Chan_Vese\nIoU:{iou:.2f}, Dice:{dice:.2f}")

    # 6) Otsu
    iou, dice = scores.get("Otsu", (0.0, 0.0))
    safe_imshow(axes[5], otsu, f"Otsu\nIoU:{iou:.2f}, Dice:{dice:.2f}")

    # 7) H1 threshold sonucunun ham hali: H0/uzamsal/morfolojik işlem yok.
    iou, dice = scores.get("Raw_H1", (0.0, 0.0))
    safe_imshow(
        axes[6],
        raw_h1_mask,
        f"Raw H1 Mask\nIoU:{iou:.2f}, Dice:{dice:.2f}",
    )
    if h1_anchor is not None and len(h1_anchor) >= 2:
        axes[6].plot(
            float(h1_anchor[1]),
            float(h1_anchor[0]),
            marker="x",
            color="red",
            markersize=9,
            markeredgewidth=2,
        )

    # 8) Yalnız H0 persistence filtresinin gürültü saydığı bölgeler.
    if h0_persistence_threshold is None:
        h0_threshold_text = "N/A"
    elif np.isfinite(h0_persistence_threshold):
        h0_threshold_text = f"{float(h0_persistence_threshold):.2f}"
    else:
        h0_threshold_text = "no split"
    safe_imshow(
        axes[7],
        h0_noise_mask,
        f"Regions rejected by H0\npersistence threshold={h0_threshold_text}",
    )

    # 9) H1 ölüm noktasından uzak ve sınırla bağlı olduğu için elenenler.
    safe_imshow(
        axes[8],
        spatial_noise_mask,
        "Far border components rejected by H1",
    )

    # 10) Morfolojik işlem öncesi H0 ve H1-uzamsal birleşimi.
    iou, dice = scores.get("Combined", (0.0, 0.0))
    safe_imshow(
        axes[9],
        combined_mask,
        f"H1 + H0 + spatial\nIoU:{iou:.2f}, Dice:{dice:.2f}",
    )

    # 11) Her iki morfolojik işlemden sonraki tek nihai sonuç.
    iou, dice = scores.get("Objects_Removed", scores.get("Topological", (0.0, 0.0)))
    object_threshold_text = (
        str(int(morph_object_size_threshold))
        if morph_object_size_threshold is not None
        else "N/A"
    )
    safe_imshow(
        axes[10],
        objects_removed_mask if objects_removed_mask is not None else cc_mask,
        "Final morphological cleanup\n"
        f"object threshold={object_threshold_text}\n"
        f"IoU:{iou:.2f}, Dice:{dice:.2f}",
    )

    # 12) Real Mask
    safe_imshow(axes[11], real_mask, "Real Mask")

    # 13) Persistence Diagram
    if pi is not None:
        try:
            finite_diagram_values = []
            for features, color in zip([0, 1], ["blue", "red"]):
                persistence_pairs = pi[features][1]
                finite_mask = torch.isfinite(persistence_pairs).all(dim=1)
                persistence_pairs = persistence_pairs[finite_mask]

                if len(persistence_pairs) == 0:
                    continue

                if len(persistence_pairs) > max_persistence_points:
                    sample_indices = torch.linspace(
                        0,
                        len(persistence_pairs) - 1,
                        max_persistence_points,
                        device=persistence_pairs.device,
                    ).long()
                    persistence_pairs = persistence_pairs[sample_indices]

                births = persistence_pairs[:, 0]
                deaths = persistence_pairs[:, 1]
                finite_diagram_values.extend([births, deaths])

                axes[12].scatter(
                    births.detach().cpu().numpy(),
                    deaths.detach().cpu().numpy(),
                    s=10,
                    label=f"H{features}",
                    alpha=0.7,
                    color=color
                )

            if finite_diagram_values:
                all_values = torch.cat(finite_diagram_values)
                min_val_pi = torch.min(all_values).item()
                max_val_pi = torch.max(all_values).item()
                axes[12].plot(
                    [min_val_pi, max_val_pi],
                    [min_val_pi, max_val_pi],
                    "k--",
                    alpha=0.5,
                )
                axes[12].legend(loc="lower right")
        except Exception as e:
            axes[12].text(
                0.5,
                0.5,
                f'Hata (PI)\n{e}',
                ha='center',
                va='center',
                color='red',
            )
    else:
        axes[12].text(0.5, 0.5, 'PI Verisi Yok', ha='center', va='center', color='red', fontsize=10)

    # bests ve th değerlerinin None olma ihtimaline karşı güvenli string dönüşümü
    bests_str = bests.round(1) if bests is not None and hasattr(bests, 'round') else "N/A"
    threshold_str = threshold.round(1) if threshold is not None and hasattr(threshold, 'round') else "N/A"
    
    axes[12].set_title(f"H1 {bests_str}\nthreshold={threshold_str}")
    axes[12].set_ylabel("Death")
    axes[12].set_xlabel("Birth")
    axes[12].set_xticks([])
    axes[12].set_yticks([])

    axes[13].axis("off")
    axes[14].axis("off")

    plt.tight_layout()

    if show:
        # Interactive mode intentionally waits until the window is closed.
        plt.show(block=True)

    # Always release the figure. Without this, batch processing keeps every
    # 3x5 figure and persistence scatter in memory and eventually appears hung.
    plt.close(fig)
    return scores


def plot_pseudo_mask_comparison(
    *,
    img_tensor,
    real_mask,
    topological_mask,
    random_walker_mask,
    morphological_mask,
    chc_otsu_mask,
    grabcut_mask,
    watershed_mask,
    show=False,
):
    """Plot one publication-ready row comparing all pseudo-mask methods.

    The original image is deliberately placed at the far left and the
    ground-truth mask at the far right.  The proposed mask is adjacent to the
    reference, making its boundary agreement easy to inspect.  All remaining
    panels are ordered consistently with the quantitative baseline table.
    """
    predictions = [
        ("Topological (Ours)", topological_mask),
        ("Random Walker", random_walker_mask),
        ("Morph. Chan--Vese", morphological_mask),
        ("CHC--Otsu", chc_otsu_mask),
        ("GrabCut", grabcut_mask),
        ("Watershed", watershed_mask),
    ]

    scores = {}
    for method_name, mask in predictions:
        if mask is None or real_mask is None:
            scores[method_name] = None
        else:
            scores[method_name] = iou_and_dice(
                np.asarray(mask),
                np.asarray(real_mask),
            )

    baseline_panels = [
        (method_name, mask, scores[method_name], False)
        for method_name, mask in predictions[1:]
    ]
    panels = [
        ("Original Image", img_tensor, None, True),
        *baseline_panels,
        (
            predictions[0][0],
            predictions[0][1],
            scores[predictions[0][0]],
            False,
        ),
        ("Ground Truth", real_mask, None, False),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(24, 3.8))
    panel_letters = "abcdefgh"

    for panel_index, (_title, image, score, is_tensor) in enumerate(panels):
        ax = axes[panel_index]
        if image is None:
            ax.text(
                0.5,
                0.5,
                "Unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        elif is_tensor:
            display_image = image.detach().cpu()
            if display_image.ndim == 3 and display_image.shape[0] in (1, 3, 4):
                display_image = display_image.permute(1, 2, 0)
            ax.imshow(np.asarray(display_image).squeeze())
        else:
            ax.imshow(np.asarray(image).squeeze().astype(bool), cmap="gray", vmin=0, vmax=1)

        ax.axis("off")
        ax.text(
            0.5,
            -0.08,
            f"({panel_letters[panel_index]})",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
        )

        if score is not None:
            iou, dice = score
            ax.text(
                0.96,
                0.04,
                f"IoU: {iou:.2f}\nDice: {dice:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="white",
                bbox={
                    "facecolor": "black",
                    "edgecolor": "none",
                    "alpha": 0.65,
                    "pad": 3,
                },
            )

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.14, wspace=0.06)
    if show:
        plt.show(block=True)
    plt.close(fig)
    return scores


def plot_scalar_field_explanation(
    img_tensor,
    clean_img,
    scalar_field,
    background_skin_rgb,
    *,
    variant_name="robust_delta_e_s_v",
    show=True,
):
    """Show how the colour-derived scalar field is constructed.

    The publication-oriented layout keeps the image stages at their natural
    aspect ratio and places a compact method diagram between the DullRazor
    result and the scalar field.  For the default ``robust_delta_e_s_v``
    variant, the diagram mirrors ``utils.color_fields.build_gray_variants``:
    robust Lab skin distance, robust HSV saturation, and inverse robust HSV
    value are averaged and scaled to [0, 255].
    """

    def as_rgb_array(image):
        if torch.is_tensor(image):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)

        if array.ndim == 3 and array.shape[0] in (1, 3, 4):
            array = np.moveaxis(array, 0, -1)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[:, :, 0]
        if array.ndim == 3 and array.shape[-1] == 4:
            array = array[:, :, :3]

        array = np.asarray(array, dtype=np.float32)
        if array.size and float(np.nanmax(array)) > 1.0:
            array = array / 255.0
        return np.clip(array, 0.0, 1.0)

    def add_panel_label(ax, label):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="black",
                edgecolor="none",
                alpha=0.72,
            ),
            zorder=10,
        )

    def add_box(ax, xy, width, height, title, subtitle, facecolor):
        box = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor="#344054",
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(box)
        ax.text(
            xy[0] + width / 2.0,
            xy[1] + height * 0.62,
            title,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color="#101828",
            zorder=3,
        )
        ax.text(
            xy[0] + width / 2.0,
            xy[1] + height * 0.27,
            subtitle,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#344054",
            linespacing=1.05,
            zorder=3,
        )
        return box

    original_rgb = as_rgb_array(img_tensor)
    clean_rgb = as_rgb_array(clean_img)
    scalar = np.asarray(scalar_field, dtype=np.float32).squeeze()
    if scalar.ndim != 2:
        raise ValueError(
            f"Expected a 2-D scalar field, got shape {scalar.shape}."
        )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16.5, 4.4),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.45, 1.0]},
        constrained_layout=True,
    )

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold", pad=9)
    axes[0].axis("off")
    add_panel_label(axes[0], "(a)")

    if background_skin_rgb is not None:
        skin_rgb = np.clip(
            np.asarray(background_skin_rgb, dtype=np.float32),
            0.0,
            1.0,
        )
        swatch_x, swatch_y = 0.69, 0.055
        swatch_width, swatch_height = 0.27, 0.22
        axes[0].add_patch(
            Rectangle(
                (swatch_x, swatch_y),
                swatch_width,
                swatch_height,
                transform=axes[0].transAxes,
                facecolor=skin_rgb,
                edgecolor="white",
                linewidth=2.2,
                zorder=7,
            )
        )
        axes[0].text(
            swatch_x + swatch_width / 2.0,
            swatch_y + swatch_height + 0.018,
            "Skin Color",
            transform=axes[0].transAxes,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.20",
                facecolor="black",
                edgecolor="white",
                linewidth=0.7,
                alpha=0.72,
            ),
            zorder=8,
        )

    axes[1].imshow(clean_rgb)
    axes[1].set_title("DullRazor Hair Removal", fontsize=12, fontweight="bold", pad=9)
    axes[1].axis("off")
    add_panel_label(axes[1], "(b)")

    process_ax = axes[2]
    process_ax.set_title(
        "Scalar-Field Construction",
        fontsize=12,
        fontweight="bold",
        pad=9,
    )
    process_ax.set_xlim(0.0, 1.0)
    process_ax.set_ylim(0.0, 1.0)
    process_ax.set_facecolor("#F8FAFC")
    process_ax.set_xticks([])
    process_ax.set_yticks([])
    for spine in process_ax.spines.values():
        spine.set_color("#D0D5DD")
        spine.set_linewidth(1.0)

    if variant_name == "robust_delta_e_s_v":
        component_specs = [
            (
                0.68,
                r"$R(\Delta E_{ab}^{*})$",
                "Lab distance to\nborder-median skin",
                "#FDE7D7",
            ),
            (0.43, r"$R(S)$", "HSV saturation", "#FEEFC7"),
            (0.18, r"$1-R(V)$", "inverse HSV value", "#DDEBF7"),
        ]
        for y, title, subtitle, color in component_specs:
            add_box(process_ax, (0.035, y), 0.48, 0.17, title, subtitle, color)

        add_box(
            process_ax,
            (0.69, 0.36),
            0.275,
            0.30,
            "Equal-weight\nfusion",
            "mean × 255",
            "#DDF3E4",
        )
        arrow_targets = [0.57, 0.51, 0.45]
        for (y, _, _, _), target_y in zip(component_specs, arrow_targets):
            process_ax.add_patch(
                FancyArrowPatch(
                    (0.52, y + 0.085),
                    (0.685, target_y),
                    transform=process_ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.25,
                    color="#667085",
                    zorder=4,
                )
            )

        process_ax.text(
            0.5,
            0.94,
            r"$R$: robust normalization (1st–99th percentiles)",
            transform=process_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.7,
            color="#475467",
        )
        process_ax.text(
            0.5,
            0.065,
            r"$F(x)=\frac{255}{3}\,[R(\Delta E_{ab}^{*})+R(S)+1-R(V)]$",
            transform=process_ax.transAxes,
            ha="center",
            va="center",
            fontsize=10.5,
            color="#101828",
        )
    else:
        add_box(
            process_ax,
            (0.10, 0.38),
            0.34,
            0.25,
            "DullRazor RGB",
            "colour-space channels",
            "#E4E7EC",
        )
        add_box(
            process_ax,
            (0.60, 0.38),
            0.30,
            0.25,
            "Scalar field",
            variant_name.replace("_", " "),
            "#DDF3E4",
        )
        process_ax.add_patch(
            FancyArrowPatch(
                (0.45, 0.505),
                (0.59, 0.505),
                transform=process_ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.4,
                color="#667085",
            )
        )
    process_ax.text(
        0.02,
        0.98,
        "(c)",
        transform=process_ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        color="#101828",
    )

    scalar_artist = axes[3].imshow(
        scalar,
        cmap="gray",
        vmin=0.0,
        vmax=255.0,
    )
    axes[3].set_title("Scalar Field", fontsize=12, fontweight="bold", pad=9)
    axes[3].axis("off")
    add_panel_label(axes[3], "(d)")
    colorbar = fig.colorbar(
        scalar_artist,
        ax=axes[3],
        orientation="horizontal",
        fraction=0.055,
        pad=0.035,
        ticks=[0, 255],
    )
    colorbar.ax.set_xticklabels(["Low", "High"])
    colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.set_label("Lesion evidence", fontsize=8, labelpad=-1)

    if show:
        plt.show(block=True)

    plt.close(fig)
    return fig


def plot_topological_cleanup_explanation(
    scalar_field,
    persistence_info,
    selected_h1,
    raw_h1_mask,
    h0_noise_mask,
    spatial_noise_mask,
    final_mask,
    real_mask,
    *,
    h1_anchor=None,
    include_theory=True,
    show=True,
    max_persistence_points=2000,
):
    """Create separate publication figures for one example and the theory.

    The example row contains the scalar field, persistence diagram, raw
    threshold mask, H0 rejection, far-border rejection, and final pseudo-mask.
    It has no titles or panel letters so multiple examples can be stacked in a
    paper.  A second figure explains the five topological stages.
    """

    def as_2d_array(data, name):
        if torch.is_tensor(data):
            array = data.detach().cpu().numpy()
        else:
            array = np.asarray(data)
        array = np.asarray(array).squeeze()
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D {name}, got shape {array.shape}.")
        return array

    def as_binary(data, name):
        return as_2d_array(data, name) > 0

    def diagram_array(info):
        diagram = info.diagram if hasattr(info, "diagram") else info[1]
        if torch.is_tensor(diagram):
            diagram = diagram.detach().cpu().numpy()
        diagram = np.asarray(diagram, dtype=np.float32)
        if diagram.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return diagram.reshape(-1, 2)

    def rejection_overlay(raw, rejected, color):
        if raw.shape != rejected.shape:
            raise ValueError(
                "Raw and rejected masks must have identical shapes, got "
                f"{raw.shape} and {rejected.shape}."
            )
        overlay = np.zeros((*raw.shape, 3), dtype=np.float32)
        overlay[raw] = (0.92, 0.92, 0.92)
        overlay[rejected] = color
        return overlay

    def theory_box(
        ax,
        xy,
        width,
        height,
        text,
        facecolor,
        edgecolor,
        *,
        fontsize=9,
    ):
        patch = FancyBboxPatch(
            xy,
            width,
            height,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(patch)
        ax.text(
            xy[0] + width / 2.0,
            xy[1] + height / 2.0,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color="#101828",
            linespacing=1.2,
            zorder=4,
        )
        return patch

    def prepare_theory_panel(ax, number, color, label, add_arrow=False):
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_box_aspect(0.92)
        ax.set_facecolor("#F8FAFC")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.35)
            spine.set_alpha(0.72)
        badge = Circle(
            (0.075, 0.925),
            0.052,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=20,
        )
        ax.add_patch(badge)
        ax.text(
            0.075,
            0.925,
            str(number),
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="white",
            zorder=21,
        )
        ax.text(
            0.145,
            0.925,
            label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#101828",
            zorder=20,
        )
        if add_arrow:
            ax.add_patch(
                FancyArrowPatch(
                    (1.015, 0.50),
                    (1.115, 0.50),
                    transform=ax.transAxes,
                    arrowstyle="-|>",
                    mutation_scale=13,
                    linewidth=1.4,
                    color="#667085",
                    clip_on=False,
                    zorder=30,
                )
            )

    scalar = as_2d_array(scalar_field, "scalar field").astype(np.float32)
    raw = as_binary(raw_h1_mask, "raw H1 mask")
    h0_rejected = as_binary(h0_noise_mask, "H0 noise mask")
    spatial_rejected = as_binary(spatial_noise_mask, "spatial noise mask")
    final = as_binary(final_mask, "final mask")
    ground_truth = as_binary(real_mask, "ground-truth mask")

    expected_shape = scalar.shape
    for name, mask in (
        ("raw H1 mask", raw),
        ("H0 noise mask", h0_rejected),
        ("spatial noise mask", spatial_rejected),
        ("final mask", final),
        ("ground-truth mask", ground_truth),
    ):
        if mask.shape != expected_shape:
            raise ValueError(
                f"{name} has shape {mask.shape}; expected {expected_shape}."
            )

    iou, dice = iou_and_dice(
        final.astype(np.uint8),
        ground_truth.astype(np.uint8),
    )

    fig, axes = plt.subplots(
        1,
        6,
        figsize=(15.8, 3.0),
        constrained_layout=True,
    )
    for ax in axes:
        ax.set_box_aspect(1)

    # Lesion-enhanced scalar field.
    axes[0].imshow(scalar, cmap="gray", vmin=0.0, vmax=255.0)
    axes[0].axis("off")

    # H0/H1 persistence diagram.  The selected H1 point is ringed.
    persistence_ax = axes[1]
    finite_values = []
    for dimension, color in ((0, "#2563EB"), (1, "#DC2626")):
        diagram = diagram_array(persistence_info[dimension])
        diagram = diagram[np.isfinite(diagram).all(axis=1)]
        if len(diagram) > max_persistence_points:
            sample_indices = np.linspace(
                0,
                len(diagram) - 1,
                max_persistence_points,
                dtype=int,
            )
            diagram = diagram[sample_indices]
        if len(diagram):
            persistence_ax.scatter(
                diagram[:, 0],
                diagram[:, 1],
                s=11,
                alpha=0.58,
                color=color,
                edgecolors="none",
                rasterized=True,
                zorder=3,
            )
            finite_values.append(diagram.reshape(-1))

    selected = None
    if selected_h1 is not None:
        selected = np.asarray(selected_h1, dtype=np.float32).reshape(-1)
        if (
            selected.size >= 2
            and np.isfinite(selected[:2]).all()
            and selected[1] > selected[0]
        ):
            persistence_ax.scatter(
                selected[0],
                selected[1],
                s=115,
                facecolors="none",
                edgecolors="#16A34A",
                linewidths=2.2,
                zorder=6,
            )
            persistence_ax.scatter(
                selected[0],
                selected[1],
                s=18,
                color="#DC2626",
                zorder=7,
            )
            finite_values.append(selected[:2])

    if finite_values:
        values = np.concatenate(finite_values)
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper <= lower:
            upper = lower + 1.0
        margin = 0.10 * (upper - lower)
        persistence_ax.plot(
            [lower, upper],
            [lower, upper],
            linestyle="--",
            linewidth=1.0,
            color="#667085",
            alpha=0.75,
            zorder=1,
        )
        persistence_ax.set_xlim(lower - margin, upper + margin)
        persistence_ax.set_ylim(lower - margin, upper + margin)
    else:
        persistence_ax.text(
            0.5,
            0.5,
            "No persistence data",
            transform=persistence_ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="#667085",
        )
    persistence_ax.set_xlabel("Birth", fontsize=9)
    persistence_ax.set_ylabel("Death", fontsize=9)
    persistence_ax.tick_params(axis="both", labelsize=8)
    persistence_ax.grid(alpha=0.16, linewidth=0.6)
    persistence_ax.set_box_aspect(1)

    # Raw H1-threshold mask and the selected H1 death location.
    axes[2].imshow(raw, cmap="gray", vmin=0, vmax=1)
    if h1_anchor is not None:
        anchor = np.asarray(h1_anchor, dtype=np.float32).reshape(-1)
        if anchor.size >= 2 and np.isfinite(anchor[:2]).all():
            axes[2].plot(
                float(anchor[1]),
                float(anchor[0]),
                marker="x",
                color="#22D3EE",
                markersize=10,
                markeredgewidth=2.4,
                zorder=10,
            )
    axes[2].axis("off")

    # H0-rejected components are magenta; retained pixels are white.
    axes[3].imshow(
        rejection_overlay(raw, h0_rejected, (0.86, 0.12, 0.48)),
        interpolation="nearest",
    )
    axes[3].axis("off")

    # Far-border components are orange; the H1 anchor remains visible.
    axes[4].imshow(
        rejection_overlay(raw, spatial_rejected, (1.0, 0.47, 0.08)),
        interpolation="nearest",
    )
    if h1_anchor is not None:
        anchor = np.asarray(h1_anchor, dtype=np.float32).reshape(-1)
        if anchor.size >= 2 and np.isfinite(anchor[:2]).all():
            axes[4].plot(
                float(anchor[1]),
                float(anchor[0]),
                marker="x",
                color="#22D3EE",
                markersize=10,
                markeredgewidth=2.4,
                zorder=10,
            )
    axes[4].axis("off")

    # Final pseudo-mask; the reference mask is used only for these scores.
    axes[5].imshow(final, cmap="gray", vmin=0, vmax=1)
    axes[5].axis("off")
    axes[5].text(
        0.5,
        0.035,
        f"IoU {iou:.3f}   Dice {dice:.3f}",
        transform=axes[5].transAxes,
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="black",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.76,
        ),
        zorder=10,
    )

    # Display and dispose of the example before creating the theory figure.
    # Keeping both GUI figure managers alive during one blocking show caused
    # duplicated or partially redrawn panels on some interactive backends.
    if show:
        plt.show(block=True)
    plt.close(fig)

    if not include_theory:
        return fig, None

    theory_fig, theory_axes = plt.subplots(
        1,
        5,
        figsize=(25.5, 4.7),
        constrained_layout=True,
    )
    (
        h1_selection_ax,
        anchor_theory_ax,
        h0_pd_ax,
        h0_theory_ax,
        spatial_theory_ax,
    ) = theory_axes

    # Theory stage 1: select a finite H1 class in the persistence diagram.
    prepare_theory_panel(
        h1_selection_ax,
        1,
        "#2563EB",
        r"Select finite $H_1$",
    )
    h1_selection_ax.plot(
        [0.14, 0.82],
        [0.18, 0.78],
        transform=h1_selection_ax.transAxes,
        linestyle="--",
        linewidth=1.1,
        color="#98A2B3",
        zorder=1,
    )
    h1_selection_ax.add_patch(
        FancyArrowPatch(
            (0.14, 0.18),
            (0.86, 0.18),
            transform=h1_selection_ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="#667085",
            zorder=2,
        )
    )
    h1_selection_ax.add_patch(
        FancyArrowPatch(
            (0.14, 0.18),
            (0.14, 0.89),
            transform=h1_selection_ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.0,
            color="#667085",
            zorder=2,
        )
    )
    h1_selection_ax.scatter(
        [0.26, 0.38, 0.67, 0.74],
        [0.32, 0.73, 0.71, 0.86],
        transform=h1_selection_ax.transAxes,
        s=[26, 30, 28, 38],
        color="#DC2626",
        alpha=0.82,
        zorder=4,
    )
    h1_selection_ax.scatter(
        [0.20, 0.34, 0.53, 0.62],
        [0.27, 0.43, 0.62, 0.69],
        transform=h1_selection_ax.transAxes,
        s=[24, 27, 25, 29],
        color="#2563EB",
        alpha=0.78,
        zorder=3,
    )
    h1_selection_ax.scatter(
        [0.38],
        [0.73],
        transform=h1_selection_ax.transAxes,
        s=190,
        facecolors="none",
        edgecolors="#16A34A",
        linewidths=2.4,
        zorder=5,
    )
    h1_selection_ax.plot(
        [0.14, 0.38],
        [0.73, 0.73],
        transform=h1_selection_ax.transAxes,
        linestyle=":",
        linewidth=1.1,
        color="#06B6D4",
        alpha=0.85,
        zorder=3,
    )
    h1_selection_ax.plot(
        0.14,
        0.73,
        marker="x",
        transform=h1_selection_ax.transAxes,
        color="#06B6D4",
        markersize=9,
        markeredgewidth=2.1,
        zorder=7,
    )
    h1_selection_ax.text(
        0.84,
        0.205,
        "Birth",
        transform=h1_selection_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#667085",
    )
    h1_selection_ax.text(
        0.155,
        0.85,
        "Death",
        transform=h1_selection_ax.transAxes,
        ha="center",
        va="top",
        rotation=90,
        fontsize=8,
        color="#667085",
    )
    h1_selection_ax.text(
        0.5,
        0.055,
        r"$i^*=\arg\max_i\;(d_i-b_i)"
        r"[\delta_{\partial\Omega}(x_i^d)+1]$",
        transform=h1_selection_ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.6,
        color="#344054",
    )
    h1_selection_ax.scatter(
        [0.68, 0.80],
        [0.13, 0.13],
        transform=h1_selection_ax.transAxes,
        s=20,
        color=["#2563EB", "#DC2626"],
        zorder=6,
    )
    h1_selection_ax.text(
        0.705,
        0.13,
        r"$H_0$",
        transform=h1_selection_ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.5,
        color="#344054",
    )
    h1_selection_ax.text(
        0.825,
        0.13,
        r"$H_1$",
        transform=h1_selection_ax.transAxes,
        ha="left",
        va="center",
        fontsize=7.5,
        color="#344054",
    )

    # Theory stage 2: derive the threshold and retain the H1 death anchor.
    prepare_theory_panel(
        anchor_theory_ax,
        2,
        "#0891B2",
        r"Threshold and $H_1$ anchor",
    )
    anchor_domain = Rectangle(
        (0.12, 0.22),
        0.76,
        0.58,
        transform=anchor_theory_ax.transAxes,
        facecolor="white",
        edgecolor="#98A2B3",
        linewidth=1.0,
        zorder=1,
    )
    anchor_theory_ax.add_patch(anchor_domain)
    anchor_component = Circle(
        (0.50, 0.51),
        0.20,
        transform=anchor_theory_ax.transAxes,
        facecolor="#E5E7EB",
        edgecolor="#475467",
        linewidth=1.2,
        zorder=3,
    )
    anchor_theory_ax.add_patch(anchor_component)
    anchor_hole = Circle(
        (0.52, 0.52),
        0.068,
        transform=anchor_theory_ax.transAxes,
        facecolor="white",
        edgecolor="#98A2B3",
        linewidth=1.0,
        zorder=4,
    )
    anchor_theory_ax.add_patch(anchor_hole)
    anchor_theory_ax.plot(
        0.52,
        0.52,
        marker="x",
        transform=anchor_theory_ax.transAxes,
        color="#06B6D4",
        markersize=11,
        markeredgewidth=2.4,
        zorder=6,
    )
    anchor_theory_ax.annotate(
        r"$x_{H_1}=x_{i^*}^{d}$",
        xy=(0.52, 0.52),
        xytext=(0.68, 0.70),
        xycoords=anchor_theory_ax.transAxes,
        textcoords=anchor_theory_ax.transAxes,
        fontsize=9,
        color="#0E7490",
        ha="center",
        arrowprops=dict(
            arrowstyle="->",
            color="#0E7490",
            linewidth=1.1,
        ),
    )
    anchor_theory_ax.text(
        0.5,
        0.10,
        r"$\tau=b_{i^*}+\alpha(d_{i^*}-b_{i^*})$",
        transform=anchor_theory_ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#344054",
    )

    # Theory stage 3: select H0 classes that are alive at the threshold.
    prepare_theory_panel(
        h0_pd_ax,
        3,
        "#7C3AED",
        r"Select active $H_0$",
    )
    plot_left, plot_bottom = 0.15, 0.18
    plot_right, plot_top = 0.86, 0.80
    tau_position = 0.52
    h0_pd_ax.add_patch(
        Rectangle(
            (plot_left, tau_position),
            tau_position - plot_left,
            plot_top - tau_position,
            transform=h0_pd_ax.transAxes,
            facecolor="#DBEAFE",
            edgecolor="none",
            alpha=0.72,
            zorder=1,
        )
    )
    h0_pd_ax.plot(
        [plot_left, plot_right],
        [plot_bottom, plot_top],
        transform=h0_pd_ax.transAxes,
        linestyle="--",
        linewidth=1.0,
        color="#98A2B3",
        zorder=2,
    )
    h0_pd_ax.plot(
        [tau_position, tau_position],
        [plot_bottom, plot_top],
        transform=h0_pd_ax.transAxes,
        linestyle=":",
        linewidth=1.3,
        color="#7C3AED",
        zorder=3,
    )
    h0_pd_ax.plot(
        [plot_left, plot_right],
        [tau_position, tau_position],
        transform=h0_pd_ax.transAxes,
        linestyle=":",
        linewidth=1.3,
        color="#7C3AED",
        zorder=3,
    )
    h0_pd_ax.scatter(
        [0.25, 0.43],
        [0.72, 0.61],
        transform=h0_pd_ax.transAxes,
        s=[42, 38],
        color="#2563EB",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )
    h0_pd_ax.scatter(
        [0.31, 0.66, 0.72],
        [0.40, 0.74, 0.78],
        transform=h0_pd_ax.transAxes,
        s=[30, 31, 31],
        color="#98A2B3",
        alpha=0.82,
        zorder=4,
    )
    h0_pd_ax.text(
        0.32,
        0.75,
        r"active: $b_i\leq\tau<d_i$",
        transform=h0_pd_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color="#1D4ED8",
    )
    h0_pd_ax.text(
        tau_position,
        0.14,
        r"$\tau$",
        transform=h0_pd_ax.transAxes,
        ha="center",
        fontsize=9,
        color="#6D28D9",
    )
    h0_pd_ax.text(
        0.49,
        0.075,
        "Birth",
        transform=h0_pd_ax.transAxes,
        ha="center",
        fontsize=8,
        color="#667085",
    )
    h0_pd_ax.text(
        0.09,
        0.49,
        "Death",
        transform=h0_pd_ax.transAxes,
        ha="center",
        va="center",
        rotation=90,
        fontsize=8,
        color="#667085",
    )

    # Theory stage 4: split active H0 persistences with adaptive Otsu.
    prepare_theory_panel(
        h0_theory_ax,
        4,
        "#0284C7",
        r"Otsu split of active $H_0$",
    )
    h0_theory_ax.set_xlim(0.0, 1.0)
    h0_theory_ax.set_ylim(0.0, 1.0)
    h0_theory_ax.set_facecolor("#F8FAFC")
    h0_theory_ax.set_xticks([])
    h0_theory_ax.set_yticks([])

    h0_theory_ax.text(
        0.31,
        0.83,
        r"$\theta_{H_0}=\mathrm{Otsu}(\{p_i\})+\Delta_{\mathrm{bin}}$",
        transform=h0_theory_ax.transAxes,
        ha="center",
        va="center",
        fontsize=10.5,
        color="#344054",
    )
    h0_theory_ax.plot(
        [0.08, 0.59],
        [0.20, 0.20],
        transform=h0_theory_ax.transAxes,
        color="#98A2B3",
        linewidth=1.2,
        clip_on=False,
    )
    h0_theory_ax.text(
        0.18,
        0.115,
        "low\npersistence",
        transform=h0_theory_ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        linespacing=0.9,
        color="#667085",
    )
    h0_theory_ax.text(
        0.50,
        0.115,
        "high\npersistence",
        transform=h0_theory_ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.6,
        linespacing=0.9,
        color="#667085",
    )
    otsu_x = 0.35
    h0_theory_ax.plot(
        [otsu_x, otsu_x],
        [0.17, 0.72],
        transform=h0_theory_ax.transAxes,
        linestyle="--",
        color="#101828",
        linewidth=1.3,
        zorder=2,
    )
    h0_theory_ax.text(
        otsu_x,
        0.74,
        r"$\theta_{H_0}$",
        transform=h0_theory_ax.transAxes,
        ha="center",
        fontsize=11,
        color="#101828",
    )
    h0_theory_ax.scatter(
        [0.16, 0.22, 0.27, 0.31],
        [0.43, 0.51, 0.46, 0.56],
        transform=h0_theory_ax.transAxes,
        s=[35, 42, 38, 34],
        color="#7DD3FC",
        edgecolors="#2563EB",
        linewidths=0.7,
        zorder=5,
    )
    h0_theory_ax.scatter(
        [0.42, 0.49, 0.56],
        [0.48, 0.58, 0.44],
        transform=h0_theory_ax.transAxes,
        s=[36, 43, 39],
        color="#1D4ED8",
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    theory_box(
        h0_theory_ax,
        (0.65, 0.55),
        0.30,
        0.20,
        r"$p_i > \theta_{H_0}$" + "\nretain",
        "#DBEAFE",
        "#1D4ED8",
    )
    theory_box(
        h0_theory_ax,
        (0.65, 0.30),
        0.30,
        0.20,
        r"$p_i \leq \theta_{H_0}$" + "\nreject",
        "#E0F2FE",
        "#38BDF8",
    )
    theory_box(
        h0_theory_ax,
        (0.65, 0.055),
        0.30,
        0.17,
        r"$H_1$ reference" + "\nalways retained",
        "#DCFCE7",
        "#16A34A",
    )

    # Theory stage 5: boundary-touching components far from the H1 anchor.
    prepare_theory_panel(
        spatial_theory_ax,
        5,
        "#EA580C",
        "Reject far-border components",
        add_arrow=False,
    )
    spatial_theory_ax.set_xlim(0.0, 1.0)
    spatial_theory_ax.set_ylim(0.0, 1.0)
    spatial_theory_ax.set_facecolor("#F8FAFC")
    spatial_theory_ax.set_xticks([])
    spatial_theory_ax.set_yticks([])

    domain = Rectangle(
        (0.045, 0.08),
        0.58,
        0.82,
        transform=spatial_theory_ax.transAxes,
        facecolor="white",
        edgecolor="#98A2B3",
        linewidth=1.2,
        zorder=1,
    )
    spatial_theory_ax.add_patch(domain)
    lesion_component = Circle(
        (0.29, 0.50),
        0.16,
        transform=spatial_theory_ax.transAxes,
        facecolor="#E5E7EB",
        edgecolor="#475467",
        linewidth=1.2,
        zorder=3,
    )
    spatial_theory_ax.add_patch(lesion_component)
    near_component = Circle(
        (0.49, 0.70),
        0.052,
        transform=spatial_theory_ax.transAxes,
        facecolor="#E5E7EB",
        edgecolor="#475467",
        linewidth=1.0,
        zorder=3,
    )
    spatial_theory_ax.add_patch(near_component)
    far_component = FancyBboxPatch(
        (0.56, 0.18),
        0.065,
        0.16,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        transform=spatial_theory_ax.transAxes,
        facecolor="#FF7A12",
        edgecolor="#C2410C",
        linewidth=1.1,
        zorder=4,
    )
    spatial_theory_ax.add_patch(far_component)
    anchor_x, anchor_y = 0.29, 0.50
    spatial_theory_ax.plot(
        anchor_x,
        anchor_y,
        marker="x",
        color="#06B6D4",
        markersize=10,
        markeredgewidth=2.3,
        transform=spatial_theory_ax.transAxes,
        zorder=6,
    )
    spatial_theory_ax.text(
        anchor_x,
        anchor_y - 0.22,
        r"$x_{H_1}$",
        transform=spatial_theory_ax.transAxes,
        ha="center",
        fontsize=10,
        color="#0E7490",
    )
    for end, label_position in (
        ((0.49, 0.70), (0.42, 0.63)),
        ((0.59, 0.26), (0.45, 0.36)),
    ):
        spatial_theory_ax.add_patch(
            FancyArrowPatch(
                (anchor_x + 0.02, anchor_y),
                end,
                transform=spatial_theory_ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=10,
                linestyle="--",
                linewidth=1.1,
                color="#667085",
                zorder=5,
            )
        )
        spatial_theory_ax.text(
            label_position[0],
            label_position[1],
            r"$d_i$",
            transform=spatial_theory_ax.transAxes,
            fontsize=9,
            color="#475467",
        )
    theory_box(
        spatial_theory_ax,
        (0.65, 0.49),
        0.32,
        0.31,
        "reject $C_i$ iff\n"
        r"$C_i\neq C_{\mathrm{ref}}$" + "\n"
        r"$C_i\cap\partial\Omega\neq\varnothing$" + "\nand\n"
        r"$d(C_i,x_{H_1})>\theta_d$",
        "#FFF1E7",
        "#EA580C",
        fontsize=8.2,
    )
    theory_box(
        spatial_theory_ax,
        (0.65, 0.16),
        0.32,
        0.20,
        "near or interior\ncomponent retained",
        "#F2F4F7",
        "#667085",
        fontsize=8.3,
    )

    if show:
        plt.show(block=True)

    plt.close(theory_fig)
    return fig, theory_fig


#### resim - persistence diagram ve threshold interaktif görselleştirme
def plot_clean_pi_and_hole_evolution(name, clean_img, gray, pi, bests, steps=100):
    """
    bests = [birth, death, persistence]
    gray >= t  -> superlevel thresholding / superlevel filtration
    """

    birth = float(bests[0])
    death = float(bests[1])

    # Birth'ten biraz yukarıdan başlat
    eps = 2.0
    start_th = birth + eps

    thresholds = np.linspace(start_th, death, steps)
    masks = [(gray >= th) for th in thresholds]

    # 6 threshold seç
    idxs = np.linspace(0, len(thresholds) - 1, 6, dtype=int)
    selected_thresholds = thresholds[idxs]
    selected_masks = [masks[i] for i in idxs]

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.15, 1.15],
        height_ratios=[1, 1],
        wspace=0.25,
        hspace=0.28
    )

    fig.suptitle(f"{name}", fontsize=15)

    # -------------------------
    # Sol üst: Clean Image
    # -------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    if clean_img is not None:
        ax0.imshow(clean_img, cmap="gray")
    else:
        ax0.text(0.5, 0.5, "Veri Yok", ha="center", va="center", color="red")
    ax0.set_title("Clean Image")
    ax0.axis("off")

    # -------------------------
    # Sol alt: Persistence Diagram
    # -------------------------
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title("Persistence Diagram")
    ax1.set_xlabel("Birth")
    ax1.set_ylabel("Death")

    if pi is not None:
        try:
            all_vals = []

            for features, color in zip([0, 1], ["blue", "red"]):
                if pi[features][1] is None or len(pi[features][1]) == 0:
                    continue

                births = pi[features][1][:, 0]
                deaths = pi[features][1][:, 1]

                births_np = births.detach().cpu().numpy()
                deaths_np = deaths.detach().cpu().numpy()

                ax1.scatter(
                    births_np,
                    deaths_np,
                    s=18,
                    alpha=0.7,
                    color=color,
                    label=f"H{features}"
                )

                all_vals.extend([births, deaths])

            if len(all_vals) > 0:
                all_cat = torch.cat(all_vals)
                mn, mx = all_cat.min().item(), all_cat.max().item()
                ax1.plot([mn, mx], [mn, mx], "k--", alpha=0.5)

            # seçilen nokta
            ax1.scatter(
                birth,
                death,
                s=40,
                color="red",
                zorder=5
            )

            # yazı aşağıda, ok yukarı doğru
            ax1.annotate(
                f"({int(round(birth))}, {int(round(death))})",
                xy=(birth, death),
                xytext=(0, -28),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=9,
                color="green",
                arrowprops=dict(
                    arrowstyle="->",
                    color="green",
                    lw=1.2,
                    shrinkA=0,
                    shrinkB=4
                )
            )

            ax1.legend(loc="lower right")

        except Exception as e:
            ax1.text(
                0.5, 0.5, f"Hata (PI)\n{e}",
                ha="center", va="center", color="red"
            )
    else:
        ax1.text(
            0.5, 0.5, "PI Verisi Yok",
            ha="center", va="center", color="red"
        )

    # -------------------------
    # Sağ: 2x3 threshold maskeleri
    # -------------------------
    subgs = gs[:, 1:].subgridspec(2, 3, wspace=0.12, hspace=0.18)

    for j in range(6):
        axm = fig.add_subplot(subgs[j // 3, j % 3])
        axm.imshow(selected_masks[j], cmap="gray")

        t = int(round(selected_thresholds[j]))

        if j == 0:
            title = f"Birth\nI ≥ {t}"
        elif j == 5:
            title = f"Death\nI ≥ {t}"
        else:
            title = f"I ≥ {t}"

        axm.set_title(title, fontsize=10)
        axm.axis("off")

    plt.tight_layout()
    plt.show()

#### clean ve persistence diagram 
def plot_clean_and_pi(name, clean_img, pi):

    import torch
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -------------------------
    # Clean Image
    # -------------------------
    if clean_img is not None:
        axes[0].imshow(clean_img, cmap="gray")
    else:
        axes[0].text(0.5, 0.5, "No Image", ha="center", va="center")

    axes[0].set_title("Clean Image")
    axes[0].axis("off")

    # -------------------------
    # Persistence Diagram
    # -------------------------
    ax = axes[1]
    ax.set_title("Persistence Diagram")
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")

    if pi is not None:
        try:
            all_vals = []

            # H0 ve H1 noktaları
            for features, color in zip([0, 1], ["blue", "red"]):
                if pi[features][1] is None or len(pi[features][1]) == 0:
                    continue

                births = pi[features][1][:, 0]
                deaths = pi[features][1][:, 1]

                ax.scatter(
                    births.detach().cpu().numpy(),
                    deaths.detach().cpu().numpy(),
                    s=20,
                    color=color,
                    alpha=0.7,
                    label=f"H{features}"
                )

                all_vals.append(births)
                all_vals.append(deaths)

            # diagonal
            mn, mx = 0.0, 1.0
            if len(all_vals) > 0:
                all_concat = torch.cat(all_vals)
                mn = all_concat.min().item()
                mx = all_concat.max().item()
                ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5)

            # -------------------------
            # most persistent H1
            # -------------------------
            if pi[1][1] is not None and len(pi[1][1]) > 0:
                h1 = pi[1][1]

                births_h1 = h1[:, 0]
                deaths_h1 = h1[:, 1]
                pers = deaths_h1 - births_h1

                idx = torch.argmax(pers)

                birth = births_h1[idx].item()
                death = deaths_h1[idx].item()

                # kırmızı merkez
                ax.scatter(
                    birth,
                    death,
                    s=35,
                    color="red",
                    zorder=6
                )

                # yeşil halka
                ax.scatter(
                    birth,
                    death,
                    s=220,
                    facecolors="none",
                    edgecolors="lime",
                    linewidths=2.5,
                    zorder=5,
                    label="Most persistent H1"
                )

                # okun ucu yeşil halkanın ALT tarafını göstersin
                offset = (mx - mn) * 0.03

                ax.annotate(
                    f"({birth:.1f}, {death:.1f})",
                    xy=(birth, death - offset),   # aşağıdan yukarı ok
                    xytext=(0, -35),              # yazı aşağıda
                    textcoords="offset points",
                    ha="center",
                    fontsize=10,
                    arrowprops=dict(
                        arrowstyle="->",
                        lw=1.0,
                        color="black"
                    )
                )

            ax.legend(loc="lower right")

        except Exception as e:
            ax.text(
                0.5, 0.5,
                f"Error\n{e}",
                ha="center",
                va="center",
                color="red"
            )

    else:
        ax.text(
            0.5, 0.5,
            "No Persistence Data",
            ha="center",
            va="center"
        )

    plt.tight_layout()
    plt.show()


#### noise removal
def plot_pipeline(img_tensor, clean_img, gray,bests):

    fig, axes = plt.subplots(1,4, figsize=(14,5))

    birth = float(bests[0])
    death = float(bests[1])


    th = birth + (death-birth)*0.2
    mask = (gray >= th)
    mask = morph.remove_small_holes(mask.astype(bool), area_threshold=32*32).astype(np.uint8)
    mask = morph.remove_small_objects(mask.astype(bool), min_size=64).astype(np.uint8)

    # -------- images --------
    axes[0].imshow(img_tensor.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Original Image", fontsize=16, pad=12)
    axes[0].axis("off")

    axes[1].imshow(clean_img)
    axes[1].set_title("Hair Removal", fontsize=16, pad=12)
    axes[1].axis("off")

    axes[2].imshow(gray, cmap="gray")
    axes[2].set_title("Color Fusion", fontsize=16, pad=12)
    axes[2].axis("off")

    axes[3].imshow(mask, cmap="gray")
    axes[3].set_title("Pseudo Mask", fontsize=16, pad=12)
    axes[3].axis("off")
    
    plt.tight_layout()
    fig.canvas.draw()

    plt.show()





    # plt.show() # Grafiği anlık ekrana basmak istersen açabilirsin

    # for features, color in zip([0, 1], ["blue", "red"]):
    #     births = pi[features][1][:, 0]
    #     deaths = pi[features][1][:, 1]

    #     # H0 için (0,255) olan noktayı çıkar
    #     if features == 0:
    #         mask = ~((births == 0) & (deaths == 255))
    #         births = births[mask]
    #         deaths = deaths[mask]

    #         axes[6].scatter(
    #             births.detach().cpu().numpy(),
    #             (deaths - births).detach().cpu().numpy(),
    #             s=20,
    #             label=f"H{features}",
    #             alpha=0.7,
    #             color=color
    #         )

    # min_val_pi = torch.min(torch.cat([pi[f][1][:,0] for f in [0,1]])).item()
    # max_val_pi = torch.max(torch.cat([pi[f][1][:,1] for f in [0,1]])).item()
    # axes[6].plot([min_val_pi, max_val_pi], [min_val_pi, max_val_pi], "k--", alpha=0.5)


    # axes[10].imshow(otsu, cmap='gray')
    # iou, dice = scores["Otsu"]
    # axes[10].set_title(f"Otsu\nIoU:{iou:.2f}, Dice:{dice:.2f}")
    # axes[10].axis("off")


    # Boş kalan akslar kapat
    # for i in range(9, len(axes)):
    #     axes[i].axis("off")

#topological_segmentaiton 
def plot_topological_results(img_tensor, random_walker_mask, morphological_mask, otsu, cc_mask, real_mask):
    
    masks = {
        "Random Walker": random_walker_mask,
        "Morphological": morphological_mask,
        "Otsu": otsu,
        "Topological": cc_mask
    }

    scores = {}
    for name, m in masks.items():
        if m is None or real_mask is None:
            scores[name] = (0.0, 0.0)
        else:
            scores[name] = iou_and_dice(m, real_mask)

    fig, axes = plt.subplots(1, 6, figsize=(16, 3))
    axes = axes.ravel()

    def safe_imshow(ax, img, score=None, is_tensor=False, cmap='gray'):
        if img is not None:
            if is_tensor:
                ax.imshow(img.permute(1,2,0).cpu().numpy())
            else:
                ax.imshow(img, cmap=cmap)

        if score is not None:
            iou, dice = score
            text = f"IoU: {iou:.2f}\nDice: {dice:.2f}"
            ax.text(
                0.04, 0.96,
                text,
                transform=ax.transAxes,
                fontsize=13,
                fontweight='bold',
                verticalalignment='top',
                color='white',
                bbox=dict(facecolor='black', alpha=0.6, pad=4)
            )

        ax.axis("off")

    # 1 Original
    safe_imshow(axes[0], img_tensor, is_tensor=True, cmap=None)

    # 2 Random Walker
    safe_imshow(axes[1], random_walker_mask, scores["Random Walker"])

    # 3 Morphological
    safe_imshow(axes[2], morphological_mask, scores["Morphological"])

    # 4 Otsu
    safe_imshow(axes[3], otsu, scores["Otsu"])

    # 5 Topological
    safe_imshow(axes[4], cc_mask, scores["Topological"])

    # 6 Ground Truth
    safe_imshow(axes[5], real_mask)

    plt.tight_layout()


def plot_area_curve_and_lesion_evolution_4x2(
    name,
    clean_img,
    gray,
    bests,
    steps=100,
    alpha=0.20,
    eps=2.0
):
    birth = float(bests[0])
    death = float(bests[1])
    pers = float(bests[2]) if len(bests) >= 3 else death - birth

    target_th = birth + alpha * pers

    thresholds = np.linspace(birth + eps, death, steps)
    masks = [(gray >= th) for th in thresholds]
    areas = np.array([m.sum() for m in masks])

    # 4x2 = 8 maske
    idxs = np.linspace(0, len(thresholds) - 1, 8, dtype=int)
    selected_thresholds = thresholds[idxs]
    selected_masks = [masks[i] for i in idxs]

    target_idx_global = np.argmin(np.abs(thresholds - target_th))
    target_idx_selected = np.argmin(np.abs(selected_thresholds - target_th))

    # Alanı bin piksel cinsinden göster
    areas_k = areas / 1000.0

    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)

    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1.1, 1.1, 1, 1, 1],
        height_ratios=[1, 1],
        wspace=0.08,
        hspace=0.08
    )

    fig.suptitle(name, fontsize=14)

    # Sol üst: lezyon
    ax_img = fig.add_subplot(gs[0, 0:2])

    if clean_img is not None:
        ax_img.imshow(clean_img, cmap="gray")
        ax_img.set_title("Clean Lesion Image", fontsize=10)
    else:
        ax_img.text(0.5, 0.5, "Veri Yok", ha="center", va="center", color="red")

    ax_img.axis("off")

    # Sol alt: area curve
    ax_area = fig.add_subplot(gs[1, 0:2])

    ax_area.plot(thresholds, areas_k, linewidth=2, label="Area A(t)")
    ax_area.scatter(
        selected_thresholds,
        areas_k[idxs],
        s=45,
        label="Sampled thresholds"
    )

    ax_area.axvline(
        target_th,
        linestyle="--",
        linewidth=2,
        color="red",
        label=fr"$b + {alpha:.2f}p$"
    )

    ax_area.scatter(
        thresholds[target_idx_global],
        areas_k[target_idx_global],
        s=100,
        color="red",
        zorder=5
    )

    ax_area.set_title("Area evolution", fontsize=10)
    ax_area.set_xlabel("Threshold", fontsize=9)
    ax_area.set_ylabel("Area (×1000 pixels)", fontsize=9)
    ax_area.tick_params(axis="both", labelsize=8)
    ax_area.grid(True, alpha=0.35)
    ax_area.legend(fontsize=8)

    # Sağ: 4x2 maskeler
    subgs = gs[:, 2:].subgridspec(
        2, 4,
        wspace=0.02,
        hspace=0.16
    )

    for j in range(8):
        axm = fig.add_subplot(subgs[j // 4, j % 4])
        axm.imshow(selected_masks[j], cmap="gray")

        t = selected_thresholds[j]

        if j == target_idx_selected:
            title = fr"$b+{alpha:.2f}p$" + f"\nI ≥ {t:.1f}"
            axm.set_title(title, fontsize=9, color="red")
        elif j == 0:
            axm.set_title(f"Near Birth\nI ≥ {t:.1f}", fontsize=9)
        elif j == 7:
            axm.set_title(f"Near Death\nI ≥ {t:.1f}", fontsize=9)
        else:
            axm.set_title(f"I ≥ {t:.1f}", fontsize=9)

        axm.axis("off")

    plt.show()

    return {
        "birth": birth,
        "death": death,
        "persistence": pers,
        "target_threshold": target_th,
        "areas": areas,
        "areas_k": areas_k,
        "thresholds": thresholds,
        "masks": masks
    }

"""
SMART H1 THRESHOULD VISUALIZATION

import numpy as np
import matplotlib.pyplot as plt

# ---- Maskeler ve alanlar ----
masks = [(gray >= th) for th in thresholds]
areas = np.array([m.sum() for m in masks])

# ---- Türevler ----
dA = np.gradient(areas)
ddA = np.gradient(dA)

# ---- İkinci türev maksimum noktası ----
der_idx = np.argmax(ddA)

# ---- 8 temsilci threshold ----
idxs = np.linspace(0, len(areas) - 1, 8, dtype=int)

# ---- Figure layout ----
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(2, 5, width_ratios=[3.5, 1, 1, 1, 1])

# =========================
# SOL: Büyük grafik
# =========================
ax = fig.add_subplot(gs[:, 0])

ax.plot(areas, label="Area A(t)", linewidth=2)
ax.plot(dA, label="First derivative dA/dt", linestyle="--")
ax.plot(ddA, label="Second derivative d²A/dt²", linestyle=":")

# Seçili threshold noktaları
ax.scatter(idxs, areas[idxs], s=90, zorder=4, label="Selected thresholds")

# 2. türev maksimum noktası (SADECE NOKTA)
ax.scatter(
    der_idx,
    ddA[der_idx],
    s=160,
    zorder=6,
    label="Max d²A/dt²"
)

ax.annotate(
    "Max curvature",
    (der_idx, ddA[der_idx]),
    textcoords="offset points",
    xytext=(6, -12),
    fontsize=10
)

ax.set_xlabel("Threshold index")
ax.set_ylabel("Pixel count / derivative value")
ax.set_title("Area evolution and derivatives across filtration")
ax.legend()
ax.grid(True)

# =========================
# SAĞ: 8 maske (2x4)
# =========================
for k, idx in enumerate(idxs):
    axm = fig.add_subplot(gs[k // 4, 1 + (k % 4)])
    axm.imshow(masks[idx], cmap="gray")
    axm.set_title(f"idx={idx}\nA={areas[idx]}", fontsize=9)
    axm.axis("off")

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# --- Thresholdlar ve maskeler ---
thresholds = np.linspace(birth, birth + 0.75 * pers, steps)
masks = [(gray >= th) for th in thresholds]

# --- %10 persistence threshold (fallback çizgisi) ---
fallback_th = birth + 0.10 * pers

# --- 8 temsilci threshold seç ---
idxs = np.linspace(0, len(thresholds) - 1, 8, dtype=int)
selected_thresholds = thresholds[idxs]

# =========================
# Figure
# =========================
fig = plt.figure(figsize=(18, 5))
gs = fig.add_gridspec(2, 8, height_ratios=[1, 3])

# =========================
# ALT: Linear persistence çizgisi
# =========================
ax_line = fig.add_subplot(gs[0, :])

# Ana persistence çizgisi
ax_line.plot(
    [birth, birth + pers],
    [0, 0],
    linewidth=3
)

# Threshold noktaları
ax_line.scatter(
    selected_thresholds,
    np.zeros_like(selected_thresholds),
    s=90,
    label="Sampled thresholds"
)

# %10 persistence noktası
ax_line.scatter(
    fallback_th,
    0,
    s=160,
    marker="o",
    label="0.10 × Persistence threshold"
)

# Doğum ve ölüm anotasyonları
ax_line.text(birth, 0.08, "Birth", ha="center", fontsize=10)
ax_line.text(birth + pers, 0.08, "Death", ha="center", fontsize=10)

ax_line.set_yticks([])
ax_line.set_xlabel("Filtration threshold value")
ax_line.set_title("Persistence interval and sampled thresholds")
ax_line.legend()
ax_line.grid(True, axis="x")

# =========================
# ÜST: 8 maske (threshold sırasıyla)
# =========================
for i, idx in enumerate(idxs):
    axm = fig.add_subplot(gs[1, i])
    axm.imshow(masks[idx], cmap="gray")
    axm.set_title(
        f"t={thresholds[idx]:.1f}",
        fontsize=9
    )
    axm.axis("off")

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# ---- Normalize area ----
y = (areas - areas.min()) / (areas.max() - areas.min() + 1e-9)

# ---- Y-eksenine göre simetrik x ----
x = np.linspace(-1, 1, len(y))

# ---- Simetrik referans doğrusu (L-method) ----
# uç noktaları birleştir
slope = (y[-1] - y[0]) / (x[-1] - x[0])
intercept = y[0] - slope * x[0]
y_line = slope * x + intercept

# ---- Dik uzaklıklar ----
distances = np.abs(y - y_line)
knee_idx = np.argmax(distances)
knee_strength = distances[knee_idx]

# ---- Plot ----
plt.figure(figsize=(7, 5))

plt.plot(x, y, linewidth=2, label="Normalized area curve")
plt.plot(x, y_line, "--", label="Symmetric reference line")

plt.scatter(
    x[knee_idx], y[knee_idx],
    s=130, color="red", zorder=5,
    label=f"Knee point (strength={knee_strength:.2f})"
)

# Y-ekseni vurgusu
plt.axvline(0, color="gray", linestyle=":", linewidth=1)

plt.xlabel("Symmetric threshold axis")
plt.ylabel("Normalized area")
plt.title("Symmetric knee detection")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
"""
