import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt 
import torch    
from skimage import measure, morphology as morph
import torchvision.transforms as T
from torch_topological.nn import CubicalComplex
from torchvision.transforms.functional import to_pil_image
from skimage.transform import resize
from scipy import ndimage as ndi
from utils.Dullrazor import dullrazor
from utils.iou_dice import iou_and_dice
from utils.Algorithms import (
    chc_otsu_pseudo_mask,
    grabcut_pseudo_mask,
    morphological_chan_vese_segmentation,
    random_walker_pseudo_mask,
    watershed_pseudo_mask,
)
from utils.color_fields import (
    build_gray_variants,
    estimate_background_skin_rgb,
    lesion_is_high,
)
from utils.padding import adaptive_pad, upsample_patch_map
from skimage.filters import threshold_otsu
from plotting import (
    plot,
    plot_pseudo_mask_comparison,
    plot_scalar_field_explanation,
    plot_topological_cleanup_explanation,
)
import time

# None: aktif H0 persistence değerlerinde otomatik Otsu.
# Sayı: bu değerden kısa yaşayan H0 bileşenlerini elle reddet.
H0_PERSISTENCE_THRESHOLD = None

# Independent switches for the interactive figures produced by this file.
# Each value can also be overridden with an environment variable of the same
# name (for example: SHOW_COMPARISON_PLOT=0).
SHOW_DIAGNOSTIC_PLOT = False
SHOW_COMPARISON_PLOT = True
SHOW_SCALAR_EXPLANATION_PLOT = False
SHOW_CLEANUP_PLOT = True

def _persistence_diagram(info):
    return info.diagram if hasattr(info, "diagram") else info[1]


def _persistence_pairing(info):
    return info.pairing if hasattr(info, "pairing") else info[0]


def h1_spatial_component_filter(mask, death_coordinate, source_shape):
    """Reject detached, border-touching components far from the main H1 hole.

    The death coordinate of the longest finite H1 bar is mapped from the H1
    filtration grid to the raw-mask grid. The component nearest this anchor is
    the reference component. Distances of all components to the anchor are
    split adaptively with Otsu; only non-reference components that both touch
    the image boundary and fall in the far-distance group are rejected.
    """
    binary = np.asarray(mask, dtype=bool)
    labels = measure.label(binary, connectivity=2)
    regions = measure.regionprops(labels)
    if death_coordinate is None or len(regions) < 2:
        return binary.copy(), np.zeros_like(binary), None, {}

    source_height, source_width = source_shape
    target_height, target_width = binary.shape
    death_row, death_column = np.asarray(death_coordinate, dtype=np.float32)[:2]
    anchor = np.array(
        [
            death_row * (target_height - 1) / max(source_height - 1, 1),
            death_column * (target_width - 1) / max(source_width - 1, 1),
        ],
        dtype=np.float32,
    )
    anchor[0] = np.clip(anchor[0], 0, target_height - 1)
    anchor[1] = np.clip(anchor[1], 0, target_width - 1)

    distances = {}
    for region in regions:
        offsets = region.coords.astype(np.float32) - anchor[None, :]
        distances[int(region.label)] = float(
            np.sqrt(np.min(np.sum(offsets * offsets, axis=1)))
        )

    reference_label = min(distances, key=distances.get)
    distance_values = np.asarray(list(distances.values()), dtype=np.float32)
    if np.all(distance_values == distance_values[0]):
        return binary.copy(), np.zeros_like(binary), reference_label, distances
    far_threshold = float(threshold_otsu(distance_values))

    rejected_labels = set()
    for region in regions:
        min_row, min_column, max_row, max_column = region.bbox
        touches_boundary = (
            min_row == 0
            or min_column == 0
            or max_row == target_height
            or max_column == target_width
        )
        is_far = distances[int(region.label)] > far_threshold
        if int(region.label) != reference_label and touches_boundary and is_far:
            rejected_labels.add(int(region.label))

    rejected_mask = np.isin(labels, list(rejected_labels))
    cleaned = binary & ~rejected_mask
    return cleaned, rejected_mask, reference_label, distances


def cubical_complex_segmentation(
    image,
    h0_image=None,
    persistence_threshold=None,
    alpha=0.20,
    lesion_high=True,
    min_object_size=64,
    return_debug=False,
):
    """
    H1 homolojisi padded ``image`` üzerinden threshold değerini belirler.
    H0 homolojisi padsiz ``h0_image`` ters çevrilerek hesaplanır ve
    belirlenen eşikteki kısa ömürlü bileşenleri temizlemek için kullanılır.
    Ayrıca en uzun H1 özelliğinin ölüm koordinatından uzak, bağlantısız ve
    görüntü sınırına temas eden bileşenler uzamsal artefakt olarak elenir.

    ``persistence_threshold=None`` iken sabit bir oran kullanılmaz; aktif
    H0 bileşenlerinin persistence değerlerinden Otsu ile veri-adaptif bir
    eşik hesaplanır. Eşik, Otsu histogramının yalnızca bir kutu genişliği
    kadar yükseltilerek sınırdaki kısa ömürlü gürültülere karşı hafifçe daha
    seçici yapılır. H1'in işaret ettiği referans bileşen hem H0 hem de son
    nesne temizliği sırasında korunur. Sayısal bir değer verilirse manuel
    eşik olarak kullanılır.
    """
    def prepare_scalar(data, name):
        scalar = np.asarray(data, dtype=np.float32)
        if scalar.ndim == 3:
            scalar = scalar[:, :, 0]
        if scalar.ndim != 2:
            raise ValueError(
                f"Expected a 2-D {name}, got {scalar.shape}."
            )
        return np.ascontiguousarray(scalar, dtype=np.float32)

    h1_scalar = prepare_scalar(image, "H1 scalar image")
    h0_scalar = prepare_scalar(
        image if h0_image is None else h0_image,
        "H0 scalar image",
    )

    h1_lesion_score = h1_scalar if lesion_high else 255.0 - h1_scalar
    h0_lesion_score = h0_scalar if lesion_high else 255.0 - h0_scalar
    h1_lesion_score = np.ascontiguousarray(h1_lesion_score, dtype=np.float32)
    h0_lesion_score = np.ascontiguousarray(h0_lesion_score, dtype=np.float32)

    cubical_complex = CubicalComplex(superlevel=False)
    h1_persistence_info = cubical_complex(torch.from_numpy(h1_lesion_score))

    # ==========================================
    # 1. AŞAMA: EŞİK BELİRLEME (SADECE H1 İLE)
    # ==========================================
    h1_diagram = _persistence_diagram(h1_persistence_info[1])
    if h1_diagram.numel() > 0:
        h1_lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
        valid_h1 = torch.isfinite(h1_diagram).all(dim=1) & (h1_lifetimes > 0)
    else:
        valid_h1 = torch.zeros(0, dtype=torch.bool)

    best_h1_death_coordinate = None
    best_h1_border_distance = None
    best_h1_selection_score = None
    if torch.any(valid_h1):
        valid_indices = torch.nonzero(valid_h1, as_tuple=False).flatten()
        h1_pairing = _persistence_pairing(h1_persistence_info[1])

        # Salt en uzun H1 çubuğu lens/FOV sınırındaki bir deliğe ait olabilir.
        # Sert bir merkez veya kenar eşiği koymadan, persistence değerini ölüm
        # koordinatının sınıra uzaklığıyla yumuşak biçimde ağırlıklandır.
        candidate_scores = []
        candidate_metadata = []
        height, width = h1_lesion_score.shape
        for candidate_index_tensor in valid_indices:
            candidate_index = int(candidate_index_tensor.detach().cpu().item())
            persistence = float(
                h1_lifetimes[candidate_index].detach().cpu().item()
            )
            if candidate_index < len(h1_pairing):
                pair = h1_pairing[candidate_index].detach().cpu().numpy()
            else:
                pair = np.empty(0, dtype=np.float32)

            if pair.size >= 4:
                death_coordinate = pair[-2:].astype(np.float32)
                row, column = death_coordinate
                border_distance = float(
                    min(row, column, height - 1 - row, width - 1 - column)
                )
                border_distance = max(border_distance, 0.0)
            else:
                death_coordinate = None
                border_distance = 0.0

            selection_score = persistence * (border_distance + 1.0)
            candidate_scores.append(selection_score)
            candidate_metadata.append(
                (candidate_index, death_coordinate, border_distance)
            )

        local_best = int(np.argmax(candidate_scores))
        (
            best_index_value,
            best_h1_death_coordinate,
            best_h1_border_distance,
        ) = candidate_metadata[local_best]
        best_h1_selection_score = float(candidate_scores[local_best])
        best_index = torch.tensor(
            best_index_value,
            device=h1_diagram.device,
            dtype=torch.long,
        )
        
        best_birth = float(h1_diagram[best_index, 0].detach().cpu())
        best_death = float(h1_diagram[best_index, 1].detach().cpu())
        best_persistence = best_death - best_birth
        
        # Sadece H1 parametreleriyle threshold hesaplanır
        threshold = best_birth + alpha * best_persistence
        best_h1 = np.array([best_birth, best_death, best_persistence], dtype=np.float32)
    else:
        # H1 bulunamazsa (çok düz bir görselse) Otsu ile yedek threshold
        threshold = float(threshold_otsu(h1_lesion_score))
        best_h1 = np.array([threshold, threshold, 0.0], dtype=np.float32)

    # H1 threshold'una göre ham maskeyi oluştur. Bu, mevcut yöntemin
    # lezyon-yüksek skoru ve >= eşik davranışını korur.
    raw_mask = h0_lesion_score >= threshold
    component_labels = measure.label(raw_mask, connectivity=2)
    (
        spatial_clean_mask,
        spatial_noise_mask,
        h1_reference_label,
        component_anchor_distances,
    ) = h1_spatial_component_filter(
        raw_mask,
        best_h1_death_coordinate,
        h1_lesion_score.shape,
    )

    # H0 padsiz lesion-score'un tersinde hesaplanır. Böylece sublevel
    # h0_filtration <= h0_threshold ile padsiz score >= threshold aynıdır.
    h0_filtration = np.ascontiguousarray(
        255.0 - h0_lesion_score,
        dtype=np.float32,
    )
    h0_threshold = 255.0 - threshold
    h0_persistence_info = cubical_complex(torch.from_numpy(h0_filtration))
    persistence_info = [
        h0_persistence_info[0],
        h1_persistence_info[1],
    ]

    # ==========================================
    # 2. AŞAMA: TEMİZLİK (SADECE H0 İLE)
    # ==========================================
    h0_diagram = _persistence_diagram(persistence_info[0]).detach().cpu().numpy()
    h0_pairing = _persistence_pairing(persistence_info[0]).detach().cpu().numpy()
    pair_count = min(len(h0_diagram), len(h0_pairing))

    label_persistence = {}
    if pair_count:
        births = h0_diagram[:pair_count, 0]
        deaths = h0_diagram[:pair_count, 1]
        effective_deaths = np.where(np.isfinite(deaths), deaths, np.inf)
        lifetimes = effective_deaths - births
        
        # Sadece H1'in bulduğu eşik değerinde "hayatta olan" (aktif) parçalar
        active = (births <= h0_threshold) & (h0_threshold < effective_deaths)

        finite_lifetimes = np.where(
            np.isfinite(lifetimes),
            lifetimes,
            float(np.max(h0_filtration)) - births,
        )

        # Her aktif H0 doğum koordinatını, H1 eşiğinde oluşan gerçek
        # uzamsal bileşene eşle. Aynı bileşene birden fazla bar birleşmişse
        # o bileşen için en büyük persistence değerini sakla.
        for pair, lifetime, is_active in zip(
            h0_pairing[:pair_count],
            finite_lifetimes,
            active,
        ):
            if not is_active:
                continue
            creator = np.asarray(pair[:2], dtype=int)
            row, column = int(creator[0]), int(creator[1])
            if 0 <= row < raw_mask.shape[0] and 0 <= column < raw_mask.shape[1]:
                label_id = int(component_labels[row, column])
                if label_id > 0:
                    label_persistence[label_id] = max(
                        label_persistence.get(label_id, 0.0),
                        float(lifetime),
                    )

    component_persistences = np.asarray(
        list(label_persistence.values()),
        dtype=np.float32,
    )
    raw_otsu_persistence_threshold = None
    otsu_bin_width = 0.0
    if persistence_threshold is not None:
        adaptive_persistence_threshold = float(persistence_threshold)
    elif (
        component_persistences.size >= 2
        and not np.all(component_persistences == component_persistences[0])
    ):
        raw_otsu_persistence_threshold = float(
            threshold_otsu(component_persistences, nbins=256)
        )
        # Otsu'nun kullandığı histogram çözünürlüğüne bağlı tek kutuluk artış:
        # persistence ölçeği değiştiğinde otomatik ölçeklenir ve ayrı bir
        # veri-seti sabiti gerektirmez.
        persistence_range = float(
            np.max(component_persistences) - np.min(component_persistences)
        )
        otsu_bin_width = persistence_range / 256.0
        adaptive_persistence_threshold = (
            raw_otsu_persistence_threshold + otsu_bin_width
        )
    else:
        # Tek bir aktif bileşen varsa karşılaştırılacak bir persistence
        # dağılımı yoktur; bu bileşen korunur.
        adaptive_persistence_threshold = -np.inf

    candidate_noise_labels = set()
    for region in measure.regionprops(component_labels):
        label_id = int(region.label)
        component_persistence = label_persistence.get(label_id)
        is_short_lived = (
            component_persistence is not None
            and component_persistence <= adaptive_persistence_threshold
        )
        if is_short_lived:
            candidate_noise_labels.add(label_id)

    # Persistence tek başına bir bileşenin lezyon mu gürültü mü olduğunu
    # söylemez. H1 ölüm koordinatına en yakın bileşen, H0-Otsu'nun kısa
    # ömürlü grubuna düşse bile referans lezyon olarak korunur.
    noise_labels = set(candidate_noise_labels)
    if h1_reference_label is not None:
        noise_labels.discard(int(h1_reference_label))

    h0_candidate_noise_mask = np.isin(
        component_labels,
        list(candidate_noise_labels),
    )
    h0_noise_mask = np.isin(component_labels, list(noise_labels))
    if noise_labels:
        h0_clean_mask = raw_mask & ~h0_noise_mask
    else:
        h0_clean_mask = raw_mask.copy()

    # H0 ve H1-uzamsal temizliğin ortak tuttuğu bölgeler, iki morfolojik
    # işlemin ve dolayısıyla nihai maskenin doğrudan girdisidir.
    combined_mask = h0_clean_mask & spatial_clean_mask

    # Ufak delikleri ve piksel adacıklarını kapat
    holes_filled_mask = morph.remove_small_holes(
        combined_mask.astype(bool),
        area_threshold=32 * 32,
        connectivity=1,
    )

    # Sabit 64 piksel bazı kopuk blokları koruyabiliyor. Mevcut bağlı
    # bileşenlerin alanlarını Otsu ile küçük/büyük olarak ayır; min_object_size
    # yalnızca güvenli alt sınır olarak kalsın.
    # Dört-komşuluk kullanılır: yalnız köşeden/ince çapraz zincirle ana
    # lezyona değen artefaktlar ayrı nesne olarak değerlendirilebilsin.
    post_hole_labels = measure.label(holes_filled_mask, connectivity=1)
    component_areas = np.asarray(
        [region.area for region in measure.regionprops(post_hole_labels)],
        dtype=np.float32,
    )
    if component_areas.size >= 2 and not np.all(
        component_areas == component_areas[0]
    ):
        adaptive_object_size = max(
            int(min_object_size),
            int(np.ceil(threshold_otsu(component_areas))),
        )
    else:
        adaptive_object_size = int(min_object_size)

    objects_removed_mask = morph.remove_small_objects(
        holes_filled_mask,
        min_size=adaptive_object_size,
        connectivity=1,
    )

    # Alan eşiği, büyük bir lens/FOV parçası varken gerçek lezyonu daha küçük
    # diye silebilir. H1 referans bileşeniyle örtüşen morfolojik nesneleri
    # geri ekleyerek bu hatayı önle.
    protected_morph_labels = set()
    if h1_reference_label is not None:
        reference_component_mask = component_labels == int(h1_reference_label)
        protected_ids = np.unique(
            post_hole_labels[reference_component_mask & holes_filled_mask]
        )
        protected_morph_labels = {
            int(label_id) for label_id in protected_ids if int(label_id) > 0
        }
    if protected_morph_labels:
        protected_morph_mask = np.isin(
            post_hole_labels,
            list(protected_morph_labels),
        )
        final_mask = objects_removed_mask | protected_morph_mask
    else:
        protected_morph_mask = np.zeros_like(holes_filled_mask, dtype=bool)
        final_mask = objects_removed_mask

    result = (
        final_mask.astype(np.uint8),
        persistence_info,
        best_h1,
        np.float32(threshold),
        h0_clean_mask.astype(np.uint8),
    )
    if not return_debug:
        return result

    debug = {
        "raw_h1_mask": raw_mask.astype(np.uint8),
        "h0_candidate_noise_mask": h0_candidate_noise_mask.astype(np.uint8),
        "h0_noise_mask": h0_noise_mask.astype(np.uint8),
        "h0_clean_mask": h0_clean_mask.astype(np.uint8),
        "spatial_noise_mask": spatial_noise_mask.astype(np.uint8),
        "spatial_clean_mask": spatial_clean_mask.astype(np.uint8),
        "combined_before_morphology": combined_mask.astype(np.uint8),
        "holes_filled_mask": holes_filled_mask.astype(np.uint8),
        "objects_removed_mask": final_mask.astype(np.uint8),
        "morph_object_size_threshold": adaptive_object_size,
        "morph_protected_mask": protected_morph_mask.astype(np.uint8),
        "morph_protected_labels": sorted(protected_morph_labels),
        "h1_death_coordinate": best_h1_death_coordinate,
        "h1_border_distance": best_h1_border_distance,
        "h1_selection_score": best_h1_selection_score,
        "h1_reference_label": h1_reference_label,
        "component_anchor_distances": component_anchor_distances,
        "h0_persistence_threshold": float(adaptive_persistence_threshold),
        "h0_raw_otsu_threshold": raw_otsu_persistence_threshold,
        "h0_otsu_bin_width": float(otsu_bin_width),
        "h0_candidate_noise_labels": sorted(candidate_noise_labels),
        "h0_rejected_labels": sorted(noise_labels),
        "h0_component_persistence": dict(label_persistence),
    }
    return (*result, debug)


def otsu_weight(channel):
    t = threshold_otsu(channel)
    fg = channel[channel > t]
    bg = channel[channel <= t]
    w1, w2 = len(fg)/len(channel.flatten()), len(bg)/len(channel.flatten())
    return w1 * w2 * (fg.mean() - bg.mean())**2


def env_flag(name, default=False):
    default_value = "1" if default else "0"
    return os.environ.get(name, default_value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def env_optional_float(name):
    value = os.environ.get(name, "").strip().lower()
    if value in {"", "auto", "otsu", "none"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number or one of: auto, otsu, none."
        ) from exc


def resolve_ml_data_root():
    """Resolve the dataset root consistently in terminals and VS Code.

    The public repository has no machine-specific fallback.  ``ML_DATA_ROOT``
    must point to the directory containing the dataset folders.
    """
    configured_root = os.environ.get("ML_DATA_ROOT", "").strip()
    if not configured_root:
        raise EnvironmentError(
            "ML_DATA_ROOT must point to the directory containing the datasets"
        )
    data_root = os.path.abspath(os.path.expanduser(configured_root))

    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"Dataset root from ML_DATA_ROOT does not exist: {data_root}. "
            "Set ML_DATA_ROOT to the directory containing the datasets."
        )

    return data_root


def maps(
    dataset,
    image_path,
    mask_path,
    idx,
    size=(256, 256),
    show_cleanup_theory=True,
    show_scalar_explanation=True,
):

    pil_img = Image.open(image_path).convert("RGB").resize(
        size,
        Image.Resampling.BILINEAR,
    )
    real_mask = Image.open(mask_path).convert("L").resize(
        size,
        Image.Resampling.NEAREST,
    )
    real_mask = np.array(real_mask)

    # Convert to tensor (automatically scales 0–255 → 0–1 and makes shape [C,H,W])
    # Apply dullrazor (assuming it expects [C,H,W] tensor in float32, 0–1 range)

    to_tensor = T.ToTensor()
    img_tensor = to_tensor(pil_img)
   # c_arr = (img_tensor * 255).astype(np.uint8)
    if dataset=="isic_2018_1" or dataset=="PH2Dataset" or dataset=="isic_2016_1":
        clean_img = dullrazor(img_tensor)
        clean_img = clean_img.permute(1, 2, 0).cpu().numpy()
    else:
        clean_img = img_tensor.permute(1, 2, 0).cpu().numpy()

    arr = np.array(
        to_pil_image((clean_img * 255).astype(np.uint8)).convert("L")
    )
    gray_variants = build_gray_variants(clean_img)
    selected_variant = os.environ.get(
        "GRAY_VARIANT",
        "robust_delta_e_s_v",
    )
    if selected_variant not in gray_variants:
        valid = ", ".join(gray_variants)
        raise ValueError(
            f"Unknown GRAY_VARIANT={selected_variant!r}. Valid values: {valid}."
        )
    gray = gray_variants[selected_variant]
    background_skin_rgb = (
        estimate_background_skin_rgb(clean_img)
        if selected_variant == "robust_delta_e_s_v"
        else None
    )

    show_diagnostic_plot = env_flag(
        "SHOW_DIAGNOSTIC_PLOT",
        default=SHOW_DIAGNOSTIC_PLOT,
    )
    show_comparison_plot = env_flag(
        "SHOW_COMPARISON_PLOT",
        default=SHOW_COMPARISON_PLOT,
    )
    show_scalar_plot = show_scalar_explanation and env_flag(
        "SHOW_SCALAR_EXPLANATION_PLOT",
        default=SHOW_SCALAR_EXPLANATION_PLOT,
    )
    show_cleanup_plot = env_flag(
        "SHOW_CLEANUP_PLOT",
        default=SHOW_CLEANUP_PLOT,
    )

    pad_size = 2
    padded   = adaptive_pad(gray, pad_size,5)
    rpadded  = resize(padded, (256, 256),order=1, preserve_range=True, anti_aliasing=True).astype(gray.dtype)

    input_image = rpadded
    name = f"Topological Segmentation ({selected_variant})"

    random_walker_mask = None
    morphological_mask = None
    chc_otsu_mask = None
    grabcut_mask = None
    watershed_mask = None
    otsu_mask = None

    if show_diagnostic_plot or show_comparison_plot:
        # Use exactly the same scalar field and orientation for every baseline.
        # This keeps the qualitative figure consistent with the controlled
        # comparison reported in Table 1.
        shared_scalar = np.rint(np.clip(gray, 0.0, 255.0)).astype(np.uint8)
        shared_lesion_high = lesion_is_high(selected_variant)
        morphological_mask, _ = morphological_chan_vese_segmentation(shared_scalar)
        random_walker_mask = random_walker_pseudo_mask(
            shared_scalar,
            lesion_high=shared_lesion_high,
        )

    if show_comparison_plot:
        chc_otsu_mask, _ = chc_otsu_pseudo_mask(shared_scalar)
        grabcut_mask = grabcut_pseudo_mask(shared_scalar)
        watershed_mask = watershed_pseudo_mask(
            shared_scalar,
            lesion_high=shared_lesion_high,
        )

    if show_diagnostic_plot:
        th = threshold_otsu(gray)
        otsu_mask = gray > th if lesion_is_high(selected_variant) else gray <= th
        otsu_mask = morph.remove_small_holes(
            otsu_mask.astype(bool),
            area_threshold=32 * 32,
        ).astype(np.uint8)
        otsu_mask = morph.remove_small_objects(
            otsu_mask.astype(bool),
            min_size=64,
            connectivity=2,
        ).astype(np.uint8)

    if "H0_PERSISTENCE_THRESHOLD" in os.environ:
        manual_h0_threshold = env_optional_float("H0_PERSISTENCE_THRESHOLD")
    else:
        manual_h0_threshold = H0_PERSISTENCE_THRESHOLD
    cc_mask, pi, bests, topology_threshold, h0_mask, topology_debug = cubical_complex_segmentation(
        input_image,
        h0_image=gray,
        persistence_threshold=manual_h0_threshold,
        alpha=0.20,
        lesion_high=lesion_is_high(selected_variant),
        return_debug=True,
    )

    print(f"--- Algoritma {name} için sonuçlar ---")
    h0_mode = "manual" if manual_h0_threshold is not None else "automatic Otsu"
    print(
        "H0 persistence threshold "
        f"({h0_mode}): {topology_debug['h0_persistence_threshold']:.3f}"
    )

    if any(
        (
            show_diagnostic_plot,
            show_comparison_plot,
            show_scalar_plot,
            show_cleanup_plot,
        )
    ):
        plt.close("all")

    if show_diagnostic_plot:
        plot(
            name,
            gray,
            img_tensor,
            clean_img,
            None,
            None,
            None,
            h0_mask,
            random_walker_mask,
            morphological_mask,
            otsu_mask,
            cc_mask,
            real_mask,
            bests,
            pi,
            topology_threshold,
            raw_h1_mask=topology_debug["raw_h1_mask"],
            h0_noise_mask=topology_debug["h0_noise_mask"],
            spatial_noise_mask=topology_debug["spatial_noise_mask"],
            spatial_clean_mask=topology_debug["spatial_clean_mask"],
            combined_mask=topology_debug["combined_before_morphology"],
            objects_removed_mask=topology_debug["objects_removed_mask"],
            morph_object_size_threshold=topology_debug["morph_object_size_threshold"],
            h1_anchor=topology_debug["h1_death_coordinate"],
            h0_persistence_threshold=topology_debug["h0_persistence_threshold"],
            background_skin_rgb=background_skin_rgb,
            show=True,
        )

    if show_comparison_plot:
        plot_pseudo_mask_comparison(
            img_tensor=img_tensor,
            real_mask=real_mask,
            topological_mask=cc_mask,
            random_walker_mask=random_walker_mask,
            morphological_mask=morphological_mask,
            chc_otsu_mask=chc_otsu_mask,
            grabcut_mask=grabcut_mask,
            watershed_mask=watershed_mask,
            show=True,
        )

    if show_scalar_plot:
        plot_scalar_field_explanation(
            img_tensor,
            clean_img,
            gray,
            background_skin_rgb,
            variant_name=selected_variant,
            show=True,
        )

    if show_cleanup_plot:
        plot_topological_cleanup_explanation(
            gray,
            pi,
            bests,
            topology_debug["raw_h1_mask"],
            topology_debug["h0_noise_mask"],
            topology_debug["spatial_noise_mask"],
            cc_mask,
            real_mask,
            h1_anchor=topology_debug["h1_death_coordinate"],
            include_theory=show_cleanup_theory,
            show=True,
        )

    if any(
        (
            show_diagnostic_plot,
            show_comparison_plot,
            show_scalar_plot,
            show_cleanup_plot,
        )
    ):
        plt.close("all")

    print("next", idx)
    return cc_mask

def main():
    dataset = "isic_2018_1"  # Change dataset here if needed
    main_path = os.path.join(resolve_ml_data_root(), dataset)

    im_train_path    = os.path.join(main_path, "train/images")
    masks_train_path = os.path.join(main_path, "train/masks")
    images_list = [f for f in sorted(os.listdir(im_train_path)) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    masks_list = [f for f in sorted(os.listdir(masks_train_path)) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    idx = 60 # Change starting index here if needed#
    for i, (img_name, mask_name) in enumerate(zip(images_list[idx:], masks_list[idx:])):
        print(f"[{i+1}/{len(images_list)}] Processing {img_name}")
        img_path = os.path.join(im_train_path, img_name)
        mask_path = os.path.join(masks_train_path, mask_name)
        maps(
            dataset,
            img_path,
            mask_path,
            idx,
            show_cleanup_theory=(i == 0),
            show_scalar_explanation=(i == 0),
        )
        idx+=1

if __name__ == "__main__":
    pcs = main()


#196 730biyi çalışan örnek-- 603 138 142 1364 1267 1347 1356 1731 1015 126 132 

#1165 246! 247!iyi örnek. 466 güzel örnek otsudan iyi

#174
#165
#178
#191
#1164, 1165, 1178 339 847 1122 372 506 809 808 1117 1724 1738 462 911 1292 188 1007  
# 647 1728 246! 247! 165 lesyon kenarda
# 1790 1074 1291 marking
# 1119 belirsiz alan - kenarda
# threshoulding
# 952 ??
#1191
#172,182
#1178 sonrası incele!!


#1731

#119+8 problem +15 +19 +35!!   133 137 142 1598 210

#339 sonrası 346!!
#799
#1731
#220, 181!!!

#597 

# 603 138 142 1364 1267 1347 1356 1731 1015 126 132 

#1165 246! 247!iyi örnek. 466 güzel örnek otsudan iyi

#99 770 1429 1211 464 346 691

# 98 1429 problem with channel combinations. 
# gray problem 746 828 714 1108 261 1469

#130 ;!! GRA

###görseller
# clean_persistence_1 - 1235
# clean_persistence_2 - 111
# clean_persistence_3 - 777
# clean_persistence_4 - 77

#noise_removal_color_fusion_1 - 1017
#noise_removal_color_fusion_2 - 1339
#noise_removal_color_fusion_3 - 1093

#topological_segmentation_1 - 246
#topological_segmentation_2 - 247
#topological_segmentation_3 - 248
#topological_segmentation_4 - 604
