"""Scalar image variants used by topology-guided pseudo-mask generation."""

import cv2
import numpy as np
from skimage import color


GRAY_VARIANT_TITLES = {
    "rgb_luma": "RGB luma",
    "inverse_lab_l": "Inverse LAB L",
    "inverse_hsv_v": "Inverse HSV V",
    "s_plus_inverse_v": "S + inverse V",
    "current_lab_hsv": "Current LAB-HSV fusion",
    "robust_delta_e_s_v": "Robust DeltaE + S + inverse V",
}


# Whether a scalar field already represents the lesion with large values.
# ``cubical_complex_segmentation`` converts every variant to this common
# lesion-high orientation before H1 thresholding.
LESION_HIGH_VARIANTS = {
    "rgb_luma": False,
    "inverse_lab_l": True,
    "inverse_hsv_v": True,
    "s_plus_inverse_v": True,
    "current_lab_hsv": True,
    "robust_delta_e_s_v": True,
}


def minmax_normalize(channel):
    """Normalize a channel to [0, 1] using its full image range."""
    channel = np.asarray(channel, dtype=np.float32)
    low = float(np.nanmin(channel))
    high = float(np.nanmax(channel))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0.0, 1.0)


def robust_normalize(channel, lower_percentile=1.0, upper_percentile=99.0):
    """Percentile-normalize a channel while limiting isolated outliers."""
    channel = np.asarray(channel, dtype=np.float32)
    finite = channel[np.isfinite(channel)]
    if finite.size == 0:
        return np.zeros_like(channel, dtype=np.float32)

    low, high = np.percentile(
        finite,
        [lower_percentile, upper_percentile],
    )
    if high <= low:
        return np.zeros_like(channel, dtype=np.float32)
    return np.clip((channel - low) / (high - low), 0.0, 1.0)


def _border_background_lab(lab, value):
    """Estimate skin/background colour robustly from a narrow image border."""
    height, width = value.shape
    border_width = max(3, int(round(min(height, width) * 0.08)))
    border = np.zeros((height, width), dtype=bool)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True

    border_values = value[border]
    darkness_limit = max(0.05, float(np.percentile(border_values, 10.0)))
    valid_border = border & (value > darkness_limit)
    samples = lab[valid_border]
    if samples.size == 0:
        samples = lab[border]
    return np.median(samples, axis=0)


def estimate_background_skin_rgb(clean_rgb):
    """Return the RGB colour represented by the robust border-skin model."""
    rgb = np.asarray(clean_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {rgb.shape}.")
    if float(np.nanmax(rgb)) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    lab = color.rgb2lab(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    background_lab = _border_background_lab(lab, hsv[:, :, 2])
    background_rgb = color.lab2rgb(background_lab.reshape(1, 1, 3))[0, 0]
    return np.clip(background_rgb, 0.0, 1.0).astype(np.float32)


def build_gray_variants(clean_rgb):
    """Return the six requested scalar image representations in [0, 255]."""
    rgb = np.asarray(clean_rgb, dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, got {rgb.shape}.")

    if float(np.nanmax(rgb)) > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = color.rgb2lab(rgb)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    lightness = lab[:, :, 0]
    channel_a = lab[:, :, 1]
    channel_b = lab[:, :, 2]

    rgb_luma = (
        0.2989 * rgb[:, :, 0]
        + 0.5870 * rgb[:, :, 1]
        + 0.1140 * rgb[:, :, 2]
    )

    robust_l = robust_normalize(lightness)
    robust_s = robust_normalize(saturation)
    robust_v = robust_normalize(value)

    current_a = minmax_normalize(channel_a)
    current_b = minmax_normalize(channel_b)
    current_s = minmax_normalize(saturation)
    current_v_inv = 1.0 - minmax_normalize(value)

    background_lab = _border_background_lab(lab, value)
    delta_e = np.linalg.norm(lab - background_lab[None, None, :], axis=2)
    robust_delta_e = robust_normalize(delta_e)

    variants = {
        "rgb_luma": rgb_luma,
        "inverse_lab_l": 1.0 - robust_l,
        "inverse_hsv_v": 1.0 - robust_v,
        "s_plus_inverse_v": 0.5 * (robust_s + (1.0 - robust_v)),
        "current_lab_hsv": (
            0.15 * current_a
            + 0.15 * current_b
            + 0.30 * current_s
            + 0.40 * current_v_inv
        ),
        "robust_delta_e_s_v": (
            robust_delta_e + robust_s + (1.0 - robust_v)
        ) / 3.0,
    }

    return {
        name: (np.clip(field, 0.0, 1.0) * 255.0).astype(np.float32)
        for name, field in variants.items()
    }


def lesion_is_high(variant_name):
    """Return whether larger values indicate stronger lesion likelihood."""
    try:
        return LESION_HIGH_VARIANTS[variant_name]
    except KeyError as exc:
        valid = ", ".join(LESION_HIGH_VARIANTS)
        raise ValueError(
            f"Unknown gray variant {variant_name!r}. Valid values: {valid}."
        ) from exc
