import numpy as np
from skimage.transform import resize

def unpad_resize(rpadded, orig_shape, pad=1, target_size=256):
    """
    rpadded    : resize edilmiş pad'li görüntü (örn. 256x256)
    orig_shape : (h, w) orijinal görüntü boyutu (pca_ch_.shape)
    pad        : her kenardaki pad miktarı (default=1)
    target_size: resize edilmiş kare boyutu (default=256)
    """
    h, w = orig_shape
    H, W = rpadded.shape  # genelde (target_size, target_size)

    # oranlarla crop sınırlarını hesapla
    top    = int(H * pad / (h + 2*pad))
    bottom = int(H * (h + pad) / (h + 2*pad))
    left   = int(W * pad / (w + 2*pad))
    right  = int(W * (w + pad) / (w + 2*pad))

    # crop pad kısmını çıkar
    cropped = rpadded[top:bottom, left:right]

    # Binary masks must use nearest-neighbour interpolation. Bilinear resize
    # followed by an integer cast can erase thin connections and create extra
    # connected components, directly changing beta_0 and beta_1.
    unique_values = np.unique(cropped)
    is_binary = unique_values.size <= 3 and np.all(
        np.isin(unique_values, [0, 1, 255])
    )
    restored = resize(
        cropped,
        (h, w),
        order=0 if is_binary else 1,
        preserve_range=True,
        anti_aliasing=False if is_binary else True,
    )
    if is_binary:
        threshold = 0.5 if restored.max(initial=0) <= 1.0 else 127.5
        restored = restored > threshold
        if np.issubdtype(rpadded.dtype, np.integer):
            return restored.astype(rpadded.dtype)
        return restored.astype(np.float32)
    return restored.astype(rpadded.dtype)

import numpy as np

def adaptive_pad(img, pad=5, win=5):
    """
    Edge-aware adaptive padding.
    If all edge decisions result in 0-padding, padding is skipped.
    """
    H, W = img.shape
    mean_global, mean_std = img.mean(), img.std()

    out = np.zeros((H + 2 * pad, W + 2 * pad), dtype=img.dtype)
    out[pad:pad+H, pad:pad+W] = img

    decisions = []  # store True if pad=255, False if pad=0

    def decide(region):
        if np.all(region == 255):
            return 255
        return 0 if region.mean() > (mean_global - mean_std/2) else 255

    # ---- Top ----
    for x in range(0, W, win):
        x_end = min(x + win, W)
        val = decide(img[:pad, x:x_end])
        decisions.append(val == 255)
        out[:pad, pad+x:pad+x_end] = val

    # ---- Bottom ----
    for x in range(0, W, win):
        x_end = min(x + win, W)
        val = decide(img[-pad:, x:x_end])
        decisions.append(val == 255)
        out[H+pad:, pad+x:pad+x_end] = val

    # ---- Left ----
    for y in range(0, H, win):
        y_end = min(y + win, H)
        val = decide(img[y:y_end, :pad])
        decisions.append(val == 255)
        out[pad+y:pad+y_end, :pad] = val

    # ---- Right ----
    for y in range(0, H, win):
        y_end = min(y + win, H)
        val = decide(img[y:y_end, -pad:])
        decisions.append(val == 255)
        out[pad+y:pad+y_end, W+pad:] = val

    # ---- GLOBAL FAIL-SAFE ----
    if not any(decisions):
        # All padding decisions were zero → skip padding
        return img.copy()

    return out


def upsample_patch_map(patch_map, patch_size, target_shape):
    """
    Expand patch-level map (H/ps, W/ps) to pixel-level (H, W).
    Each patch gets the same value in its region.
    """
    H, W = target_shape
    h_patches, w_patches = patch_map.shape
    expanded = np.zeros((H, W), dtype=patch_map.dtype)

    for i in range(h_patches):
        for j in range(w_patches):
            expanded[i*patch_size:(i+1)*patch_size,
                     j*patch_size:(j+1)*patch_size] = patch_map[i, j]
    return expanded
