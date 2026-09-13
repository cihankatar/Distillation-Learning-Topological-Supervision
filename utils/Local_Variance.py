import numpy as np
from skimage.util import view_as_windows
from scipy import ndimage


def local_variance(img, win_size=13, combine="max", mode="gradproj", eps=1e-6):
    """
    Çok kanallı (ör. RGB) görüntü için lokal varyans hesaplar.

    Args:
        img (np.ndarray): HxWxC görüntü (C kanal sayısı).
        win_size (int): Lokal pencere boyutu (tek sayı olmalı).
        combine (str): Kanallar arası nasıl birleştirileceği.
                       "mean" -> ortalama, "max" -> maksimum, None -> ayrı döner
        mode (str): "window" (klasik)
                    "gradproj" (gradyana dik projeksiyon, hızlı)
                    "pca_smooth" (smoothed structure tensor)

    Returns:
        np.ndarray: HxW (tek kanal) veya HxWxC (combine=None ise)
    """
    if img.ndim == 2:  # grayscale
        img = img[..., None]

    H, W, C = img.shape
    vars_per_channel = []

    half = win_size // 2
    coords = np.stack(np.meshgrid(np.arange(-half, half+1),
                                  np.arange(-half, half+1),
                                  indexing='ij'), axis=-1)  # (win,win,2)
    dx, dy = coords[..., 0], coords[..., 1]

    for c in range(C):
        channel = img[..., c]

        if mode == "window":
            padded = np.pad(channel, half, mode='reflect')
            windows = view_as_windows(padded, (win_size, win_size))
            mean = windows.mean(axis=(-1, -2))
            var = ((windows - mean[..., None, None])**2).mean(axis=(-1, -2))

        elif mode == "gradproj":
            # hızlı gradient-based projeksiyon
            gx = ndimage.sobel(channel, axis=1)
            gy = ndimage.sobel(channel, axis=0)
            theta = np.arctan2(gy, gx)  # rad

            padded = np.pad(channel, half, mode='reflect')
            windows = view_as_windows(padded, (win_size, win_size))

            var = np.zeros((H, W))
            for i in range(H):
                for j in range(W):
                    patch = windows[i, j]
                    # gradyana dik yön
                    t = theta[i, j]
                    u = dx * np.cos(t) + dy * np.sin(t)
                    # pikselleri o eksende grupla
                    order = np.argsort(u.ravel())
                    line = patch.ravel()[order]
                    var[i, j] = line.var()

        elif mode == "pca":
            gx = ndimage.sobel(channel, axis=1)
            gy = ndimage.sobel(channel, axis=0)
            # Structure tensor elemanları (smooth yok)
            Ixx = gx * gx
            Iyy = gy * gy
            Ixy = gx * gy


            tmp = np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2)
            lambda1 = 0.5*(Ixx + Iyy + tmp)
            lambda2 = 0.5*(Ixx + Iyy - tmp)

            # daha stabil metrik
            var = (lambda1 + lambda2)  # toplam varyans
            # coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)


        elif mode == "pca_smooth":
            gx = ndimage.sobel(channel, axis=1)
            gy = ndimage.sobel(channel, axis=0)

            sigma = win_size / 7.0
            Ixx = ndimage.gaussian_filter(gx*gx, sigma=sigma)
            Iyy = ndimage.gaussian_filter(gy*gy, sigma=sigma)
            Ixy = ndimage.gaussian_filter(gx*gy, sigma=sigma)

            tmp = np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2)
            lambda1 = 0.5*(Ixx + Iyy + tmp)
            lambda2 = 0.5*(Ixx + Iyy - tmp)

            # daha stabil metrik
            var = (lambda1 + lambda2)  # toplam varyans
            # coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + eps)


        else:
            raise ValueError(f"Unknown mode {mode}")

        vars_per_channel.append(var)

    vars_per_channel = np.stack(vars_per_channel, axis=-1)  # HxWxC

    if combine == "mean":
        return vars_per_channel.mean(axis=-1)
    elif combine == "max":
        return vars_per_channel.max(axis=-1)
    else:
        return vars_per_channel

def local_variance_(img, win_size=21, combine="max"):
    """
    Çok kanallı (ör. RGB) görüntü için lokal varyans hesaplar.

    Args:
        img (np.ndarray): HxWxC görüntü (C kanal sayısı).
        win_size (int): Lokal pencere boyutu (tek sayı olmalı).
        combine (str): Kanallar arası nasıl birleştirileceği.
                       "mean" -> ortalama, "max" -> maksimum, None -> ayrı döner

    Returns:
        np.ndarray: HxW (tek kanal) veya HxWxC (combine=None ise)
    """
    if img.ndim == 2:  # grayscale durumunda
        img = img[..., None]  # HxW -> HxWx1

    H, W, C = img.shape
    vars_per_channel = []

    for c in range(C):
        channel = img[..., c]
        padded = np.pad(channel, win_size//2, mode='reflect')
        windows = view_as_windows(padded, (win_size, win_size))
        mean = windows.mean(axis=(-1, -2))
        var = ((windows - mean[..., None, None])**2).mean(axis=(-1, -2))
        vars_per_channel.append(var)

    vars_per_channel = np.stack(vars_per_channel, axis=-1)  # HxWxC

    if combine == "mean":
        return vars_per_channel.mean(axis=-1)
    elif combine == "max":
        return vars_per_channel.max(axis=-1)
    else:
        return vars_per_channel  # ayrı ayrı döndür (HxWxC)


# sample usage.
    # var_map = local_variance(clean_img, win_size=3, combine="max", mode="window", eps=1e-6) # max  variance across channels
    # var_norm = (var_map - var_map.min()) / (var_map.max() - var_map.min() + 1e-8)
    # pca_ch = np.clip(pca_ch_ + 0.3 * var_norm, 0, 1)*255 # pca channel enhanced
