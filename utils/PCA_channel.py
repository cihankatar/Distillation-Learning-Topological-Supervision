
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

import numpy as np
from sklearn.decomposition import PCA
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_opening, disk
from sklearn.decomposition import NMF

def NMF_channel(img, gray):

    A, B, S = img[..., 0], img[..., 1], img[..., 2]
    X = np.stack([A.flatten(), B.flatten(), S.flatten()], axis=1).astype(float)
    # normalize 0-1
    X = (X - X.min()) / (X.ptp()+1e-12)
    H, W = A.shape

    nmf = NMF(n_components=1, init='nndsvda', random_state=0, max_iter=500)
    W = nmf.fit_transform(X)   # N x 1
    component = np.array(W).reshape(H, H)
    return (component - component.min()) / (component.max() - component.min() + 1e-12)


def pca_channel(img):
    """
    Apply PCA to reduce multi-channel image (2 or 3 channels) to 1 channel.
    Ensure PCA direction aligns with reference gray image (same bright/dark orientation).
    """
    h, w, c = img.shape
    flat = img.reshape(-1, c).astype(np.float32)
    
    # PCA -> 1 bileşen
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(flat)
    pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-12)
    pc1_img = pc1.reshape(h, w)
    
    # --- Gray ile yön hizalama ---
    # g = gray.astype(np.float32)
    # g = (g - g.min()) / (g.max() - g.min() + 1e-12)
    
    # corr = np.corrcoef(pc1_img.ravel(), gray.ravel())[0, 1]
    # if corr < 0:
    #     pc1_img = 1 - pc1_img  # yön ters ise düzelt
        
    return pc1_img

def best_channel(c_reversed):
    R, G, B = c_reversed[...,0], c_reversed[...,1], c_reversed[...,2]
    channels = [R, G, B]
    contrasts = [np.std(ch) for ch in channels]
    best_idx = np.argmax(contrasts)
    best_channel = channels[best_idx]
    min_val = best_channel.min()
    max_val = best_channel.max()
    best_channel_norm = (((best_channel - min_val) / (max_val - min_val + 1e-8) * 255).astype(np.uint8))/255

    return best_channel_norm