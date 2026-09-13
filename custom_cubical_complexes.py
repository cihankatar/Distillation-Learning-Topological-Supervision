import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt

def custom_cubical_complex(img, alpha=0.7, radius=1.5):
    """
    img: 2D numpy array (0-255 arası)
    alpha: intensity vs distance ağırlığı
    radius: komşuluk yarıçapı (piksel)
    return: best H0 birth, best H0 death (0-255 arası)
    """
    H, W = img.shape
    values = img.flatten()
    coords = [(i,j) for i in range(H) for j in range(W)]
    n = len(values)
    max_dist = np.sqrt((H-1)**2 + (W-1)**2)

    st = gd.SimplexTree()

    # Vertex ekle (0-255 filtre)
    for i, val in enumerate(values):
        st.insert([i], filtration=val)

    # Edge ekle (lokal pencere)
    for i in range(H):
        for j in range(W):
            idx = i*W + j
            for di in [-1,0,1]:
                for dj in [-1,0,1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i+di, j+dj
                    if 0 <= ni < H and 0 <= nj < W:
                        nidx = ni*W + nj
                        # intensity farkını 0-255 arası
                        diff = abs(float(values[idx]) - float(values[nidx]))
                        # mesafe 0–radius → 0–255
                        dist_scaled = (np.sqrt(di**2 + dj**2) / radius) * 255
                        dist_scaled = min(dist_scaled, 255)  # radius dışında clip
                        # filtrasyon: α·intensity + (1-α)·distance
                        filt = alpha * diff + (1 - alpha) * dist_scaled
                        st.insert([idx,nidx], filtration=filt)

    st.make_filtration_non_decreasing()

    # Persistence hesapla
    diag = st.persistence()

    # Persistence diagram çiz
    gd.plot_persistence_diagram(diag)
    plt.title("Persistence Diagram (H0 only, 0-255)")
    plt.show()  
    
    h0_intervals = st.persistence_intervals_in_dimension(0)

    # Sonsuz intervali at
    finite_intervals = [iv for iv in h0_intervals if np.isfinite(iv[1])]

    if len(finite_intervals) == 0:
        # tüm interval sonsuz, örnek olarak vertex max değeri döndürülebilir
        return np.max(values), np.max(values)

    # En uzun intervali bul
    lifetimes = [iv[1]-iv[0] for iv in finite_intervals]
    best_idx = np.argmax(lifetimes)
    best_birth, best_death = finite_intervals[best_idx]

    return best_birth, best_death  # 0-255 aralığında
