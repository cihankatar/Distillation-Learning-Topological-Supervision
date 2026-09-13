import numpy as np
import cv2
import skimage.segmentation as seg
import skimage.filters as filters
import skimage.morphology as morph
from skimage.segmentation import morphological_chan_vese
import torch
from skimage import color
from skimage.filters import threshold_otsu
from PIL import Image   
from scipy import ndimage as ndi

def medsam_pseudo_mask(img_np, gray_for_otsu, model, processor, device):
    """
    1. Otsu ile kaba bir maske çıkarır.
    2. O maskeden bir Bounding Box (Kutu) hesaplar.
    3. Resmi ve Kutuyu MedSAM'e verip kusursuz maskeyi alır.
    """
    H, W = gray_for_otsu.shape
    
    # Adım 1: Otsu ile kaba maske ve Kutu (Bounding Box) Bulma
    try:
        th_val = threshold_otsu(gray_for_otsu)
        rough_mask = (gray_for_otsu > th_val).astype(np.uint8)
        
        # Kutunun koordinatlarını bul (y_min, x_min, y_max, x_max)
        y_indices, x_indices = np.where(rough_mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("Otsu failed to find a foreground.")
            
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # MedSAM'in bağlamı (context) daha iyi anlaması için kutuya biraz pay (padding) ekliyoruz
        pad = 10
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(W, x_max + pad)
        y_max = min(H, y_max + pad)
        
        input_boxes = [[[x_min, y_min, x_max, y_max]]]
        
    except Exception:
        # Eğer Otsu çuvallarsa, görüntünün ortasını (center crop) kutu olarak ver
        input_boxes = [[[int(W*0.2), int(H*0.2), int(W*0.8), int(H*0.8)]]]

    # Adım 2: MedSAM'e Resmi ve Kutuyu Gönderme
    # MedSAM 3 kanallı RGB resim ister.
    if len(img_np.shape) == 2:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_np
        
    # Resmi PIL formatına çevir
    img_pil = Image.fromarray(img_rgb)
    
    # İşlemciye (Processor) verileri hazırla
    inputs = processor(img_pil, input_boxes=[input_boxes], return_tensors="pt").to(device)
    
    # Modeli çalıştır (No gradient computation for speed)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Adım 3: Çıktıyı İşleme
    # MedSAM her kutu için 3 farklı maske olasılığı döndürür, en yüksek skorluyu alıyoruz
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"].cpu(), 
        inputs["reshaped_input_sizes"].cpu()
    )
    
    # İlk resim, İlk kutu, En iyi maske (indeks 0)
    best_mask = masks[0][0][0].numpy() 
    
    final_mask = best_mask.astype(np.uint8)
    
    # Ufak temizlikler (isteğe bağlı)
    final_mask = morph.remove_small_holes(final_mask.astype(bool), area_threshold=32*32).astype(np.uint8)
    
    return final_mask

def morphological_chan_vese_segmentation(arr):
    """
    Perform Morphological Chan-Vese segmentation on a 2D grayscale image.
    Args:
        arr (np.ndarray): 2D array representing the grayscale image.
    Returns:
        np.ndarray: Binary mask of the segmented region.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(arr)

    height, width = arr.shape
    init_mask = np.zeros(arr.shape, dtype=bool)
    init_mask[
        height // 3 : 2 * height // 3,
        width // 3 : 2 * width // 3,
    ] = True  # Resolution-independent central initialization.
    final_mask = morphological_chan_vese(enhanced, 200, init_level_set=init_mask)

    final_mask = morph.remove_small_holes(final_mask.astype(bool), area_threshold=32*32).astype(np.uint8)
    final_mask = morph.remove_small_objects(final_mask.astype(bool), min_size=64).astype(np.uint8)

    return final_mask, enhanced

def random_walker_pseudo_mask(img_np, lesion_high=False):
    """
    Convert a 2D scalar field to a pseudo-mask via Random Walker.

    ``lesion_high`` specifies the orientation of the shared scalar field.  The
    marker labels always use 1 for lesion and 2 for background, so the same
    input field can be compared fairly with the other segmentation methods.
    """
    scalar = np.asarray(img_np, dtype=np.float32)
    thresh = float(filters.threshold_otsu(scalar))
    delta = 0.10 * float(scalar.max() - scalar.min())
    markers = np.zeros_like(scalar, dtype=np.int32)

    if lesion_high:
        markers[scalar > (thresh + delta)] = 1
        markers[scalar < (thresh - delta)] = 2
        fallback_mask = scalar >= thresh
    else:
        markers[scalar < (thresh - delta)] = 1
        markers[scalar > (thresh + delta)] = 2
        fallback_mask = scalar <= thresh

    # Random Walker requires at least one seed for both classes.  Degenerate
    # fields fall back to the orientation-aware Otsu result.
    if not np.any(markers == 1) or not np.any(markers == 2):
        return _clean_binary_mask(fallback_mask)

    rw_labels = seg.random_walker(scalar, markers, beta=90, mode='bf')
    final_mask = (rw_labels == 1).astype(np.uint8)
    return _clean_binary_mask(final_mask)


def _clean_binary_mask(mask, hole_area=32 * 32, min_object_size=64):
    """Apply the common binary cleanup used by the classical baselines."""
    cleaned = morph.remove_small_holes(
        np.asarray(mask, dtype=bool),
        area_threshold=hole_area,
    )
    cleaned = morph.remove_small_objects(
        cleaned,
        min_size=min_object_size,
    )
    return cleaned.astype(np.uint8)


def grabcut_pseudo_mask(img_np):
    """
    GrabCut algoritması kullanarak merkez odaklı pseudo-mask üretimi.
    Görüntünün ortasını ön plan (lezyon), kenarlarını arka plan varsayar.
    """
    # GrabCut 8-bit 3 kanallı görüntü (BGR/RGB) ister
    if img_np.dtype != np.uint8:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    mask = np.zeros(img_np.shape[:2], np.uint8)
    
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    
    # Dikdörtgen (Bounding Box) belirleme: Görüntünün kenarlarından %10 boşluk bırak
    H, W = img_np.shape
    margin_h, margin_w = int(H * 0.1), int(W * 0.1)
    rect = (margin_w, margin_h, W - 2 * margin_w, H - 2 * margin_h)
    
    # 5 iterasyon ile GrabCut çalıştır
    cv2.grabCut(img_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    # GrabCut çıktısı: 0=Kesin Arka plan, 2=Muhtemel Arka plan, 1=Kesin Ön plan, 3=Muhtemel Ön plan
    # 1 ve 3'ü maske olarak alıyoruz
    final_mask = np.where((mask == 1) | (mask == 3), 1, 0).astype(np.uint8)
    
    # Cleanup
    final_mask = morph.remove_small_holes(final_mask.astype(bool), area_threshold=32*32).astype(np.uint8)
    final_mask = morph.remove_small_objects(final_mask.astype(bool), min_size=64).astype(np.uint8)
    
    return final_mask

def watershed_pseudo_mask(img_np, lesion_high=False):
    """
    Watershed dönüşümü ile morfolojik tabanlı maske üretimi.
    Birbirine yakın veya içiçe geçmiş renk geçişlerini iyi ayırır.
    """
    if img_np.dtype != np.uint8:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    # Orient the Otsu foreground without changing the shared scalar field.
    threshold_mode = cv2.THRESH_BINARY if lesion_high else cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(img_np, 0, 255, threshold_mode + cv2.THRESH_OTSU)
    
    # Gürültü temizleme (Opening)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Kesin arka plan alanını bul (Dilatasyon ile genişletilmiş)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Kesin ön plan (lezyon) alanını Mesafe Dönüşümü (Distance Transform) ile bul
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Belirsiz bölgeyi bul
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Marker (tohum) etiketlemesi
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img_bgr, markers)
    
    # Watershed sınırı -1 yapar, arka plan 1 olur. Biz >1 olanları lezyon (ön plan) sayıyoruz
    final_mask = np.zeros_like(img_np, dtype=np.uint8)
    final_mask[markers > 1] = 1
    
    return _clean_binary_mask(final_mask)

def kmeans_pseudo_mask(img_np):
    """
    K-Means kümeleme (K=2) kullanarak maske üretimi.
    Fuzzy C-Means'in daha hızlı ve optimize alternatifidir.
    """
    if img_np.dtype != np.uint8:
        img_np = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    # Veriyi OpenCV K-Means'in anlayacağı şekle (tek boyutlu float32 array) getir
    pixel_values = img_np.reshape((-1, 1)).astype(np.float32)
    
    # Kriterler: (Tür, max_iterasyon, epsilon)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Melanom/Lezyonlar her zaman çevre deriden daha koyu renklidir.
    # Merkez (center) değeri daha düşük (karanlık) olan küme indeksini lezyon kabul et.
    dark_cluster_idx = np.argmin(centers)
    
    # Maskeyi oluştur ve orijinal boyutuna getir
    final_mask = (labels.flatten() == dark_cluster_idx).astype(np.uint8)
    final_mask = final_mask.reshape(img_np.shape)
    
    # Cleanup
    final_mask = morph.remove_small_holes(final_mask.astype(bool), area_threshold=32*32).astype(np.uint8)
    final_mask = morph.remove_small_objects(final_mask.astype(bool), min_size=64).astype(np.uint8)
    
    return final_mask

def chc_otsu_pseudo_mask(img_np, q_level=8, n=0.5, min_object_size=64, hole_area=32*32):
    """
    CHC-Otsu pseudo-mask generation.

    Based on the CHC-Otsu idea:
    Color Histogram Clustering + saliency map + Otsu thresholding +
    binary morphological post-processing.

    Args:
        img_np (np.ndarray):
            RGB image with shape (H, W, 3), or grayscale image with shape (H, W).
            Values can be uint8 [0,255] or float [0,1].
        q_level (int):
            Quantization level per channel. q_level=8 gives 8x8x8 = 512 possible clusters.
        n (float):
            Spatial center-prior scaling parameter. Smaller values penalize border regions more.
        min_object_size (int):
            Minimum object size for small-object removal.
        hole_area (int):
            Area threshold for hole filling.

    Returns:
        final_mask (np.ndarray):
            Binary lesion mask, uint8, values {0,1}.
        saliency_map (np.ndarray):
            CHC saliency map in [0,1].
    """

    # --- 1) Prepare RGB image ---
    img = img_np.copy()

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    H, W, _ = img.shape

    # Normalize RGB to [0,1]
    rgb = img.astype(np.float32) / 255.0

    # Convert RGB to Lab and normalize Lab roughly to [0,1]
    lab = color.rgb2lab(rgb)
    lab_norm = np.zeros_like(lab, dtype=np.float32)
    lab_norm[..., 0] = lab[..., 0] / 100.0          # L: [0,100]
    lab_norm[..., 1] = (lab[..., 1] + 128.0) / 255.0  # a: approx [-128,127]
    lab_norm[..., 2] = (lab[..., 2] + 128.0) / 255.0  # b: approx [-128,127]
    lab_norm = np.clip(lab_norm, 0, 1)

    # --- 2) Color quantization: 8x8x8 bins by default ---
    bins = q_level
    rgb_q = np.floor(rgb * bins).astype(np.int32)
    rgb_q = np.clip(rgb_q, 0, bins - 1)

    cluster_id = (
        rgb_q[..., 0] * bins * bins +
        rgb_q[..., 1] * bins +
        rgb_q[..., 2]
    )

    flat_cluster = cluster_id.reshape(-1)
    unique_ids, inverse = np.unique(flat_cluster, return_inverse=True)
    K = len(unique_ids)

    # --- 3) Regional feature extraction ---
    lab_flat = lab_norm.reshape(-1, 3)

    yy, xx = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij"
    )

    x_flat = (xx.reshape(-1) / max(W - 1, 1)).astype(np.float32)
    y_flat = (yy.reshape(-1) / max(H - 1, 1)).astype(np.float32)

    counts = np.bincount(inverse, minlength=K).astype(np.float32)
    weights = counts / counts.sum()

    mean_lab = np.zeros((K, 3), dtype=np.float32)
    mean_x = np.zeros(K, dtype=np.float32)
    mean_y = np.zeros(K, dtype=np.float32)

    for c in range(3):
        mean_lab[:, c] = np.bincount(inverse, weights=lab_flat[:, c], minlength=K) / (counts + 1e-8)

    mean_x = np.bincount(inverse, weights=x_flat, minlength=K) / (counts + 1e-8)
    mean_y = np.bincount(inverse, weights=y_flat, minlength=K) / (counts + 1e-8)

    # --- 4) Color contrast feature ---
    # color_contrast_i = sum_j weight_j * ||Lab_i - Lab_j||
    color_dist = np.linalg.norm(
        mean_lab[:, None, :] - mean_lab[None, :, :],
        axis=2
    )

    color_contrast = color_dist @ weights

    # --- 5) Spatial feature and contrast-ratio weighted saliency ---
    spatial_dist = np.sqrt(
        (mean_x[:, None] - mean_x[None, :]) ** 2 +
        (mean_y[:, None] - mean_y[None, :]) ** 2
    )

    contrast_ratio = (color_contrast[:, None] + 0.05) / (color_contrast[None, :] + 0.05)

    saliency_cluster = np.sum(
        weights[None, :] * contrast_ratio * np.exp(-spatial_dist),
        axis=1
    )

    # Center prior: penalize clusters far from image center
    center_dist = ((mean_x - 0.5) ** 2 + (mean_y - 0.5) ** 2) / (n * n + 1e-8)
    saliency_cluster = np.exp(-center_dist) * (weights * color_contrast + saliency_cluster)

    # Normalize cluster saliency to [0,1]
    saliency_cluster = saliency_cluster - saliency_cluster.min()
    saliency_cluster = saliency_cluster / (saliency_cluster.max() + 1e-8)

    # Assign each pixel its cluster saliency
    saliency_map = saliency_cluster[inverse].reshape(H, W).astype(np.float32)

    # --- 6) Otsu thresholding ---
    thresh = filters.threshold_otsu(saliency_map)
    mask = saliency_map >= thresh

    # Optional: keep the largest connected component tendency
    # because lesions are often dominant central/salient objects.
    mask = morph.remove_small_holes(mask.astype(bool), area_threshold=hole_area)
    mask = morph.remove_small_objects(mask.astype(bool), min_size=min_object_size)

    # Fill holes and smooth slightly
    mask = ndi.binary_fill_holes(mask)
    mask = morph.binary_closing(mask, morph.disk(3))

    final_mask = mask.astype(np.uint8)

    return final_mask, saliency_map

def smart_h1_threshold_ıou(gray, birth, pers, steps=50, window_size=5):
    """
    Belirli bir aralıkta threshold'u değiştirerek maskeler üretir.
    Her adımı, önceki 'window_size' (örn: 5) adet maske ile karşılaştırır.
    Ortalama IoU skorunun en yüksek olduğu (en kararlı) threshold'u döndürür.
    """
    
    # 1. Threshold Aralığını Belirle
    thresholds = np.linspace(birth, birth + pers*0.75, steps)
    
    history = []      # Önceki maskeleri tutacak liste
    iou_scores = []   # Her threshold için hesaplanan stabilite skoru
    
    for i, th in enumerate(thresholds):
        # Maskeyi oluştur (threshold arttıkça maske küçülür veya değişir)
        current_mask = (gray >= th)
        
        # Eğer geçmişte yeterince maske yoksa skoru 0 ver (veya sadece mevcutlarla hesapla)
        if len(history) == 0:
            iou_scores.append(0.0)
        else:
            # Geriye dönük en fazla 5 maskeyi al
            recent_masks = history[-window_size:]
            
            current_ious = []
            for prev_mask in recent_masks:
                # Intersection (Kesişim) ve Union (Birleşim) hesapla
                # Boolean dizileri int'e çevirmeden logical operation yapmak daha hızlıdır
                intersection = np.logical_and(current_mask, prev_mask).sum()
                union = np.logical_or(current_mask, prev_mask).sum()
                
                if union == 0:
                    current_ious.append(0.0)
                else:
                    current_ious.append(intersection / (union + 1e-9))
            
            # Mevcut maskenin, önceki 5 maske ile olan ORTALAMA IoU'su
            avg_iou = np.mean(current_ious)
            iou_scores.append(avg_iou)
        
        # Mevcut maskeyi hafızaya ekle
        history.append(current_mask)
        
        # Hafıza şişmesin diye sadece son 5+1 tanesini tutsak yeterli olurdu ama
        # döngü içinde steps sayısı az (50) olduğu için hepsini tutmak sorun yaratmaz.
        # Yine de optimizasyon için:
        if len(history) > window_size:
             # Sadece sonrakiler için gerekecek olanları tutabiliriz ama 
             # logic gereği history listesi kayan pencere mantığıyla zaten işleniyor.
             pass

    # 2. En Yüksek Skoru Bul
    # İlk birkaç adımda (window_size dolana kadar) IoU düşük çıkabilir, bu normaldir.
    scores_arr = np.array(iou_scores)
    
    # Güvenlik: Eğer tüm skorlar 0 ise (örn: simsiyah resim), varsayılan bir değer dön
    if scores_arr.sum() == 0:
        return birth + 0.15 * pers

    # En yüksek IoU'nun olduğu index
    best_idx = np.argmax(scores_arr)
    
    # 3. İlgili Threshold'u Döndür
    best_th = thresholds[best_idx]
    
    return best_th

def fast_h1_threshold(gray, birth, pers, steps=50):
        
    thresholds = np.linspace(birth, birth + pers, steps)
    areas = np.array([(gray >= th).sum() for th in thresholds])

    return birth + 0.20 * pers

def smart_h1_threshold_twoline(gray, birth, pers, steps=50):
    
    thresholds = np.linspace(birth, birth + pers, steps)
    areas = np.array([(gray >= th).sum() for th in thresholds])

    # 2. Normalizasyon (0-1 Arası)
    x_norm = np.linspace(0, 1, len(areas))
    
    if areas.max() == areas.min():
        return birth + 0.15 * pers
        
    y_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-9)

    # --- YENİ BLOĞ: KESİN İKİ DOĞRU UYDURMA (BRUTE-FORCE) ---
    best_error = float('inf')
    k_optimal_norm = 0.15 # Başlangıç failsafe değeri
    
    # Her iki doğru için en az 3'er nokta bırakarak tüm olası kırılma (k) noktalarını dene
    for i in range(3, len(x_norm) - 3):
        
        # Veriyi i. indeksten ikiye böl
        x1, y1 = x_norm[:i], y_norm[:i]
        x2, y2 = x_norm[i:], y_norm[i:]
        
        # np.polyfit(x, y, 1) -> 1. dereceden (lineer) doğru uydurur ve (eğim, kesişim) döner
        m1, c1 = np.polyfit(x1, y1, 1)
        m2, c2 = np.polyfit(x2, y2, 1)
        
        # Tahmin edilen Y değerleri
        y1_pred = m1 * x1 + c1
        y2_pred = m2 * x2 + c2
        
        # Toplam Karesel Hata (Sum of Squared Errors)
        error1 = np.sum((y1 - y1_pred)**2)
        error2 = np.sum((y2 - y2_pred)**2)
        total_error = error1 + error2
        
        # Eğer bu bölme noktası şu ana kadarki en düşük hatayı verdiyse, kaydet
        if total_error < best_error:
            best_error = total_error
            k_optimal_norm = x_norm[i]
    # --------------------------------------------------------

    # 5. Sonucu Gerçek Değere Çevir
    best_th = birth + (k_optimal_norm * pers)
    
    if best_th < birth + 0.05 * pers:
        best_th = birth + 0.05 * pers

    return best_th

def find_true_corner_origin(areas):
    """
    Türev veya curve_fit kullanmadan, alan eğrisinin sol alt köşeye (orijine) 
    en yakın olduğu gerçek "dirsek" noktasını (idx 10-14 civarı) bulur.
    """
    # 1. Eksenleri 0 ile 1 arasına normalize et (Çok önemli!)
    # Eğer normalize etmezsek 30.000'lik alan değerleri, 50'lik x eksenini ezer.
    x_norm = np.linspace(0, 1, len(areas))
    
    min_a, max_a = np.min(areas), np.max(areas)
    if max_a == min_a:
         return 0 # Alan hiç değişmiyorsa en başı dön
         
    y_norm = (areas - min_a) / (max_a - min_a)
    
    # 2. Sol alt köşeye (0, 0) olan Öklid uzaklığının karesini hesapla
    # Uzaklık formülü: D^2 = X^2 + Y^2
    distances = (x_norm ** 2) + (y_norm ** 2)
    
    # 3. Uzaklığın MİNİMUM olduğu indeks, aradığımız "gürültü bitti, nesne kaldı" noktasıdır
    best_idx = np.argmin(distances)
    
    return best_idx

def smart_h1_threshold_orjin(gray, birth, pers, steps=30):
    """
    Alan eğrisinin orijine (0,0) en yakın olduğu noktayı bularak
    arka plan gürültüsünün bittiği en temiz maske eşiğini (threshold) döndürür.
    """
    # 1. Threshold Aralığını ve Alanları Belirle (Doğum anından itibaren)
    thresholds = np.linspace(birth, birth + 0.75 * pers, steps)
    areas = np.array([(gray >= th).sum() for th in thresholds], dtype=float)

    # Güvenlik (Failsafe): Alan hiç değişmiyorsa veya görsel boşsa
    if areas.max() == areas.min():
        return birth + 0.15 * pers

    # 2. X ve Y Eksenlerini 0 ile 1 Arasına Normalize Et
    # X ekseni: Threshold adımları (0'dan 1'e doğru ilerler)
    x_norm = np.linspace(0, 1, steps)
    
    # Y ekseni: Alan değerleri (1'den 0'a doğru azalır)
    y_norm = (areas - areas.min()) / (areas.max() - areas.min() + 1e-9)

    # 3. Sol Alt Köşeye (Orijine, yani 0,0 noktasına) Olan Uzaklığı Hesapla
    # Geometrik uzaklığın karesi: D^2 = X^2 + Y^2 (Kök almaya gerek yok, minimum nokta değişmez)
    origin_distances = (x_norm ** 2) + (y_norm ** 2)

    # 4. Uzaklığın MİNİMUM olduğu indeks (Eğrinin köşeye en çok yanaştığı o tatlı nokta)
    best_idx = np.argmin(origin_distances)

    # İlgili eşik değerini seç
    best_th = thresholds[best_idx]
    
    return best_th

def smart_h1_threshold(gray, birth, pers, steps=30):

    thresholds = np.linspace(birth, birth + 0.75 * pers, steps)
    areas = np.array([(gray >= th).sum() for th in thresholds])

    dy = np.gradient(areas)  
    ddy = np.gradient(dy)           
    der_idx = np.argmax(ddy)

    x = np.linspace(1, 0, len(areas))                   
    y = (areas - areas.min()) / (areas.max() - areas.min() + 1e-9)

    diff = np.abs(x - y)
    knee_idx = np.argmax(diff)         
    dx = x[-1] - x[0]
    dy = y[-1] - y[0]
    
    slope = dy / dx
    intercept = y[0] - slope * x[0]
    y_line = slope * x + intercept    
    distances = np.abs(y - y_line)
    
    knee_idx = np.argmax(distances)
    knee_strength = distances[knee_idx]
    if knee_strength < 0.25: 

        best_th = birth + 0.15 * pers
        return best_th

    if der_idx + 2 < len(thresholds):
        best_th = thresholds[der_idx + 2]
        #best_th = birth + 0.10 * pers
        return best_th
    else:
        best_th = birth + 0.15 * pers
        return best_th


    # # OTSU METHOD VISUALIZATION
    ##########################################################################################
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(img_np.ravel(), bins=256, color='gray', alpha=0.7)
    plt.axvline(thresh, color='red', linestyle='--', label=f'Otsu Threshold = {thresh:.3f}')
    plt.title('Otsu Threshold on Grayscale Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()

    # Show original and thresholded image
    plt.subplot(1, 2, 2)
    plt.imshow(rough_binary, cmap='gray')
    plt.title('Binary Mask (Lesion < Threshold)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()



    
