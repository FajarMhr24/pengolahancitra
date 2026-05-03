import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
from utils.helper import tampilkan_hasil

def demo_watershed(img):
    print("\n" + "="*55)
    print("  BAGIAN 5 — WATERSHED SEGMENTATION")
    print("="*55)

    _, biner = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel     = np.ones((3, 3), np.uint8)
    opening    = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg    = cv2.dilate(opening, kernel, iterations=3)

    dist_tf    = ndimage.distance_transform_edt(opening)

    coords     = peak_local_max(dist_tf, min_distance=15, labels=opening)
    mask_peak  = np.zeros(dist_tf.shape, dtype=bool)
    mask_peak[tuple(coords.T)] = True
    markers    = label(mask_peak)
    n_markers  = markers.max()
    print(f"\n  Objek terdeteksi (markers): {n_markers}")

    ws_labels  = watershed(-dist_tf, markers, mask=opening)

    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(1, ws_labels.max() + 1):
        warna = tuple(int(c) for c in np.random.randint(80, 230, 3))
        overlay[ws_labels == i] = warna

    batas = np.zeros_like(img)
    for i in range(1, ws_labels.max() + 1):
        kontur, _ = cv2.findContours((ws_labels == i).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, kontur, -1, (255, 255, 255), 1)

    print(f"  Jumlah region hasil watershed: {ws_labels.max()}")

    tampilkan_hasil(
        "Watershed Segmentation — Tahap demi Tahap",
        [img, biner, opening, dist_tf, mask_peak.astype(np.uint8)*255, overlay],
        ["1. Citra Asli",
         "2. Otsu Threshold",
         "3. Morphological Opening",
         "4. Distance Transform",
         "5. Peak Markers",
         "6. Hasil Watershed"],
        cmap_list=['gray','gray','gray','hot','gray', None],
        simpan='/mydocument/praktik_p_citra/out_5_watershed.png'
    )
