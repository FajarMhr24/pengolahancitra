"""
============================================================
PENGOLAHAN CITRA DIGITAL — SEGMENTASI CITRA
Contoh Implementasi Lengkap dengan Python & OpenCV
============================================================
Topik yang dibahas:
  1. Thresholding (Global, Otsu, Adaptif)
  2. Region Growing
  3. Deteksi Tepi (Sobel, Canny, Laplacian)
  4. Segmentasi Berbasis Clustering (K-Means)
  5. Watershed Segmentation
  6. Evaluasi Segmentasi (IoU, Dice)
============================================================
Dependensi:
    pip install opencv-python numpy matplotlib scikit-image scipy
============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import deque
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label
from scipy import ndimage

# ─────────────────────────────────────────────────────────
# UTILITAS UMUM
# ─────────────────────────────────────────────────────────

def buat_citra_sintetis(ukuran=256):
    """
    Membuat citra grayscale sintetis berisi beberapa objek bulat
    dengan intensitas berbeda sebagai data uji.
    """
    img = np.full((ukuran, ukuran), 50, dtype=np.uint8)

    # Latar belakang noise ringan
    noise = np.random.randint(0, 20, (ukuran, ukuran), dtype=np.uint8)
    img = cv2.add(img, noise)

    # Objek lingkaran: (pusat_x, pusat_y, jari-jari, intensitas)
    objek = [
        (80,  80,  45, 200),
        (180, 80,  35, 160),
        (80,  180, 30, 220),
        (190, 175, 40, 180),
        (128, 128, 20, 240),
    ]
    for (cx, cy, r, val) in objek:
        cv2.circle(img, (cx, cy), r, val, -1)
        # Sedikit blur per objek agar lebih realistis
        mask = np.zeros_like(img)
        cv2.circle(mask, (cx, cy), r, 255, -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def tampilkan_hasil(judul_besar, gambar_list, judul_list, cmap_list=None, simpan=None):
    """
    Helper untuk menampilkan beberapa citra dalam satu figure.
    gambar_list : list of numpy arrays
    judul_list  : list of str
    """
    n = len(gambar_list)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.suptitle(judul_besar, fontsize=14, fontweight='bold', y=1.01)

    axes_flat = np.array(axes).flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes_flat):
        if i < n:
            cmap = (cmap_list[i] if cmap_list else None) or 'gray'
            ax.imshow(gambar_list[i], cmap=cmap)
            ax.set_title(judul_list[i], fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    if simpan:
        plt.savefig(simpan, dpi=120, bbox_inches='tight')
        print(f"  [Gambar disimpan: {simpan}]")
    plt.show()


# ─────────────────────────────────────────────────────────
# 1. THRESHOLDING
# ─────────────────────────────────────────────────────────

def demo_thresholding(img):
    """
    Mendemonstrasikan tiga jenis thresholding:
    a) Global (manual)
    b) Metode Otsu (otomatis)
    c) Thresholding Adaptif (Gaussian)
    """
    print("\n" + "="*55)
    print("  BAGIAN 1 — THRESHOLDING")
    print("="*55)

    # ── (a) Thresholding Global Manual ──
    T_manual = 128
    _, thresh_global = cv2.threshold(img, T_manual, 255, cv2.THRESH_BINARY)
    print(f"\n[1a] Thresholding Global  → T = {T_manual}")

    # ── (b) Metode Otsu ──
    T_otsu, thresh_otsu = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    print(f"[1b] Metode Otsu          → T optimal = {T_otsu:.1f} (dihitung otomatis)")

    # ── Otsu manual untuk memahami mekanisme ──
    T_hitung, sigma2_max = _hitung_otsu_manual(img)
    print(f"     Verifikasi manual     → T = {T_hitung}, sigma²_B_maks = {sigma2_max:.2f}")

    # ── (c) Thresholding Adaptif ──
    thresh_adaptif = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,   # ukuran neighbourhood (harus ganjil)
        C=5             # konstanta pengurang
    )
    print("[1c] Thresholding Adaptif → blockSize=31, C=5")

    # ── Histogram + garis threshold ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.suptitle("Histogram & Threshold", fontweight='bold')
    axes[0].hist(img.ravel(), bins=256, range=(0, 255), color='steelblue', alpha=0.7)
    axes[0].axvline(T_manual, color='orange', linewidth=1.5, label=f'Global T={T_manual}')
    axes[0].axvline(T_otsu,   color='red',    linewidth=1.5, label=f'Otsu T={T_otsu:.0f}')
    axes[0].set_title("Histogram Intensitas"); axes[0].legend(fontsize=8)
    axes[1].imshow(img, cmap='gray')
    axes[1].set_title("Citra Asli"); axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/mydocument/praktik_p_citra/out_1_histogram.png', dpi=110, bbox_inches='tight')
    plt.show()

    tampilkan_hasil(
        "Thresholding — Perbandingan Metode",
        [img, thresh_global, thresh_otsu, thresh_adaptif],
        ["Citra Asli (Grayscale)",
         f"Global  T={T_manual}",
         f"Otsu    T={T_otsu:.0f} (auto)",
         "Adaptif (Gaussian)"],
        simpan='/mydocument/praktik_p_citra/out_1_thresholding.png'
    )
    return thresh_otsu


def _hitung_otsu_manual(img):
    """
    Implementasi manual algoritma Otsu untuk keperluan pembelajaran.
    Memaksimalkan variansi antar kelas (between-class variance).

    Formula:
        sigma²_B(T) = omega_0 * omega_1 * (mu_0 - mu_1)²
    """
    hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
    total    = img.size

    sigma2_max = 0
    T_optimal  = 0

    sum_total = np.dot(np.arange(256), hist)
    sum_bg    = 0.0
    w_bg      = 0

    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break

        sum_bg += t * hist[t]
        mu_bg   = sum_bg / w_bg
        mu_fg   = (sum_total - sum_bg) / w_fg

        sigma2 = (w_bg / total) * (w_fg / total) * (mu_bg - mu_fg) ** 2

        if sigma2 > sigma2_max:
            sigma2_max = sigma2
            T_optimal  = t

    return T_optimal, sigma2_max


# ─────────────────────────────────────────────────────────
# 2. REGION GROWING
# ─────────────────────────────────────────────────────────

def demo_region_growing(img):
    """
    Implementasi Region Growing dari scratch menggunakan BFS.

    Algoritma:
      1. Tentukan seed point
      2. Tambahkan tetangga jika |I(tetangga) - I(seed)| <= threshold
      3. Ulangi sampai antrian kosong (BFS)
    """
    print("\n" + "="*55)
    print("  BAGIAN 2 — REGION GROWING")
    print("="*55)

    def region_growing_bfs(citra, seed, threshold=25, konektivitas=8):
        """
        BFS-based region growing.
        Mengembalikan mask boolean area yang tersegmentasi.
        """
        h, w    = citra.shape
        visited = np.zeros((h, w), dtype=bool)
        mask    = np.zeros((h, w), dtype=bool)
        antrian = deque([seed])
        visited[seed[0], seed[1]] = True

        # Definisi arah: 4-konektivitas atau 8-konektivitas
        if konektivitas == 4:
            arah = [(-1,0),(1,0),(0,-1),(0,1)]
        else:
            arah = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

        nilai_seed = int(citra[seed[0], seed[1]])

        while antrian:
            y, x = antrian.popleft()
            mask[y, x] = True

            for dy, dx in arah:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if abs(int(citra[ny, nx]) - nilai_seed) <= threshold:
                        antrian.append((ny, nx))

        return mask

    # Uji dengan berbagai seed dan threshold
    seeds = [(80, 80), (180, 80), (128, 128)]
    thresholds = [15, 30, 50]

    print(f"\n  Seed points diuji: {seeds}")
    print(f"  Threshold diuji  : {thresholds}\n")

    hasil = []
    label_list = []

    for seed in seeds:
        for T in thresholds:
            mask = region_growing_bfs(img, seed, threshold=T)
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            overlay[mask] = [0, 200, 100]          # warna hijau = region
            cy, cx = seed
            cv2.drawMarker(overlay, (cx, cy), (255, 50, 50), cv2.MARKER_STAR, 12, 2)
            hasil.append(overlay)
            label_list.append(f"Seed({cx},{cy}) T={T}")

    tampilkan_hasil(
        "Region Growing — Pengaruh Seed Point & Threshold",
        hasil[:8],
        label_list[:8],
        cmap_list=[None]*8,
        simpan='/mydocument/praktik_p_citra/out_2_region_growing.png'
    )

    # Contoh multi-region growing
    print("  Multi-region growing: 3 seed → 3 region berbeda warna")
    multi_mask = np.zeros((*img.shape, 3), dtype=np.uint8)
    warna = [(220, 80, 80), (80, 180, 80), (80, 120, 220)]
    for i, (seed, w_rgb) in enumerate(zip(seeds, warna)):
        mask = region_growing_bfs(img, seed, threshold=30)
        multi_mask[mask] = w_rgb

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle("Region Growing — Multi-Region", fontweight='bold')
    ax[0].imshow(img, cmap='gray'); ax[0].set_title("Citra Asli"); ax[0].axis('off')
    ax[1].imshow(multi_mask); ax[1].set_title("3 Region (seed berbeda)"); ax[1].axis('off')
    for i, (seed, c) in enumerate(zip(seeds, warna)):
        ax[1].plot(seed[1], seed[0], '*', color=[v/255 for v in c], markersize=12)
        
    plt.tight_layout()
    plt.savefig('/mydocument/praktik_p_citra/out_2b_multiregion.png', dpi=110, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────
# 3. DETEKSI TEPI (EDGE DETECTION)
# ─────────────────────────────────────────────────────────

def demo_deteksi_tepi(img):
    """
    Membandingkan operator-operator deteksi tepi:
    - Sobel (Gx, Gy, magnitude)
    - Laplacian of Gaussian (LoG)
    - Canny (dengan double threshold)
    """
    print("\n" + "="*55)
    print("  BAGIAN 3 — DETEKSI TEPI")
    print("="*55)

    # ── Sobel ──
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))
    _, sobel_bin = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    print("\n[3a] Sobel → Gx + Gy → magnitude → binary threshold")

    # Implementasi manual kernel Sobel untuk pembelajaran
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = Kx.T
    gx_manual = cv2.filter2D(img.astype(np.float32), -1, Kx)
    gy_manual = cv2.filter2D(img.astype(np.float32), -1, Ky)
    mag_manual = np.sqrt(gx_manual**2 + gy_manual**2)
    print(f"     Kernel Sobel Kx:\n{Kx}")

    # ── Laplacian of Gaussian (LoG) ──
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    lap  = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
    lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))
    print("[3b] LoG → Gaussian blur (5x5) → Laplacian")

    # ── Algoritma Canny ──
    # Uji berbagai pasangan (T_low, T_high)
    canny_1 = cv2.Canny(img, threshold1=30,  threshold2=80)
    canny_2 = cv2.Canny(img, threshold1=50,  threshold2=150)
    canny_3 = cv2.Canny(img, threshold1=100, threshold2=200)
    print("[3c] Canny: T_low/T_high = (30/80), (50/150), (100/200)")
    print("Langkah: Gaussian → Sobel → Non-max suppression → Double threshold → Hysteresis")

    tampilkan_hasil(
        "Deteksi Tepi — Perbandingan Operator",
        [img,
         np.uint8(np.clip(gx_manual, 0, 255)),
         np.uint8(np.clip(gy_manual, 0, 255)),
         sobel_bin,
         lap_abs,
         canny_1, canny_2, canny_3],
        ["Citra Asli",
         "Sobel Gx (horizontal)",
         "Sobel Gy (vertikal)",
         "Sobel Magnitude (bin)",
         "Laplacian of Gaussian",
         "Canny (30/80)",
         "Canny (50/150)",
         "Canny (100/200)"],
        simpan='/mydocument/praktik_p_citra/out_3_deteksi_tepi.png'
    )
    return canny_2


# ─────────────────────────────────────────────────────────
# 4. CLUSTERING — K-MEANS
# ─────────────────────────────────────────────────────────

def demo_kmeans(img):
    """
    Segmentasi menggunakan K-Means clustering.
    Setiap piksel dianggap sebagai titik data 1D (intensitas).
    Algoritma mengelompokkan piksel ke dalam K cluster.
    """
    print("\n" + "="*55)
    print("  BAGIAN 4 — K-MEANS CLUSTERING")
    print("="*55)

    def segmentasi_kmeans(citra, K):
        """
        Menerapkan K-Means pada citra grayscale.
        Mengembalikan citra tersegmentasi dan label tiap piksel.
        """
        # Reshape menjadi vektor kolom
        data = citra.reshape(-1, 1).astype(np.float32)

        kriteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,   # iterasi maksimum
            0.2    # epsilon
        )
        _, labels, centroid = cv2.kmeans(
            data, K, None, kriteria,
            attempts=10,
            flags=cv2.KMEANS_RANDOM_CENTERS
        )

        centroid = np.uint8(centroid)
        hasil    = centroid[labels.flatten()].reshape(citra.shape)
        return hasil, labels.reshape(citra.shape), centroid.flatten()

    nilai_K = [2, 3, 4, 5]
    gambar_hasil = [img]
    label_hasil  = ["Citra Asli"]

    for K in nilai_K:
        seg, label_map, centroid = segmentasi_kmeans(img, K)
        centroid_str = ', '.join(str(c) for c in sorted(centroid))
        print(f"\n  K={K}: centroid intensitas = [{centroid_str}]")
        gambar_hasil.append(seg)
        label_hasil.append(f"K-Means K={K}")

    tampilkan_hasil(
        "Segmentasi K-Means — Pengaruh Jumlah Cluster (K)",
        gambar_hasil,
        label_hasil,
        simpan='/mydocument/praktik_p_citra/out_4_kmeans.png'
    )

    # Visualisasi label berwarna untuk K=3
    seg3, labels3, centroid3 = segmentasi_kmeans(img, 3)
    color_map = np.array([[220, 80, 80], [80, 200, 120], [80, 130, 220]], dtype=np.uint8)
    colored   = color_map[labels3]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle("K-Means K=3 — Analisis Detail", fontweight='bold')
    axes[0].imshow(img, cmap='gray'); axes[0].set_title("Citra Asli"); axes[0].axis('off')
    axes[1].imshow(seg3, cmap='gray'); axes[1].set_title("Hasil (grayscale)"); axes[1].axis('off')
    axes[2].imshow(colored); axes[2].set_title("Label per cluster (berwarna)"); axes[2].axis('off')
    
    legenda = [mpatches.Patch(color=[v/255 for v in color_map[i]], label=f"Cluster {i+1} (I≈{centroid3[i]})") for i in range(3)]
    axes[2].legend(handles=legenda, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/mydocument/praktik_p_citra/out_4b_kmeans_color.png', dpi=110, bbox_inches='tight')
    plt.show()


# ─────────────────────────────────────────────────────────
# 5. WATERSHED SEGMENTATION
# ─────────────────────────────────────────────────────────

def demo_watershed(img):
    """
    Segmentasi Watershed berbasis marker.

    Langkah:
      1. Thresholding → biner
      2. Morphological operations → tentukan foreground pasti
      3. Distance transform → temukan pusat objek
      4. Peak local max → marker seed
      5. Watershed → batas region
    """
    print("\n" + "="*55)
    print("  BAGIAN 5 — WATERSHED SEGMENTATION")
    print("="*55)

    # Step 1: Otsu thresholding
    _, biner = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2: Morphological opening untuk menghilangkan noise
    kernel     = np.ones((3, 3), np.uint8)
    opening    = cv2.morphologyEx(biner, cv2.MORPH_OPEN, kernel, iterations=2)

    # Tentukan background pasti
    sure_bg    = cv2.dilate(opening, kernel, iterations=3)

    # Step 3: Distance transform
    dist_tf    = ndimage.distance_transform_edt(opening)

    # Step 4: Peak local max → markers
    coords     = peak_local_max(dist_tf, min_distance=15, labels=opening)
    mask_peak  = np.zeros(dist_tf.shape, dtype=bool)
    mask_peak[tuple(coords.T)] = True
    markers    = label(mask_peak)
    n_markers  = markers.max()
    print(f"\n  Objek terdeteksi (markers): {n_markers}")

    # Step 5: Watershed
    ws_labels  = watershed(-dist_tf, markers, mask=opening)

    # Buat visualisasi overlay
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i in range(1, ws_labels.max() + 1):
        warna = tuple(int(c) for c in np.random.randint(80, 230, 3))
        overlay[ws_labels == i] = warna

    # Tandai batas watershed
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


# ─────────────────────────────────────────────────────────
# 6. EVALUASI SEGMENTASI
# ─────────────────────────────────────────────────────────

def demo_evaluasi(img):
    """
    Menghitung metrik evaluasi standar segmentasi:
    - Pixel Accuracy
    - IoU (Intersection over Union / Jaccard Index)
    - Dice Coefficient
    - Precision & Recall
    """
    print("\n" + "="*55)
    print("  BAGIAN 6 — EVALUASI SEGMENTASI")
    print("="*55)

    def pixel_accuracy(pred, gt):
        return np.sum(pred == gt) / gt.size

    def iou(pred, gt):
        interseksi = np.logical_and(pred, gt).sum()
        gabungan   = np.logical_or(pred, gt).sum()
        return interseksi / gabungan if gabungan > 0 else 0.0

    def dice(pred, gt):
        interseksi = np.logical_and(pred, gt).sum()
        return 2 * interseksi / (pred.sum() + gt.sum()) if (pred.sum() + gt.sum()) > 0 else 0.0

    def precision_recall(pred, gt):
        TP = np.logical_and(pred, gt).sum()
        FP = np.logical_and(pred, ~gt).sum()
        FN = np.logical_and(~pred, gt).sum()
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        return prec, rec

    # Buat ground truth sintetis (lingkaran di tengah)
    gt = np.zeros_like(img, dtype=bool)
    cv2.circle(gt.view(np.uint8), (128, 128), 50, 255, -1)
    gt = gt.astype(bool)

    # Buat beberapa prediksi dengan tingkat kecocokan berbeda
    prediksi_list  = []
    label_pred     = []

    # Prediksi 1: sangat baik (offset kecil)
    p1 = np.zeros_like(img, dtype=bool)
    cv2.circle(p1.view(np.uint8), (130, 130), 48, 255, -1)
    prediksi_list.append(p1.astype(bool))
    label_pred.append("Prediksi A\n(hampir sempurna)")

    # Prediksi 2: lebih kecil
    p2 = np.zeros_like(img, dtype=bool)
    cv2.circle(p2.view(np.uint8), (128, 128), 35, 255, -1)
    prediksi_list.append(p2.astype(bool))
    label_pred.append("Prediksi B\n(under-segmentation)")

    # Prediksi 3: lebih besar
    p3 = np.zeros_like(img, dtype=bool)
    cv2.circle(p3.view(np.uint8), (128, 128), 65, 255, -1)
    prediksi_list.append(p3.astype(bool))
    label_pred.append("Prediksi C\n(over-segmentation)")

    # Prediksi 4: sangat meleset
    p4 = np.zeros_like(img, dtype=bool)
    cv2.circle(p4.view(np.uint8), (80, 80), 30, 255, -1)
    prediksi_list.append(p4.astype(bool))
    label_pred.append("Prediksi D\n(posisi salah)")

    print(f"\n  {'Prediksi':<28} {'PA':>6} {'IoU':>6} {'Dice':>6} {'Prec':>6} {'Recall':>6}")
    print(f"  {'-'*28} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")

    for pred, lbl in zip(prediksi_list, label_pred):
        pa_val          = pixel_accuracy(pred, gt)
        iou_val         = iou(pred, gt)
        dice_val        = dice(pred, gt)
        prec_val, rec_val = precision_recall(pred, gt)
        nama = lbl.replace('\n', ' ')
        print(f"  {nama:<28} {pa_val:>6.3f} {iou_val:>6.3f} {dice_val:>6.3f} {prec_val:>6.3f} {rec_val:>6.3f}")

    print(f"\n  Catatan: IoU dan Dice berkorelasi → Dice = 2*IoU / (1 + IoU)")

    # Visualisasi: ground truth vs prediksi
    gambar_vis = [img]
    judul_vis  = ["Citra Asli"]

    def buat_overlay_evaluasi(citra, pred, gt):
        """Warna: hijau=TP, merah=FP, biru=FN"""
        vis = cv2.cvtColor(citra, cv2.COLOR_GRAY2RGB).copy()
        vis[np.logical_and(pred, gt)]  = [80,  200, 80]   # TP
        vis[np.logical_and(pred, ~gt)] = [200, 80,  80]   # FP
        vis[np.logical_and(~pred, gt)] = [80,  80,  200]  # FN
        return vis

    for pred, lbl in zip(prediksi_list, label_pred):
        iou_v  = iou(pred, gt)
        dice_v = dice(pred, gt)
        overlay = buat_overlay_evaluasi(img, pred, gt)
        gambar_vis.append(overlay)
        judul_vis.append(f"{lbl}\nIoU={iou_v:.3f} | Dice={dice_v:.3f}")

    tampilkan_hasil(
        "Evaluasi Segmentasi (Hijau=TP, Merah=FP, Biru=FN)",
        gambar_vis,
        judul_vis,
        cmap_list=['gray'] + [None]*4,
        simpan='/mydocument/praktik_p_citra/out_6_evaluasi.png'
    )


# ─────────────────────────────────────────────────────────
# PROGRAM UTAMA
# ─────────────────────────────────────────────────────────

def main():
    print("\n" + "="*55)
    print("  PENGOLAHAN CITRA — SEGMENTASI CITRA")
    print("  Implementasi Python + OpenCV")
    print("="*55)

    np.random.seed(42)
    img = buat_citra_sintetis(ukuran=256)
    print(f"\n  Citra sintetis dibuat: {img.shape}, dtype={img.dtype}")
    print(f"  Intensitas: min={img.min()}, max={img.max()}, mean={img.mean():.1f}")

    # Jalankan semua demo secara berurutan
    demo_thresholding(img)
    demo_region_growing(img)
    demo_deteksi_tepi(img)
    demo_kmeans(img)
    demo_watershed(img)
    demo_evaluasi(img)

    print("\n" + "="*55)
    print("  SELESAI — Semua demo berhasil dijalankan.")
    print("  File output tersimpan di direktori saat ini.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
