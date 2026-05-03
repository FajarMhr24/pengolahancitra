import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.helper import tampilkan_hasil

def demo_kmeans(img):
    print("\n" + "="*55)
    print("  BAGIAN 4 — K-MEANS CLUSTERING")
    print("="*55)

    def segmentasi_kmeans(citra, K):
        data = citra.reshape(-1, 1).astype(np.float32)
        kriteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,   
            0.2    
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
