import cv2
import numpy as np
import matplotlib.pyplot as plt

def buat_citra_sintetis(ukuran=256):
    img = np.full((ukuran, ukuran), 50, dtype=np.uint8)
    noise = np.random.randint(0, 20, (ukuran, ukuran), dtype=np.uint8)
    img = cv2.add(img, noise)

    objek = [
        (80,  80,  45, 200),
        (180, 80,  35, 160),
        (80,  180, 30, 220),
        (190, 175, 40, 180),
        (128, 128, 20, 240),
    ]
    for (cx, cy, r, val) in objek:
        cv2.circle(img, (cx, cy), r, val, -1)
        mask = np.zeros_like(img)
        cv2.circle(mask, (cx, cy), r, 255, -1)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def tampilkan_hasil(judul_besar, gambar_list, judul_list, cmap_list=None, simpan=None):
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
