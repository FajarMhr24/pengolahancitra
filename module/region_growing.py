import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from utils.helper import tampilkan_hasil

def demo_region_growing(img):
    print("\n" + "="*55)
    print("  BAGIAN 2 — REGION GROWING")
    print("="*55)

    def region_growing_bfs(citra, seed, threshold=25, konektivitas=8):
        h, w    = citra.shape
        visited = np.zeros((h, w), dtype=bool)
        mask    = np.zeros((h, w), dtype=bool)
        antrian = deque([seed])
        visited[seed[0], seed[1]] = True

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
            overlay[mask] = [0, 200, 100]          
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
