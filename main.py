import numpy as np

# Import dari folder utils dan modules
from utils.helper import buat_citra_sintetis
from modules.thresholding import demo_thresholding
from modules.region_growing import demo_region_growing
from modules.deteksi_tepi import demo_deteksi_tepi
from modules.clustering import demo_kmeans
from modules.watershed import demo_watershed
from modules.evaluasi import demo_evaluasi

def main():
    print("\n" + "="*55)
    print("  PENGOLAHAN CITRA — SEGMENTASI CITRA")
    print("  Implementasi Python + OpenCV")
    print("="*55)

    np.random.seed(42)
    img = buat_citra_sintetis(ukuran=256)
    
    demo_thresholding(img)
    demo_region_growing(img)
    demo_deteksi_tepi(img)
    demo_kmeans(img)
    demo_watershed(img)
    demo_evaluasi(img)

    print("\n" + "="*55)
    print("  SELESAI — Semua demo berhasil dijalankan.")
    print("="*55 + "\n")

if __name__ == "__main__":
    main()
