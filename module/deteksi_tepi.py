import cv2
import numpy as np
from utils.helper import tampilkan_hasil

def demo_deteksi_tepi(img):
    print("\n" + "="*55)
    print("  BAGIAN 3 — DETEKSI TEPI")
    print("="*55)

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_mag = np.uint8(np.clip(sobel_mag, 0, 255))
    _, sobel_bin = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    print("\n[3a] Sobel → Gx + Gy → magnitude → binary threshold")

    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = Kx.T
    gx_manual = cv2.filter2D(img.astype(np.float32), -1, Kx)
    gy_manual = cv2.filter2D(img.astype(np.float32), -1, Ky)
    mag_manual = np.sqrt(gx_manual**2 + gy_manual**2)
    print(f"     Kernel Sobel Kx:\n{Kx}")

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    lap  = cv2.Laplacian(blur, cv2.CV_64F, ksize=3)
    lap_abs = np.uint8(np.clip(np.abs(lap), 0, 255))
    print("[3b] LoG → Gaussian blur (5x5) → Laplacian")

    canny_1 = cv2.Canny(img, threshold1=30,  threshold2=80)
    canny_2 = cv2.Canny(img, threshold1=50,  threshold2=150)
    canny_3 = cv2.Canny(img, threshold1=100, threshold2=200)
    print("[3c] Canny: T_low/T_high = (30/80), (50/150), (100/200)")

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
