import cv2
import numpy as np
from utils.helper import tampilkan_hasil

def demo_evaluasi(img):
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

    gt = np.zeros_like(img, dtype=bool)
    cv2.circle(gt.view(np.uint8), (128, 128), 50, 255, -1)
    gt = gt.astype(bool)

    prediksi_list  = []
    label_pred     = []

    p1 = np.zeros_like(img, dtype=bool)
    cv2.circle(p1.view(np.uint8), (130, 130), 48, 255, -1)
    prediksi_list.append(p1.astype(bool))
    label_pred.append("Prediksi A\n(hampir sempurna)")

    p2 = np.zeros_like(img, dtype=bool)
    cv2.circle(p2.view(np.uint8), (128, 128), 35, 255, -1)
    prediksi_list.append(p2.astype(bool))
    label_pred.append("Prediksi B\n(under-segmentation)")

    p3 = np.zeros_like(img, dtype=bool)
    cv2.circle(p3.view(np.uint8), (128, 128), 65, 255, -1)
    prediksi_list.append(p3.astype(bool))
    label_pred.append("Prediksi C\n(over-segmentation)")

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

    gambar_vis = [img]
    judul_vis  = ["Citra Asli"]

    def buat_overlay_evaluasi(citra, pred, gt):
        vis = cv2.cvtColor(citra, cv2.COLOR_GRAY2RGB).copy()
        vis[np.logical_and(pred, gt)]  = [80,  200, 80]   
        vis[np.logical_and(pred, ~gt)] = [200, 80,  80]   
        vis[np.logical_and(~pred, gt)] = [80,  80,  200]  
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
