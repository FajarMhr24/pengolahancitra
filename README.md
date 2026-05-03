# UTS PENGOLAHAN CITRA

### Nama  : Elisabeth Erni Marbun Banjarnahor
### Kelas  : I241E
### Nim    : 312410525

# Penjelasan Function

## 1. Fungsi Visualisasi (Plot Gambar)

(Kemungkinan nama function: `tampilkan_gambar` / sejenisnya)

### Tujuan:

Fungsi ini digunakan untuk menampilkan beberapa gambar sekaligus dalam bentuk subplot menggunakan matplotlib.

### Cara Kerja:

* Menerima:
  * `gambar_list` → list gambar
  * `judul_list` → judul tiap gambar
  * `cmap_list` → jenis warna (default: grayscale)
* Menggunakan `plt.subplots()` untuk membuat grid
* Loop setiap gambar:

  ```python
  for i, ax in enumerate(axes_flat):
  ```
* Menampilkan gambar:

  ```python
  ax.imshow(gambar_list[i], cmap=cmap)
  ```

* Menghilangkan axis:

  ```python
  ax.axis('off')
  ```
  ### Penyimpanan:

  ```python
  plt.savefig(simpan, dpi=120, bbox_inches='tight')
  ```

  Artinya:

  * Gambar bisa otomatis disimpan ke file output

  ## 2. demo_thresholding(img)

  Tujuan:

Mendemonstrasikan berbagai metode thresholding untuk segmentasi citra.

Biasanya berisi:

* Global threshold:
  ```python
  cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
  ``` 

* otsu:
  ```python
  cv2.THRESH_OTSU
  ```

* Adaptive:
  ```python
  cv2.adaptiveThreshold()
  ```

  output :

  ![foto](https://github.com/Elisabethbanjarnahor/Uts-pengolahan-citra/blob/96780e0cda070536684d6da1a1c6c30ea0c4c13e/foto/out_1_thresholding.png)

## 3. demo_region_growing(img)

Tujuan:

Melakukan segmentasi berbasis pertumbuhan region dari seed.

Cara kerja:

* Tentukan seed point
* Gunakan BFS:
  * Queue untuk menyimpan pixel
* Cek tetangga:
  * 4 arah / 8 arah
* Bandingkan intensitas:
  ```python
  abs(pixel - seed) < threshold
  ```
output :

![foto](https://github.com/Elisabethbanjarnahor/Uts-pengolahan-citra/blob/96780e0cda070536684d6da1a1c6c30ea0c4c13e/foto/out_2_region_growing.png)

## 4. demo_deteksi_tepi(img)

Tujuan :

Mendeteksi tepi/batas objek dalam citra.

Biasanya:

* Sobel:
  ```python
  cv2.Sobel()
  ```

* Canny:
  ```python
  cv2.Canny()
  ```

output: 

![foto](https://github.com/Elisabethbanjarnahor/Uts-pengolahan-citra/blob/96780e0cda070536684d6da1a1c6c30ea0c4c13e/foto/out_3_deteksi_tepi.png)

## 5. demo_kmeans(img)

Tujuan :
Segmentasi menggunakan clustering (unsupervised learning).

Proses : 
  1. Segmentasi menggunakan clustering (unsupervised learning).
  2. Terapkan:
     ```python
     cv2.kmeans()
     ```
   3. Assign label cluster ke pixel
   4. Assign label cluster ke pixel

output :

![foto](https://github.com/Elisabethbanjarnahor/Uts-pengolahan-citra/blob/96780e0cda070536684d6da1a1c6c30ea0c4c13e/foto/out_4_kmeans.png)

output kmeans colour:
![foto](https://github.com/Elisabethbanjarnahor/Uts-pengolahan-citra/blob/96780e0cda070536684d6da1a1c6c30ea0c4c13e/foto/out_4b_kmeans_color.png)

## 6. demo_watershed(img)

Tujuan :
Segmentasi objek yang saling menempel

Tahapan : 
  1. Thresholding
  2. Morphology:
     ```python
     cv2.morphologyEx()
     ```
  3. Distance transform:
     ```python
     cv2.distanceTransform()
     ```
  4. Marker detection
  5. Watershed:
     ```python
     cv2.watershed()
     ```

![foto](https://github.com/Elisabethbanjarnahor/Uts-pengolahan-citra/blob/96780e0cda070536684d6da1a1c6c30ea0c4c13e/foto/out_5_watershed.png)

## 7. demo_evaluasi(img)

Tujuan:

Mengukur kualitas hasil segmentasi

Metrik:
  * Pixel Accuracy
  * IoU
  * Dice
  * Precision & Recall

Biasanya : 
bandingkan:
```pyhon
prediksi vs ground_truth
```

![foto](https://github.com/Elisabethbanjarnahor/Uts-pengolahan-citra/blob/96780e0cda070536684d6da1a1c6c30ea0c4c13e/foto/out_6_evaluasi.png)

## 8. main()

Fungsi Utama Program

Isi:

```pyhon
demo_thresholding(img)
demo_region_growing(img)
demo_deteksi_tepi(img)
demo_kmeans(img)
demo_watershed(img)
demo_evaluasi(img)
```

Artinya :
Semua algoritma dijalankan secara berurutan

Output Terminal:
```python
print("SELESAI — Semua demo berhasil dijalankan.")
```

## 9. Entry Point Python

```python
if __name__ == "__main__":
    main()
```
Artinya:
Program hanya berjalan jika file dijalankan langsung (bukan di-import)
