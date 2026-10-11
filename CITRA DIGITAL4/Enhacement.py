import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ==========================================
# 1. KONFIGURASI FOLDER OUTPUT PERBANDINGAN
# ==========================================
folder_perbandingan = Path("hasil/perbandingan")
folder_perbandingan.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. DAFTAR LENGKAP SEMUA FILE GAMBAR (01 S.D 09)
# ==========================================
daftar_gambar = [
    "01_HighQuality_Enhanced.jpg",
    "02_LowContrast.jpg",
    "03_Blurred.jpg",
    "04_HighNoise.jpg",
    "05_LowResolution_Upsampled.jpg",
    "06_Faded_Underexposed.jpg",
    "07_ColorShift_WarmTint.jpg",
    "08_JPEGCompression_Artifacts.jpg",
    "09_CombinedDegradation.jpg",
]

beta_val = 40

# ==========================================
# 3. PROSES LOOPING OTOMATIS
# ==========================================
for img_path in daftar_gambar:
  if not Path(img_path).exists():
    print(f"File {img_path} tidak ditemukan di direktori, dilewati.")
    continue

  # Baca citra
  img_bgr = cv2.imread(img_path)
  nama_file = Path(img_path).stem
  img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  # Implementasi Metode Enhancement
  img_brightness = cv2.convertScaleAbs(img_gray, alpha=1.0, beta=beta_val)
  img_contrast = cv2.normalize(
      img_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
  )
  img_eq = cv2.equalizeHist(img_gray)

  # Buat Figure dengan Grid 4 Baris x 2 Kolom
  fig, axes = plt.subplots(4, 2, figsize=(12, 16))

  # Baris 1: Citra Asli & Histogramnya
  axes[0, 0].imshow(img_gray, cmap="gray")
  axes[0, 0].set_title("Citra Asli (Grayscale)")
  axes[0, 0].axis("off")

  axes[0, 1].hist(img_gray.ravel(), 256, [0, 256], color="black")
  axes[0, 1].set_title("Histogram Asli")
  axes[0, 1].set_xlim([0, 256])

  # Baris 2: Brightness Adjustment & Histogramnya
  axes[1, 0].imshow(img_brightness, cmap="gray")
  axes[1, 0].set_title(f"Brightness Adjustment (beta={beta_val})")
  axes[1, 0].axis("off")

  axes[1, 1].hist(img_brightness.ravel(), 256, [0, 256], color="black")
  axes[1, 1].set_title("Histogram Brightness Adjustment")
  axes[1, 1].set_xlim([0, 256])

  # Baris 3: Contrast Stretching & Histogramnya
  axes[2, 0].imshow(img_contrast, cmap="gray")
  axes[2, 0].set_title("Contrast Stretching")
  axes[2, 0].axis("off")

  axes[2, 1].hist(img_contrast.ravel(), 256, [0, 256], color="black")
  axes[2, 1].set_title("Histogram Contrast Stretching")
  axes[2, 1].set_xlim([0, 256])

  # Baris 4: Histogram Equalization & Histogramnya
  axes[3, 0].imshow(img_eq, cmap="gray")
  axes[3, 0].set_title("Histogram Equalization")
  axes[3, 0].axis("off")

  axes[3, 1].hist(img_eq.ravel(), 256, [0, 256], color="black")
  axes[3, 1].set_title("Histogram Equalization")
  axes[3, 1].set_xlim([0, 256])

  # Pengaturan Layout
  plt.suptitle(f"Perbandingan Enhancement - {nama_file}", fontsize=16, y=0.98)
  plt.tight_layout(rect=[0, 0, 1, 0.97])

  # Simpan hasil gabungan
  path_output = folder_perbandingan / f"{nama_file}_perbandingan.png"
  plt.savefig(str(path_output), dpi=150)
  plt.close()

  print(f"Berhasil membuat: {nama_file}_perbandingan.png")

print(
    "\nSelesai! Semua file perbandingan dari 01 sampai 09 berhasil dibuat di"
    " folder 'hasil/perbandingan'."
)