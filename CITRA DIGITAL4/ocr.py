import cv2
import numpy as np
from pathlib import Path
import pytesseract

# ==========================================
# KONFIGURASI PATH TESSERACT (Sesuaikan jika di Windows)
# ==========================================
# Contoh path default di Windows:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ==========================================
# PILIH FILE GAMBAR YANG INGIN DI-OCR
# ==========================================
# Contoh menggunakan file 01 atau 02 yang sudah dibahas
file_gambar = "09_CombinedDegradation.jpg"  # Bisa diganti "02_LowContrast.jpg", dll.

if not Path(file_gambar).exists():
  print(f"File {file_gambar} tidak ditemukan!")
else:
  # 1. Baca Citra & Konversi ke Grayscale
  img_bgr = cv2.imread(file_gambar)
  img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

  # 2. Terapkan Metode Enhancement yang Diuji
  beta_val = 40
  img_brightness = cv2.convertScaleAbs(img_gray, alpha=1.0, beta=beta_val)
  img_contrast = cv2.normalize(
      img_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
  )
  img_eq = cv2.equalizeHist(img_gray)

  # Kamus metode untuk di-looping
  metode_dict = {
      "1. Citra Asli (Grayscale)": img_gray,
      "2. Brightness Adjustment": img_brightness,
      "3. Contrast Stretching": img_contrast,
      "4. Histogram Equalization": img_eq,
  }

  print(f"=== HASIL EKSTRAKSI OCR UNTUK: {file_gambar} ===\n")

  # Konfigurasi Tesseract (OEM 3 = Default, PSM 6 = Asumsi blok teks tunggal)
  custom_config = r"--oem 3 --psm 6"

  # 3. Proses OCR untuk Masing-masing Metode
  for nama_metode, citra_proses in metode_dict.items():
    # Jalankan OCR menggunakan Tesseract (Bahasa Indonesia/Inggris 'eng')
    teks_hasil = pytesseract.image_to_string(
        citra_proses, config=custom_config, lang="eng"
    )

    print(f"[{nama_metode}]")
    print(
        f"Teks terbaca : {teks_hasil.strip() if teks_hasil.strip() else '[GAGAL MEMBACA TEKS]'}"
    )
    print("-" * 50)