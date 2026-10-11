import cv2
import numpy as np
from pathlib import Path


# ==============================
# FOLDER
# ==============================

INPUT_FOLDER = Path(".")
OUTPUT_FOLDER = Path("hasil")

GRAY_FOLDER = OUTPUT_FOLDER / "grayscale"
MEAN_FOLDER = OUTPUT_FOLDER / "mean"
MEDIAN_FOLDER = OUTPUT_FOLDER / "median"
GAUSSIAN_FOLDER = OUTPUT_FOLDER / "gaussian"
SHARPENING_FOLDER = OUTPUT_FOLDER / "sharpening"
COMPARISON_FOLDER = OUTPUT_FOLDER / "perbandingan"


# ==============================
# MEMBUAT FOLDER JIKA BELUM ADA
# ==============================
for folder in [GRAY_FOLDER, MEAN_FOLDER, MEDIAN_FOLDER, GAUSSIAN_FOLDER, SHARPENING_FOLDER, COMPARISON_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)


# ==============================
# MENCARI SEMUA CITRA
# ==============================
extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
image_files = []
for ext in extensions:
    image_files.extend(INPUT_FOLDER.glob(ext))
image_files = sorted(image_files)

if not image_files:
    print("Tidak ada citra yang ditemukan di direktori proyek!")
    exit()

print("=" * 60)
print("ENHANCEMENT CITRA UNTUK OCR (LAYOUT GRID 2x3 WORD-FRIENDLY)")
print("=" * 60)
print(f"Jumlah citra: {len(image_files)}\n")


# ==============================
# PROSES SETIAP CITRA
# ==============================
for image_path in image_files:
    print(f"Memproses: {image_path.name}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"  Gagal membaca: {image_path.name}")
        continue

    # 1. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Mean Filter
    mean_filter = cv2.blur(gray, (3, 3))

    # 3. Median Filter
    median_filter = cv2.medianBlur(gray, 3)

    # 4. Gaussian Filter
    gaussian_filter = cv2.GaussianBlur(gray, (3, 3), 0)

    # 5. Sharpening
    kernel_sharpening = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpening = cv2.filter2D(gray, -1, kernel_sharpening)

    filename = image_path.stem + ".png"

    # Simpan file satuan
    cv2.imwrite(str(GRAY_FOLDER / filename), gray)
    cv2.imwrite(str(MEAN_FOLDER / filename), mean_filter)
    cv2.imwrite(str(MEDIAN_FOLDER / filename), median_filter)
    cv2.imwrite(str(GAUSSIAN_FOLDER / filename), gaussian_filter)
    cv2.imwrite(str(SHARPENING_FOLDER / filename), sharpening)

    # ==========================================================
    # 6. MODEL PERBANDINGAN: GRID 2 KOLOM x 3 BARIS (COCOK DI WORD)
    # ==========================================================
    
    # Ubah semua ke BGR
    orig_bgr = image.copy()
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    mean_bgr = cv2.cvtColor(mean_filter, cv2.COLOR_GRAY2BGR)
    median_bgr = cv2.cvtColor(median_filter, cv2.COLOR_GRAY2BGR)
    gauss_bgr = cv2.cvtColor(gaussian_filter, cv2.COLOR_GRAY2BGR)
    sharp_bgr = cv2.cvtColor(sharpening, cv2.COLOR_GRAY2BGR)

    # Lebar panel disesuaikan agar pas di dokumen Word (tidak terlalu besar/kecil)
    panel_width = 400
    
    def resize_by_width(img, target_w):
        h = int(img.shape[0] * target_w / img.shape[1])
        return cv2.resize(img, (target_w, h))

    orig_bgr = resize_by_width(orig_bgr, panel_width)
    gray_bgr = resize_by_width(gray_bgr, panel_width)
    mean_bgr = resize_by_width(mean_bgr, panel_width)
    median_bgr = resize_by_width(median_bgr, panel_width)
    gauss_bgr = resize_by_width(gauss_bgr, panel_width)
    sharp_bgr = resize_by_width(sharp_bgr, panel_width)

    # Fungsi untuk memberi judul/label di atas gambar
    def add_top_label(img, text):
        label_h = 35
        canvas = cv2.copyMakeBorder(img, label_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        tx = (canvas.shape[1] - tw) // 2
        ty = 24
        cv2.putText(canvas, text, (tx, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)
        return canvas

    orig_labeled = add_top_label(orig_bgr, "1. Original")
    gray_labeled = add_top_label(gray_bgr, "2. Grayscale")
    mean_labeled = add_top_label(mean_bgr, "3. Mean Filter")
    median_labeled = add_top_label(median_bgr, "4. Median Filter")
    gauss_labeled = add_top_label(gauss_bgr, "5. Gaussian Filter")
    sharp_labeled = add_top_label(sharp_bgr, "6. Sharpening")

    # Menyamakan tinggi gambar per baris agar rapi saat digabung
    def make_pair(img_left, img_right):
        h_max = max(img_left.shape[0], img_right.shape[0])
        def pad_h(img, target_h):
            dh = target_h - img.shape[0]
            if dh > 0:
                return cv2.copyMakeBorder(img, 0, dh, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            return img
        
        l_padded = pad_h(img_left, h_max)
        r_padded = pad_h(img_right, h_max)
        
        # Jarak pemisah vertikal antar kolom (15 pixel warna abu-abu terang)
        gap_col = np.ones((h_max, 15, 3), dtype=np.uint8) * 220
        return cv2.hconcat([l_padded, gap_col, r_padded])

    row1 = make_pair(orig_labeled, gray_labeled)
    row2 = make_pair(mean_labeled, median_labeled)
    row3 = make_pair(gauss_labeled, sharp_labeled)

    # Menyamakan lebar antar baris jika ada selisih sedikit
    w_max = max(row1.shape[1], row2.shape[1], row3.shape[1])
    def pad_w(img, target_w):
        dw = target_w - img.shape[1]
        if dw > 0:
            return cv2.copyMakeBorder(img, 0, 0, 0, dw, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        return img

    row1 = pad_w(row1, w_max)
    row2 = pad_w(row2, w_max)
    row3 = pad_w(row3, w_max)

    # Jarak pemisah horizontal antar baris (20 pixel)
    gap_row = np.ones((20, w_max, 3), dtype=np.uint8) * 220

    # Gabungkan menjadi grid vertikal (Baris 1 -> Jarak -> Baris 2 -> Jarak -> Baris 3)
    comparison_grid = cv2.vconcat([row1, gap_row, row2, gap_row, row3])

    # Simpan hasil perbandingan baru ke folder
    cv2.imwrite(str(COMPARISON_FOLDER / filename), comparison_grid)

    print("  ✓ Selesai diproses & digrid 2x3.")

print("\n" + "=" * 60)
print("SELESAI. Model perbandingan kini berbentuk Grid 2x3 (Sangat rapi di Word).")
print("=" * 60)