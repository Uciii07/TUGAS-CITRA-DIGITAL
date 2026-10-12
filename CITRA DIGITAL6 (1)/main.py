import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import cv2

# =========================================================
# KONFIGURASI FOLDER & PARAMETER UTAMA
# =========================================================
FOLDER_INPUT = "."
FOLDER_HASIL = "hasil"

GLOBAL_THRESHOLD_VAL = 127
THRESHOLD_PERCENTAGE = 2.0  # Batas minimal rasio piksel %
PIXEL_THRESHOLD = 1500      

KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# Daftar sub-folder penyimpanan lengkap
SUB_FOLDERS = {
    "roi": os.path.join(FOLDER_HASIL, "1_roi_original"),
    "gray": os.path.join(FOLDER_HASIL, "2_grayscale"),
    "global_thresh": os.path.join(FOLDER_HASIL, "3_global_thresh"),
    "global_opening": os.path.join(FOLDER_HASIL, "3a_global_opening"),
    "global_closing": os.path.join(FOLDER_HASIL, "3b_global_closing"),
    "otsu_thresh": os.path.join(FOLDER_HASIL, "4_otsu_thresh"),
    "otsu_opening": os.path.join(FOLDER_HASIL, "4a_otsu_opening"),
    "otsu_closing": os.path.join(FOLDER_HASIL, "4b_otsu_closing"),
    "perbandingan": os.path.join(FOLDER_HASIL, "7_perbandingan")
}

# Buat folder utama dan seluruh sub-folder secara otomatis
for folder_path in SUB_FOLDERS.values():
    os.makedirs(folder_path, exist_ok=True)


# =========================================================
# FUNGSI PROSES CITRA LENGKAP
# =========================================================
def proses_dan_ekspor_folder(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Gagal membaca file: {image_path}")
        return None

    nama_file = os.path.basename(image_path)
    nama_tanpa_ext = os.path.splitext(nama_file)[0]

    roi = img
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    total_pixels = gray.shape[0] * gray.shape[1]

    # 1. JALUR GLOBAL THRESHOLDING + MORFOLOGI
    _, thresh_global = cv2.threshold(gray, GLOBAL_THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)
    g_open = cv2.morphologyEx(thresh_global, cv2.MORPH_OPEN, KERNEL, iterations=1)
    g_close = cv2.morphologyEx(g_open, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    
    px_global = cv2.countNonZero(g_close)
    ratio_global = (px_global / total_pixels) * 100

    # 2. JALUR OTSU THRESHOLDING + MORFOLOGI
    otsu_val, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    o_open = cv2.morphologyEx(thresh_otsu, cv2.MORPH_OPEN, KERNEL, iterations=1)
    o_close = cv2.morphologyEx(o_open, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    
    px_otsu = cv2.countNonZero(o_close)
    ratio_otsu = (px_otsu / total_pixels) * 100

    # 3. ATURAN KEPUTUSAN (DECISION RULE)
    status = "SIGNATURE PRESENT" if ratio_otsu > THRESHOLD_PERCENTAGE and px_otsu > PIXEL_THRESHOLD else "SIGNATURE ABSENT"

    # SIMPAN FILE SATUAN KE MASING-MASING SUB-FOLDER
    cv2.imwrite(os.path.join(SUB_FOLDERS["roi"], f"{nama_tanpa_ext}_roi.png"), roi)
    cv2.imwrite(os.path.join(SUB_FOLDERS["gray"], f"{nama_tanpa_ext}_gray.png"), gray)
    cv2.imwrite(os.path.join(SUB_FOLDERS["global_thresh"], f"{nama_tanpa_ext}_global.png"), thresh_global)
    cv2.imwrite(os.path.join(SUB_FOLDERS["global_opening"], f"{nama_tanpa_ext}_global_open.png"), g_open)
    cv2.imwrite(os.path.join(SUB_FOLDERS["global_closing"], f"{nama_tanpa_ext}_global_close.png"), g_close)
    cv2.imwrite(os.path.join(SUB_FOLDERS["otsu_thresh"], f"{nama_tanpa_ext}_otsu.png"), thresh_otsu)
    cv2.imwrite(os.path.join(SUB_FOLDERS["otsu_opening"], f"{nama_tanpa_ext}_otsu_open.png"), o_open)
    cv2.imwrite(os.path.join(SUB_FOLDERS["otsu_closing"], f"{nama_tanpa_ext}_otsu_close.png"), o_close)

    # 4. VISUALISASI PERBANDINGAN GABUNGAN (GRID 3x3: MENAMPILKAN SEMUA TAHAPAN)
    plt.figure(figsize=(14, 12))
    plt.suptitle(f"Analisis Lengkap Segmentasi & Morfologi: {nama_file}", fontsize=14, fontweight='bold')

    # Panel 1: Citra Original ROI
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("1. Citra Original (ROI)", fontsize=10)
    plt.axis("off")

    # Panel 2: Grayscale
    plt.subplot(3, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("2. Grayscale", fontsize=10)
    plt.axis("off")

    # Panel 3: Global Threshold
    plt.subplot(3, 3, 3)
    plt.imshow(thresh_global, cmap="gray")
    plt.title(f"3. Global Threshold (T={GLOBAL_THRESHOLD_VAL})", fontsize=10)
    plt.axis("off")

    # Panel 4: Otsu Threshold
    plt.subplot(3, 3, 4)
    plt.imshow(thresh_otsu, cmap="gray")
    plt.title(f"4. Otsu Threshold (T={int(otsu_val)})", fontsize=10)
    plt.axis("off")

    # Panel 5: Global Opening
    plt.subplot(3, 3, 5)
    plt.imshow(g_open, cmap="gray")
    plt.title("5. Global Opening", fontsize=10)
    plt.axis("off")

    # Panel 6: Global Closing
    plt.subplot(3, 3, 6)
    plt.imshow(g_close, cmap="gray")
    plt.title(f"6. Global Closing\n(Piksel FG: {px_global})", fontsize=10)
    plt.axis("off")

    # Panel 7: Otsu Opening
    plt.subplot(3, 3, 7)
    plt.imshow(o_open, cmap="gray")
    plt.title("7. Otsu Opening", fontsize=10)
    plt.axis("off")

    # Panel 8: Otsu Closing
    plt.subplot(3, 3, 8)
    plt.imshow(o_close, cmap="gray")
    plt.title(f"8. Otsu Closing\n(Piksel FG: {px_otsu})", fontsize=10)
    plt.axis("off")

    # Panel 9: Status Keputusan & Karakteristik Piksel
    plt.subplot(3, 3, 9)
    warna_teks = "green" if status == "SIGNATURE PRESENT" else "red"
    plt.text(0.5, 0.5, f"{status}\n\nPiksel Otsu: {px_otsu} px\nRasio: {ratio_otsu:.2f}%\nPiksel Global: {px_global} px", 
             fontsize=10, color=warna_teks, ha='center', va='center', weight='bold',
             bbox=dict(boxstyle="round,pad=0.5", edgecolor=warna_teks, facecolor="white", lw=2))
    plt.title("9. Status Keputusan", fontsize=10)
    plt.axis("off")

    plt.tight_layout()
    path_perbandingan = os.path.join(SUB_FOLDERS["perbandingan"], f"Perbandingan_{nama_tanpa_ext}.png")
    plt.savefig(path_perbandingan, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Selesai: {nama_file:<30} | Global: {px_global} px | Otsu: {px_otsu} px | Status: {status}")
    
    return [nama_file, px_global, f"{ratio_global:.2f}%", int(otsu_val), px_otsu, f"{ratio_otsu:.2f}%", status]


# =========================================================
# PROGRAM UTAMA
# =========================================================
def main():
    if not os.path.exists(FOLDER_INPUT):
        print(f"Folder '{FOLDER_INPUT}' tidak ditemukan!")
        return

    files = sorted([f for f in os.listdir(FOLDER_INPUT) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if not files:
        print(f"Tidak ada gambar di dalam folder '{FOLDER_INPUT}'.")
        return

    rekap_data = []

    print("=" * 110)
    print("PROSES ANALISIS LENGKAP: THRESHOLDING & MORFOLOGI (GRID 3x3)")
    print("=" * 110)

    for file_name in files:
        full_path = os.path.join(FOLDER_INPUT, file_name)
        row = proses_dan_ekspor_folder(full_path)
        if row:
            rekap_data.append(row)

    # Simpan Rekapitulasi Data ke CSV
    csv_path = os.path.join(FOLDER_HASIL, "rekapitulasi_hasil.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Nama File", "Piksel Global", "Rasio Global", "Nilai Threshold Otsu", "Piksel Otsu", "Rasio Otsu", "Status Keputusan"])
        writer.writerows(rekap_data)

    print("=" * 110)
    print(f"[BERHASIL] Seluruh hasil perbandingan lengkap (Grid 3x3) dan CSV tersimpan di folder '{FOLDER_HASIL}/'!")


if __name__ == "__main__":
    main()