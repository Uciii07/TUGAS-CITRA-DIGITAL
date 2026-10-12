import os
import cv2
import numpy as np

FOLDER_INPUT = "."

def buat_sampel_kosong():
    # Ambil daftar gambar yang ada di folder saat ini
    files = sorted([f for f in os.listdir(FOLDER_INPUT) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    if not files:
        print("Tidak ada gambar ditemukan di folder untuk dijadikan referensi ukuran.")
        return

    print("=" * 60)
    print("MEMBUAT SAMPEL UJI 'SIGNATURE ABSENT' (KERTAS KOSONG)")
    print("=" * 60)

    # Buat 9 sampel uji kosong
    for i in range(1, 10):
        # Ambil referensi ukuran dari gambar pertama (atau bergantian)
        ref_img_path = files[(i - 1) % len(files)]
        ref_img = cv2.imread(ref_img_path)
        
        if ref_img is None:
            continue
            
        h, w, _ = ref_img.shape

        # Buat gambar kosong berwarna putih (menyerupai kertas kosong / tanpa tanda tangan)
        # Nilai 255 artinya putih penuh (background)
        blank_img = np.ones((h, w, 3), dtype=np.uint8) * 255

        # Nama file output sesuai permintaan
        nama_output = f"sampel uji {i}.jpg"
        path_output = os.path.join(FOLDER_INPUT, nama_output)

        cv2.imwrite(path_output, blank_img)
        print(f"Berhasil dibuat: {nama_output} (Ukuran: {w}x{h} piksel)")

    print("=" * 60)
    print("[SELESAI] 9 sampel uji absent (sampel uji 1 - 9) berhasil dibuat!")

if __name__ == "__main__":
    buat_sampel_kosong()