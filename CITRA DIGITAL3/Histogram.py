import cv2
import matplotlib.pyplot as plt
import os

# Membuat folder untuk menyimpan histogram
os.makedirs("hasil/histogram", exist_ok=True)

for i in range(1, 10):

    # Nama file gambar asli
    nama_file = f"gb{i}.jpeg"

    # Membaca gambar asli
    gb = cv2.imread(nama_file)

    # Mengecek apakah gambar berhasil dibaca
    if gb is None:
        print(f"gb {nama_file} tidak ditemukan!")
        continue

    # Mengubah BGR menjadi RGB untuk ditampilkan dengan benar
    gb_rgb = cv2.cvtColor(gb, cv2.COLOR_BGR2RGB)

    # Mengubah gambar menjadi grayscale
    grayscale = cv2.cvtColor(gb, cv2.COLOR_BGR2GRAY)

    # Membuat satu figure
    plt.figure(figsize=(12, 8))

    # ==========================================
    # HISTOGRAM CITRA ASLI (RGB)
    # ==========================================
    plt.subplot(2, 1, 1)

    plt.hist(
        gb_rgb[:, :, 0].ravel(),
        bins=256,
        range=(0, 256),
        alpha=0.5,
        label="Red"
    )

    plt.hist(
        gb_rgb[:, :, 1].ravel(),
        bins=256,
        range=(0, 256),
        alpha=0.5,
        label="Green"
    )

    plt.hist(
        gb_rgb[:, :, 2].ravel(),
        bins=256,
        range=(0, 256),
        alpha=0.5,
        label="Blue"
    )

    plt.title(f"Histogram Citra Asli - gb {i}")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.legend()
    plt.xlim(0, 255)

    # ==========================================
    # HISTOGRAM GRAYSCALE
    # ==========================================
    plt.subplot(2, 1, 2)

    plt.hist(
        grayscale.ravel(),
        bins=256,
        range=(0, 256)
    )

    plt.title(f"Histogram Grayscale - gb {i}")
    plt.xlabel("Intensitas Piksel")
    plt.ylabel("Jumlah Piksel")
    plt.xlim(0, 255)

    # Mengatur jarak antar histogram
    plt.tight_layout()

    # Menyimpan histogram
    nama_hasil = f"hasil/histogram/histogram{i}.png"
    plt.savefig(nama_hasil)

    # Menutup figure
    plt.close()

    print(f"Berhasil membuat: {nama_hasil}")

print("\nSemua histogram selesai dibuat!")