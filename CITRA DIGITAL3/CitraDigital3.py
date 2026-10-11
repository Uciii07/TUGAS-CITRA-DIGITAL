import cv2
import os
import matplotlib.pyplot as plt

# Membuat folder hasil jika belum ada
os.makedirs("hasil", exist_ok=True)

for i in range(1, 10):

    # Nama file gambar
    nama_file = f"gb{i}.jpeg"

    # Membaca gambar
    gb = cv2.imread(nama_file)

    # Cek apakah gambar berhasil dibaca
    if gb is None:
        print(f"gb {nama_file} tidak ditemukan!")
        continue

    # Mengubah gambar menjadi grayscale
    grayscale = cv2.cvtColor(gb, cv2.COLOR_BGR2GRAY)

    # Menyimpan hasil grayscale
    nama_hasil = f"hasil/grayscale{i}.jpg"
    cv2.imwrite(nama_hasil, grayscale)

    # Menampilkan gambar dengan ukuran lebih kecil
    plt.figure(figsize=(6, 4))
    plt.imshow(grayscale, cmap="gray")
    plt.title(f"Grayscale - gb {i}")
    plt.axis("off")
    plt.show()

    print(f"Berhasil: {nama_hasil}")