
Proyek ini dikembangkan untuk memenuhi tugas Pengolahan Citra Digital. Program ini melakukan analisis citra tanda tangan menggunakan dua metode thresholding (Global Thresholding dan Otsu Thresholding), menerapkan operasi morfologi (Opening dan Closing), menghitung karakteristik piksel foreground, serta mengevaluasi status keputusan (Signature Present atau Signature Absent).

---

## Panduan Instalasi & Cara Menjalankan Program

Ikuti langkah-langkah di bawah ini mulai dari proses kloning repositori hingga menjalankan program di komputer Anda:

### 1. Clone Repository (Kloning Repositori)
Buka terminal atau Command Prompt (CMD) di komputer Anda, lalu jalankan perintah berikut:
git clone https://github.com/Uciii07/tugas6CPD_Siti-Surti.git
cd tugas6CPD_Siti-Surti

### 2. Install Kebutuhan Modul (Dependencies)
Pastikan Python sudah terinstal di sistem Anda. Install pustaka yang diperlukan dengan menjalankan perintah:
pip install opencv-python numpy matplotlib

### 3. Persiapan Data Sampel Gambar
- Letakkan file gambar uji tanda tangan berformat .jpg, .jpeg, atau .png di dalam direktori utama folder proyek ini.
- Jika Anda ingin membuat sampel citra uji kosong (Signature Absent) secara otomatis, jalankan skrip berikut di terminal:
  python buat_sampel_absent.py

### 4. Jalankan Program Utama
Jalankan skrip utama untuk memproses seluruh citra secara otomatis:
python main.py

---

## Struktur Folder Output
Setelah program selesai dieksekusi, folder baru bernama hasil/ akan otomatis dibuat di dalam direktori proyek dengan struktur sebagai berikut:
- 1_roi_original/ : Kumpulan citra area minat (Region of Interest).
- 2_grayscale/ : Citra hasil konversi ke tingkat keabuan (grayscale).
- 3_global_thresh/ & 3b_global_closing/ : Hasil segmentasi metode Global Thresholding dan Morfologi Closing.
- 4_otsu_thresh/ & 4b_otsu_closing/ : Hasil segmentasi metode Otsu Thresholding dan Morfologi Closing.
- 7_perbandingan/ : Visualisasi gabungan lengkap berbentuk Grid 3x3 dari seluruh tahapan proses.
- rekapitulasi_hasil.csv : File laporan rekapitulasi data piksel foreground, rasio, dan status keputusan semua sampel.

---

## Author & Informasi Pengembang
- Nama / Author: Siti Surti
- NIM: F1G124077
- Mata Kuliah: Pengolahan Citra Digital
- Institusi: Universitas Halu Oleo
"""