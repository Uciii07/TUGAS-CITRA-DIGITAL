
import cv2, os, re, csv
import numpy as np
import pytesseract
from pytesseract import Output

FOLDER = "gambar"
OUT = "hasil_proses"
GT = "571012022000056"

for p in [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
]:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break
else:
    raise FileNotFoundError("Tesseract tidak ditemukan!")

TAHAP = [
    "1_citra_asli", "2_grayscale", "3_global_thresh",
    "4_global_opening", "5_global_closing", "6_otsu_thresh",
    "7_otsu_opening", "8_otsu_closing", "9_perbandingan"
]

for kategori in ["ttd", "nomor_ijazah"]:
    for t in TAHAP:
        os.makedirs(os.path.join(OUT, kategori, t), exist_ok=True)


def simpan(kategori, tahap, nama, img):
    cv2.imwrite(os.path.join(OUT, kategori, tahap, nama), img)


def cer(teks):
    a = list(teks)
    b = list(GT)
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        baru = [i]
        for j, y in enumerate(b, 1):
            baru.append(min(baru[-1]+1, dp[j]+1,
                            dp[j-1]+(x != y)))
        dp = baru
    return dp[-1] / len(GT)


def proses(gray, mask=None):
    hasil = {}
    _, g = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    _, o = cv2.threshold(gray, 0, 255,
                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if mask is not None:
        g[mask > 0] = 0
        o[mask > 0] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hasil["3_global_thresh"] = g
    hasil["4_global_opening"] = cv2.morphologyEx(g, cv2.MORPH_OPEN, k)
    hasil["5_global_closing"] = cv2.morphologyEx(
        hasil["4_global_opening"], cv2.MORPH_CLOSE, k, iterations=2)
    hasil["6_otsu_thresh"] = o
    hasil["7_otsu_opening"] = cv2.morphologyEx(o, cv2.MORPH_OPEN, k)
    hasil["8_otsu_closing"] = cv2.morphologyEx(
        hasil["7_otsu_opening"], cv2.MORPH_CLOSE, k, iterations=2)
    return hasil


def buat_mask(gray):
    d = pytesseract.image_to_data(
        gray, config="--psm 6", output_type=Output.DICT)
    mask = np.zeros(gray.shape, np.uint8)
    for i, teks in enumerate(d["text"]):
        try:
            if teks.strip() and float(d["conf"][i]) >= 20:
                x, y = d["left"][i], d["top"][i]
                w, h = d["width"][i], d["height"][i]
                cv2.rectangle(mask, (max(0,x-5), max(0,y-5)),
                              (x+w+5, y+h+5), 255, -1)
        except (ValueError, TypeError):
            pass
    return mask


def deteksi(gray, mask):
    tahap = proses(gray, mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    _, adapt = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adapt[mask > 0] = 0
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, k)
    adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k, iterations=2)

    kandidat = []
    for metode, b in [("Otsu", tahap["8_otsu_closing"]),
                      ("Adaptive", adapt)]:
        n, _, stats, _ = cv2.connectedComponentsWithStats(b, 8)
        h, w = gray.shape
        for i in range(1, n):
            x, y, bw, bh, area = stats[i, :5]
            if area < h*w*0.0003 or bw < w*0.08:
                continue
            rasio = bw / max(bh, 1)
            kepadatan = area / max(bw*bh, 1)
            skor = (2*(bw >= w*0.15) + 2*(rasio >= 2)
                    + (area >= h*w*0.001) + (kepadatan < 0.5))
            if skor >= 4:
                kandidat.append((skor, area, x, y, bw, bh, metode, b))

    if not kandidat:
        return "ABSENT", 0, None, tahap["8_otsu_closing"]

    _, area, x, y, w, h, metode, b = max(
        kandidat, key=lambda z: (z[0], z[1]))
    return "PRESENT", area/(gray.shape[0]*gray.shape[1])*100, (x,y,w,h), b


files = [f for f in sorted(os.listdir(FOLDER))
         if f.lower().endswith((".jpg", ".jpeg", ".png"))]
hasil = []

for nama in files:
    img = cv2.imread(os.path.join(FOLDER, nama))
    if img is None:
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dasar = os.path.splitext(nama)[0] + ".png"

    # Simpan tahap citra untuk nomor ijazah
    tn = {"1_citra_asli": img, "2_grayscale": gray}
    tn.update(proses(gray))
    for t, gambar in tn.items():
        simpan("nomor_ijazah", t, dasar, gambar)

    # OCR dengan tiga enhancement
    enh = {
        "Contrast Stretching": cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX),
        "CLAHE": cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(gray),
        "Gamma Correction": cv2.LUT(gray, np.array(
            [((i/255)**(1/1.5))*255 for i in range(256)]).astype("uint8"))
    }

    ocr = []
    for metode, gambar in enh.items():
        teks = pytesseract.image_to_string(gambar, config="--psm 6")
        kandidat = re.findall(r"\d{10,}", re.sub(r"\s+", "", teks))
        nomor = min(kandidat, key=lambda x: abs(len(x)-len(GT))) if kandidat else ""
        ocr.append((cer(nomor), metode, nomor, gambar))

    error, metode_terbaik, nomor, gambar_terbaik = min(ocr, key=lambda x:x[0])
    simpan("nomor_ijazah", "2_grayscale",
           os.path.splitext(nama)[0] + "_enhancement.png", gambar_terbaik)

    # Deteksi tanda tangan
    mask = buat_mask(gray)
    status, persen, kotak, binary = deteksi(gray, mask)
    tt = {"1_citra_asli": img, "2_grayscale": gray}
    tt.update(proses(gray, mask))
    for t, gambar in tt.items():
        simpan("ttd", t, dasar, gambar)

    anotasi = img.copy()
    if kotak:
        x, y, w, h = kotak
        cv2.rectangle(anotasi, (x,y), (x+w,y+h), (0,255,0), 3)

    simpan("ttd", "9_perbandingan", dasar, anotasi)
    simpan("nomor_ijazah", "9_perbandingan", dasar, gambar_terbaik)

    # Simpan perbandingan tahapan sebagai kolase
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for kategori, data in [("ttd", tt), ("nomor_ijazah", tn)]:
        panel = [img, gray] + [data[t] for t in TAHAP[2:8]]
        fig, ax = plt.subplots(4, 2, figsize=(10, 12))
        for a, gambar, judul in zip(ax.flat, panel, TAHAP[:8]):
            a.imshow(gambar, cmap="gray" if len(gambar.shape)==2 else None)
            a.set_title(judul)
            a.axis("off")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT, kategori, "9_perbandingan",
                                 os.path.splitext(nama)[0]+"_kolase.png"))
        plt.close(fig)

    hasil.append([nama, nomor, round(error*100, 2),
                  metode_terbaik, status, round(persen, 4)])

# Ringkasan terminal dan CSV
print("\nNAMA GAMBAR | HASIL OCR | CER (%) | ENHANCEMENT | TTD")
for r in hasil:
    print(r)

with open(os.path.join(OUT, "ringkasan_hasil.csv"),
          "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["Nama Gambar","Hasil OCR","CER (%)",
                "Enhancement Terbaik","Tanda Tangan","Area (%)"])
    w.writerows(hasil)

print("\nTotal gambar:", len(hasil))
print("Folder hasil:", os.path.abspath(OUT))