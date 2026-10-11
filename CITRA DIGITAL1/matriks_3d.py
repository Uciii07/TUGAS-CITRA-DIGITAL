def perkalian_matriks(A, B):
    depth = len(A)

    baris_A = len(A[0])
    kolom_A = len(A[0][0])

    baris_B = len(B[0])
    kolom_B = len(B[0][0])

    # Syarat perkalian matriks
    if kolom_A != baris_B:
        return None

    hasil = []

    # Perkalian setiap matriks pada array 3D
    for d in range(depth):
        matriks_hasil = []

        for i in range(baris_A):
            baris_hasil = []

            for j in range(kolom_B):
                jumlah = 0

                for k in range(kolom_A):
                    jumlah += A[d][i][k] * B[d][k][j]

                baris_hasil.append(jumlah)

            matriks_hasil.append(baris_hasil)

        hasil.append(matriks_hasil)

    return hasil


def tampilkan_array(array, nama):
    print(f"\nArray {nama}:")

    for d, matriks in enumerate(array):
        print(f"\nMatriks ke-{d + 1}")

        for baris in matriks:
            print(" ".join(map(str, baris)))


# ========================================
# PROGRAM UTAMA
# ========================================

print("=" * 40)
print("   PERKALIAN MATRIKS ARRAY 3D")
print("=" * 40)


# ========================================
# ARRAY A
# ========================================

A = [
    [
        [11, 22],
        [35, 48]
    ],

    [
        [59, 67],
        [77, 88]
    ]
]


# ========================================
# ARRAY B
# ========================================

B = [
    [
        [19, 15],
        [60, 12]
    ],

    [
        [14, 73],
        [78, 77]
    ]
]


# ========================================
# PERKALIAN
# ========================================

hasil = perkalian_matriks(A, B)


# ========================================
# MENAMPILKAN HASIL
# ========================================

tampilkan_array(A, "A")
tampilkan_array(B, "B")

print("\n" + "=" * 40)
print("        HASIL PERKALIAN A x B")
print("=" * 40)

tampilkan_array(hasil, "A x B")