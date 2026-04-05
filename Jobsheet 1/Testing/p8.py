import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

# 1. Memuat gambar dari absolute path custom
def get_project_root():
    return os.path.abspath(os.path.dirname(__file__))

# Pastikan Anda memiliki folder 'assets' dan file 'pompom.png' di dalamnya
path_gambar_custom = os.path.join(get_project_root(), "assets", "pompom.png")

if os.path.exists(path_gambar_custom):
    gambar = io.imread(path_gambar_custom)
    
    # PENTING: Jika gambar memiliki 4 channel (RGBA), ambil 3 channel pertama saja (RGB)
    # Karena format JPEG tidak mendukung channel Alpha (transparansi).
    if len(gambar.shape) == 3 and gambar.shape[-1] == 4:
        gambar = gambar[:, :, :3]
else:
    # Placeholder jika file tidak ditemukan
    gambar = np.zeros((512, 512, 3), dtype=np.uint8)
    print(f"Peringatan: File di {path_gambar_custom} tidak ditemukan.")

# 2. Simpan gambar dalam format lossy (JPEG) dan lossless (PNG)
jpeg_quality_50 = "custom_lossy_50.jpg"
jpeg_quality_90 = "custom_lossy_90.jpg"
png_lossless = "custom_lossless.png"

io.imsave(jpeg_quality_50, gambar, quality=50)  # JPEG lossy kualitas rendah
io.imsave(jpeg_quality_90, gambar, quality=90)  # JPEG lossy kualitas tinggi
io.imsave(png_lossless, gambar)  # PNG lossless

# 3. Fungsi untuk menghitung ukuran file
def ukuran_file(nama_file):
    size_bytes = os.path.getsize(nama_file)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    # Menggunakan \n agar tampilan teks di plot lebih rapi
    return f"{size_bytes} Bytes\n({size_kb:.2f} KB, {size_mb:.4f} MB)"

# Menampilkan ukuran file dari masing-masing format
nama_file_kompresi = [jpeg_quality_50, jpeg_quality_90, png_lossless]
ukuran_kompresi = [ukuran_file(file) for file in nama_file_kompresi]

# 4. Menampilkan gambar hasil kompresi
fig, axes = plt.subplots(1, 3, figsize=(15, 6))

kompresi_label = [
    "JPEG (Lossy, Q=50)",
    "JPEG (Lossy, Q=90)",
    "PNG (Lossless)"
]

for ax, file, label in zip(axes, nama_file_kompresi, kompresi_label):
    img = io.imread(file)
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    ax.set_title(
        f"{label}\nUkuran:\n{ukuran_file(file)}",
        fontsize=12,
        fontweight="bold",
        color="black",
        pad=10
    )
    ax.axis("off")

plt.suptitle(
    "Perbedaan Kompresi Lossy dan Lossless dalam Citra Digital\n(Gambar Custom)",
    fontsize=16,
    fontweight="bold",
    color="darkblue",
    y=1.05
)

plt.tight_layout()
plt.show()

# 5. Menampilkan informasi perbedaan kompresi dalam teks
print("Perbedaan Kompresi Lossy dan Lossless:")

# Replace digunakan agar saat di-print ke console tidak ada enter ganda dari fungsi ukuran_file
print("\n1. JPEG (Lossy, Quality=50):")
print("- Menggunakan kompresi lossy dengan kualitas rendah (Q=50).")
print("- Detail gambar berkurang, muncul artefak kompresi.")
print("- Ukuran file lebih kecil.")
print(f"- Ukuran file: {ukuran_kompresi[0].replace(chr(10), ' ')}")

print("\n2. JPEG (Lossy, Quality=90):")
print("- Menggunakan kompresi lossy dengan kualitas lebih tinggi (Q=90).")
print("- Detail gambar masih cukup baik, artefak lebih sedikit.")
print("- Ukuran file lebih besar dibanding Q=50, tetapi lebih kecil dibanding lossless.")
print(f"- Ukuran file: {ukuran_kompresi[1].replace(chr(10), ' ')}")

print("\n3. PNG (Lossless):")
print("- Menggunakan kompresi lossless, tidak ada kehilangan data.")
print("- Detail gambar tetap sempurna seperti aslinya.")
print("- Ukuran file lebih besar dibanding JPEG lossy.")
print(f"- Ukuran file: {ukuran_kompresi[2].replace(chr(10), ' ')}")