from skimage import io
import matplotlib.pyplot as plt
from skimage import data

# Membuat citra contoh dari skimage
gambar = data.astronaut()

# Simpan gambar dalam format BMP, JPEG, dan TIFF
io.imsave("astronaut.bmp", gambar)
io.imsave("astronaut.jpg", gambar, quality=95)  # JPEG menggunakan parameter quality
io.imsave("astronaut.tiff", gambar)

# Fungsi untuk menghitung ukuran file
import os

def ukuran_file(nama_file):
    size_bytes = os.path.getsize(nama_file)
    size_kb = size_bytes / 1024
    size_mb = size_kb / 1024
    return f"{size_bytes} Bytes ({size_kb:.2f} KB, {size_mb:.4f} MB)"

# Menampilkan informasi ukuran file dari masing-masing format
format_citra = ["BMP", "JPEG", "TIFF"]
nama_file = ["astronaut.bmp", "astronaut.jpg", "astronaut.tiff"]
ukuran = [ukuran_file(file) for file in nama_file]

# Tampilkan gambar dan informasi karakteristik format citra
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, fmt, file in zip(axes, format_citra, nama_file):
    img = io.imread(file)
    ax.imshow(img)
    ax.set_title(
        f"{fmt}\nUkuran:\n{ukuran_file(file)}",
        fontsize=12,
        fontweight="bold",
        color="black",
        pad=10
    )
    ax.axis("off")

plt.suptitle(
    "Karakteristik Format Citra BMP, JPEG, dan TIFF",
    fontsize=16,
    fontweight="bold",
    color="darkblue",
    y=1.05
)

plt.tight_layout()
plt.show()

# Menampilkan informasi karakteristik format citra
print("Karakteristik Format Citra:")

print("\n1. BMP (Bitmap):")
print("- Format tidak terkompresi, menghasilkan ukuran file yang besar.")
print("- Kualitas gambar sangat baik, tidak ada kehilangan data.")
print("- Cocok untuk pengolahan citra yang memerlukan ketelitian tinggi.")
print(f"- Ukuran file: {ukuran[0]}")

print("\n2. JPEG (Joint Photographic Experts Group):")
print("- Format terkompresi dengan metode lossy (menghilangkan sebagian data).")
print("- Ukuran file lebih kecil dibanding BMP dan TIFF.")
print("- Cocok untuk fotografi, media sosial, dan tampilan web.")
print(f"- Ukuran file: {ukuran[1]}")

print("\n3. TIFF (Tagged Image File Format):")
print("- Format fleksibel yang mendukung kompresi lossy atau lossless.")
print("- Digunakan dalam pencetakan profesional dan pemrosesan citra medis.")
print("- Ukuran file bisa besar tergantung tingkat kompresi yang digunakan.")
print(f"- Ukuran file: {ukuran[2]}")