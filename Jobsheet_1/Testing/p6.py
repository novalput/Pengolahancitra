import matplotlib.pyplot as plt
from skimage import io
import os
import numpy as np

# Memuat gambar dari absolute path custom
def get_project_root():
    return os.path.abspath(os.path.dirname(__file__))

# Pastikan Anda memiliki folder 'assets' dan file 'pompom.png' di dalamnya
path_gambar_custom = os.path.join(get_project_root(), "assets", "pompom.png")

if os.path.exists(path_gambar_custom):
    gambar_rgb = io.imread(path_gambar_custom)
    
    # Jika gambar memiliki 4 channel (RGBA), ambil 3 channel pertama saja (RGB)
    if gambar_rgb.shape[-1] == 4:
        gambar_rgb = gambar_rgb[:, :, :3]
else:
    # Placeholder jika file tidak ditemukan agar kode tidak error
    gambar_rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    print(f"Peringatan: File di {path_gambar_custom} tidak ditemukan.")

# Ekstrak tiga kanal warna (R, G, B)
red_channel = gambar_rgb[:, :, 0]    # Kanal merah
green_channel = gambar_rgb[:, :, 1]  # Kanal hijau
blue_channel = gambar_rgb[:, :, 2]   # Kanal biru

# Ukuran gambar untuk memastikan koordinat tidak melebihi batas (Out of Bounds)
h, w = gambar_rgb.shape[:2]

# Koordinat tiga titik sampel untuk menampilkan nilai piksel RGB
titik_koordinat = [
    (min(100, w - 1), min(100, h - 1)), 
    (min(200, w - 1), min(150, h - 1)), 
    (min(300, w - 1), min(250, h - 1))
]

# Menampilkan gambar asli dan masing-masing kanal warna
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Tampilkan gambar asli dengan titik penanda
axes[0].imshow(gambar_rgb)
axes[0].set_title("Gambar Asli (RGB)")
for x, y in titik_koordinat:
    axes[0].scatter(x, y, color='yellow', s=50, edgecolors='black', linewidth=1.2)

# Tampilkan kanal merah dengan titik
axes[1].imshow(red_channel, cmap="Reds")
axes[1].set_title("Kanal Merah (Red)")
for x, y in titik_koordinat:
    axes[1].scatter(x, y, color='black', s=50, edgecolors='white', linewidth=1.2)

# Tampilkan kanal hijau dengan titik
axes[2].imshow(green_channel, cmap="Greens")
axes[2].set_title("Kanal Hijau (Green)")
for x, y in titik_koordinat:
    axes[2].scatter(x, y, color='black', s=50, edgecolors='white', linewidth=1.2)

# Tampilkan kanal biru dengan titik
axes[3].imshow(blue_channel, cmap="Blues")
axes[3].set_title("Kanal Biru (Blue)")
for x, y in titik_koordinat:
    axes[3].scatter(x, y, color='black', s=50, edgecolors='white', linewidth=1.2)

# Hilangkan sumbu koordinat
for ax in axes:
    ax.axis("off")

plt.suptitle(
    "Urutan Tiga Nilai Warna dalam Citra (Gambar Custom) dengan Titik Sampel",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()

# Menampilkan contoh nilai piksel untuk titik-titik yang dipilih
print("Nilai RGB pada titik koordinat yang dipilih:")
for i, (x, y) in enumerate(titik_koordinat):
    r_val = gambar_rgb[y, x, 0]
    g_val = gambar_rgb[y, x, 1]
    b_val = gambar_rgb[y, x, 2]

    print(f"Titik {i+1} - Koordinat ({x}, {y}):")
    print(f" Red (Merah)  : {r_val}")
    print(f" Green (Hijau): {g_val}")
    print(f" Blue (Biru)  : {b_val}")
    print(f" RGB Triplet  : ({r_val}, {g_val}, {b_val})\n")