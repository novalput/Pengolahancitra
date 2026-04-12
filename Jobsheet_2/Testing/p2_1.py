import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters

# ===================== 1. DETEKSI TEPI ROBERTS PADA CITRA GRAYSCALE =====================

# Ambil gambar grayscale dari skimage
image_gray = data.coins()  # Menggunakan gambar koin (grayscale)

# Terapkan Operator Roberts
roberts_gray = filters.roberts(image_gray)

# ===================== 2. DETEKSI TEPI ROBERTS PADA CITRA BERWARNA =====================

# Ambil gambar berwarna dari skimage
image_color = data.astronaut()  # Menggunakan gambar astronaut (berwarna)

# Pisahkan kanal warna (R, G, B)
red_channel = image_color[:, :, 0]    # Kanal merah
green_channel = image_color[:, :, 1]  # Kanal hijau
blue_channel = image_color[:, :, 2]   # Kanal biru

# Terapkan Operator Roberts pada masing-masing kanal warna
roberts_red = filters.roberts(red_channel)
roberts_green = filters.roberts(green_channel)
roberts_blue = filters.roberts(blue_channel)

# Gabungkan hasil deteksi tepi dari ketiga kanal
roberts_color = np.stack((roberts_red, roberts_green, roberts_blue), axis=2)

# ===================== VISUALISASI HASIL =====================

fig, axes = plt.subplots(2, 5, figsize=(20, 10))
ax = axes.ravel()

# -------------------- Baris 1: Citra Grayscale --------------------
ax[0].imshow(image_gray, cmap='gray')
ax[0].set_title("Gambar Asli (Grayscale)")
ax[0].axis("off")

ax[1].imshow(roberts_gray, cmap='gray')
ax[1].set_title("Deteksi Tepi Roberts (Grayscale)")
ax[1].axis("off")

# Kosongkan posisi 3, 4, dan 5 agar baris pertama rapi
for i in range(2, 5):
    ax[i].axis("off")

# -------------------- Baris 2: Citra Berwarna --------------------
ax[5].imshow(image_color)
ax[5].set_title("Gambar Asli (Berwarna)")
ax[5].axis("off")

ax[6].imshow(roberts_red, cmap='Reds')
ax[6].set_title("Deteksi Tepi (Kanal Merah)")
ax[6].axis("off")

ax[7].imshow(roberts_green, cmap='Greens')
ax[7].set_title("Deteksi Tepi (Kanal Hijau)")
ax[7].axis("off")

ax[8].imshow(roberts_blue, cmap='Blues')
ax[8].set_title("Deteksi Tepi (Kanal Biru)")
ax[8].axis("off")

ax[9].imshow(roberts_color)
ax[9].set_title("Deteksi Tepi Gabungan RGB")
ax[9].axis("off")

plt.tight_layout()
plt.show()