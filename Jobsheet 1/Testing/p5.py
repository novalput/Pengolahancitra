# Menjelaskan model warna aditif RGB dalam citra digital dengan RGB
# Triplet dan warna hitam
import numpy as np
import matplotlib.pyplot as plt

# Membuat tiga kanal warna utama dalam model RGB
red_channel = np.zeros((100, 100, 3), dtype=np.uint8)
green_channel = np.zeros((100, 100, 3), dtype=np.uint8)
blue_channel = np.zeros((100, 100, 3), dtype=np.uint8)

# Mengatur warna merah, hijau, dan biru murni
red_channel[:, :, 0] = 255    # (255, 0, 0) - Merah
green_channel[:, :, 1] = 255  # (0, 255, 0) - Hijau
blue_channel[:, :, 2] = 255   # (0, 0, 255) - Biru

# Membuat kombinasi warna dari RGB
cyan = green_channel + blue_channel            # (0, 255, 255) - Cyan
magenta = red_channel + blue_channel           # (255, 0, 255) - Magenta
yellow = red_channel + green_channel           # (255, 255, 0) - Yellow
white = red_channel + green_channel + blue_channel  # (255, 255, 255) - Putih
black = np.zeros((100, 100, 3), dtype=np.uint8)     # (0, 0, 0) - Hitam

# Menampilkan hasil dengan RGB Triplet
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# List gambar, nama warna, dan RGB triplet
color_list = [red_channel, green_channel, blue_channel, white, cyan, magenta, yellow, black]
color_names = [
    "Merah (Red)", "Hijau (Green)", "Biru (Blue)", "Putih (White)", 
    "Cyan", "Magenta", "Kuning (Yellow)", "Hitam (Key)"
]
rgb_values = [
    "(255, 0, 0)", "(0, 255, 0)", "(0, 0, 255)", "(255, 255, 255)",
    "(0, 255, 255)", "(255, 0, 255)", "(255, 255, 0)", "(0, 0, 0)"
]

# Plot warna primer, hasil campuran, dan hitam dengan RGB triplet
for ax, color, name, rgb in zip(axes.flat, color_list, color_names, rgb_values):
    ax.imshow(color)
    ax.set_title(f"{name}\nRGB {rgb}", fontsize=10, fontweight="bold")
    ax.axis('off')

plt.suptitle(
    "Model Warna Aditif RGB dengan RGB Triplet & Hitam",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()
plt.show()