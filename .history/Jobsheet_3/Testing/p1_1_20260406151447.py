import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, filters

# Ambil gambar bawaan dari skimage
image = data.coins()  # Menggunakan gambar koin bawaan dari skimage

# Konversi ke grayscale jika perlu
gray_image = color.rgb2gray(image) if image.ndim == 3 else image

# Hitung gradien menggunakan Operator Sobel
sobel_x = filters.sobel_h(gray_image)  # Sobel di arah horizontal
sobel_y = filters.sobel_v(gray_image)  # Sobel di arah vertikal
sobel_edge = filters.sobel(gray_image)  # Kombinasi keduanya

# Visualisasi hasil
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
ax = axes.ravel()

ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title("Gambar Asli")
ax[0].axis("off")

ax[1].imshow(sobel_x, cmap='gray')
ax[1].set_title("Gradien Sobel X")
ax[1].axis("off")

ax[2].imshow(sobel_y, cmap='gray')
ax[2].set_title("Gradien Sobel Y")
ax[2].axis("off")

ax[3].imshow(sobel_edge, cmap='gray')
ax[3].set_title("Deteksi Tepi Sobel")
ax[3].axis("off")

plt.tight_layout()
plt.show()

# Hasil Praktikum P1