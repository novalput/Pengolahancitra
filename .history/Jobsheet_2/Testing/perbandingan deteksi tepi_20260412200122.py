import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, feature
from skimage.filters.rank import gradient
from skimage.morphology import disk

# ===================== 1. LOAD GAMBAR DARI HP =====================
# Ganti dengan path gambar kalian
image_path = '1.jpg'   # contoh: 'foto_hp.jpg'
image_color = io.imread(image_path)

# Konversi ke grayscale
image_gray = color.rgb2gray(image_color)

# ===================== 2. DETEKSI TEPI =====================

# --- GRAYSCALE ---
sobel_gray = filters.sobel(image_gray)
roberts_gray = filters.roberts(image_gray)
prewitt_gray = filters.prewitt(image_gray)
kirsch_gray = gradient((image_gray * 255).astype(np.uint8), disk(1))
canny_gray = feature.canny(image_gray, sigma=2)

# --- FUNGSI UNTUK RGB ---
def edge_rgb(image, operator):
    r = operator(image[:, :, 0])
    g = operator(image[:, :, 1])
    b = operator(image[:, :, 2])
    return np.stack((r, g, b), axis=2)

sobel_color = edge_rgb(image_color, filters.sobel)
roberts_color = edge_rgb(image_color, filters.roberts)
prewitt_color = edge_rgb(image_color, filters.prewitt)
kirsch_color = edge_rgb(image_color, lambda img: gradient(img, disk(1)))

# Canny hanya grayscale
canny_color = feature.canny(image_gray, sigma=2)

# ===================== 3. VISUALISASI =====================

fig, axes = plt.subplots(6, 4, figsize=(20, 20))
ax = axes.ravel()

# Baris 1: Gambar asli
ax[0].imshow(image_gray, cmap='gray')
ax[0].set_title("Grayscale")
ax[0].axis("off")

ax[2].imshow(image_color)
ax[2].set_title("RGB")
ax[2].axis("off")

# Sobel
ax[5].imshow(sobel_gray, cmap='gray')
ax[5].set_title("Sobel Gray")
ax[5].axis("off")

ax[7].imshow(sobel_color)
ax[7].set_title("Sobel RGB")
ax[7].axis("off")

# Roberts
ax[9].imshow(roberts_gray, cmap='gray')
ax[9].set_title("Roberts Gray")
ax[9].axis("off")

ax[11].imshow(roberts_color)
ax[11].set_title("Roberts RGB")
ax[11].axis("off")

# Prewitt
ax[13].imshow(prewitt_gray, cmap='gray')
ax[13].set_title("Prewitt Gray")
ax[13].axis("off")

ax[15].imshow(prewitt_color)
ax[15].set_title("Prewitt RGB")
ax[15].axis("off")

# Kirsch
ax[17].imshow(kirsch_gray, cmap='gray')
ax[17].set_title("Kirsch Gray")
ax[17].axis("off")

ax[19].imshow(kirsch_color)
ax[19].set_title("Kirsch RGB")
ax[19].axis("off")

# Canny
ax[21].imshow(canny_gray, cmap='gray')
ax[21].set_title("Canny Gray")
ax[21].axis("off")

ax[23].imshow(canny_color, cmap='gray')
ax[23].set_title("Canny (dari RGB)")
ax[23].axis("off")

# Matikan axis kosong
for i in range(len(ax)):
    ax[i].axis("off")

plt.tight_layout()
plt.show()