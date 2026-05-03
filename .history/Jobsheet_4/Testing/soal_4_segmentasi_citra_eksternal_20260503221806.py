import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters, segmentation, color, io
from sklearn.cluster import KMeans
from skimage.color import rgb2lab, lab2rgb
import warnings

# 1. Memuat citra (Menggunakan data.coins sebagai contoh citra sederhana)
# Anda bisa mengganti baris ini dengan: image = io.imread('URL_GAMBAR_INTERNET')
image = data.coins()
if len(image.shape) == 3: # Jika gambar berwarna
    image_gray = color.rgb2gray(image)
    image_rgb = image
else: # Jika gambar grayscale
    image_gray = image
    image_rgb = color.gray2rgb(image)

# --- Metode 1: Otsu Thresholding ---
thresh_otsu = filters.threshold_otsu(image_gray)
binary_otsu = image_gray > thresh_otsu

# --- Metode 2: K-Means Clustering (K=2) ---
# Menggunakan intensitas piksel untuk segmentasi foreground/background
pixel_features = image_gray.reshape(-1, 1)
kmeans = KMeans(n_clusters=2, random_state=0, n_init=10)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    labels = kmeans.fit_predict(pixel_features)
segmented_kmeans = labels.reshape(image_gray.shape)

# --- Metode 3: Watershed ---
# Membuat elevation map menggunakan Sobel
elevation_map = filters.sobel(image_gray)
# Menentukan marker secara otomatis berdasarkan nilai ekstrim intensitas
markers = np.zeros_like(image_gray)
markers[image_gray < 50] = 1 # Latar belakang
markers[image_gray > 150] = 2 # Objek
segmentation_watershed = segmentation.watershed(elevation_map, markers)
watershed_boundaries = segmentation.mark_boundaries(image_rgb, segmentation_watershed)

# 2. Visualisasi Hasil
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
ax = axes.ravel()

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(binary_otsu, cmap='gray')
ax[1].set_title('Otsu Thresholding')
ax[1].axis('off')

ax[2].imshow(segmented_kmeans, cmap='viridis')
ax[2].set_title('K-Means Clustering')
ax[2].axis('off')

ax[3].imshow(watershed_boundaries)
ax[3].set_title('Watershed Segmentation')
ax[3].axis('off')

plt.tight_layout()
plt.show()

# Diskusi Singkat:
# - Otsu cocok jika kontras objek dan latar sangat tegas.
# - K-Means bagus untuk memisahkan grup warna/intensitas namun tidak peduli lokasi piksel.
# - Watershed sangat unggul untuk memisahkan objek yang hampir berhimpit (seperti koin/sel).