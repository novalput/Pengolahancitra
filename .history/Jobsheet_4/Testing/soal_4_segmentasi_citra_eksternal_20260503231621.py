import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, segmentation, color
from sklearn.cluster import KMeans
import warnings

# 1. Memuat citra dari folder assets
image_path = 'assets/durian.jpg'   # Pastikan file ada di folder assets
image_color = io.imread(image_path)

# Cek apakah citra grayscale atau berwarna
if len(image_color.shape) == 3:
    image_gray = color.rgb2gray(image_color)
    image_rgb = image_color
else:
    image_gray = image_color
    image_rgb = color.gray2rgb(image_color)

# --- Metode 1: Otsu Thresholding ---
thresh_otsu = filters.threshold_otsu(image_gray)
binary_otsu = image_gray > thresh_otsu

# --- Metode 2: K-Means Clustering (K=3) ---
pixel_features = image_gray.reshape(-1, 1)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    labels = kmeans.fit_predict(pixel_features)
segmented_kmeans = labels.reshape(image_gray.shape)

# --- Metode 3: Watershed ---
# Membuat elevation map menggunakan Sobel
elevation_map = filters.sobel(image_gray)

# PERBAIKAN: Gunakan dtype=np.int32 agar tidak terjadi error "No matching signature"
markers = np.zeros(image_gray.shape, dtype=np.int32)

# Menentukan marker berdasarkan ambang batas intensitas
# Nilai 0.2 dan 0.7 mungkin perlu disesuaikan dengan histogram gambar Anda
markers[image_gray < 0.2] = 1 # Marker latar belakang
markers[image_gray > 0.7] = 2 # Marker objek

# Terapkan Watershed
segmentation_watershed = segmentation.watershed(elevation_map, markers)

# Memvisualisasikan batas segmentasi pada citra asli
watershed_boundaries = segmentation.mark_boundaries(image_rgb, segmentation_watershed)

# 2. Visualisasi Hasil Perbandingan
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
ax = axes.ravel()

ax[0].imshow(image_rgb)
ax[0].set_title('Original Image (Assets)')
ax[0].axis('off')

ax[1].imshow(binary_otsu, cmap='gray')
ax[1].set_title('Otsu Thresholding')
ax[1].axis('off')

ax[2].imshow(segmented_kmeans, cmap='viridis')
ax[2].set_title(f'K-Means (K={n_clusters})')
ax[2].axis('off')

ax[3].imshow(watershed_boundaries)
ax[3].set_title('Watershed Segmentation')
ax[3].axis('off')

plt.tight_layout()
plt.show()
# Diskusi Singkat:
# Bandingkan apakah Otsu mampu memisahkan latar belakang dengan bersih.
# K-Means akan membagi citra menjadi 3 tingkatan keabuan berdasarkan klaster.
# Watershed akan sangat bergantung pada penempatan marker yang Anda tentukan di atas.