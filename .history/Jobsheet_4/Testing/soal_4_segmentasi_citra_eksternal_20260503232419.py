import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, segmentation, color, util
from sklearn.cluster import KMeans
import warnings

# 1. Memuat citra dari folder assets
image_path = 'assets/dur.jpg' 
image_color = io.imread(image_path)

# Konversi ke float agar rentang 0-1 konsisten
image_float = util.img_as_float(image_color)

if len(image_float.shape) == 3:
    image_gray = color.rgb2gray(image_float)
    image_rgb = image_float
else:
    image_gray = image_float
    image_rgb = color.gray2rgb(image_float)

# --- Metode 1: Otsu Thresholding ---
thresh_otsu = filters.threshold_otsu(image_gray)
binary_otsu = image_gray > thresh_otsu

# --- Metode 2: K-Means Clustering ---
pixel_features = image_gray.reshape(-1, 1)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    labels = kmeans.fit_predict(pixel_features)
segmented_kmeans = labels.reshape(image_gray.shape)

# --- Metode 3: Watershed (DIPERBAIKI) ---
# Menggunakan Sobel untuk peta elevasi (tepi)
elevation_map = filters.sobel(image_gray)

# Membuat marker yang lebih cerdas:
# Kita ambil nilai sangat rendah sebagai background dan sangat tinggi sebagai foreground
markers = np.zeros(image_gray.shape, dtype=np.int32)

# Menggunakan persentil agar adaptif terhadap kecerahan gambar durian
low_limit = np.percentile(image_gray, 10)  # 10% piksel tergelap
high_limit = np.percentile(image_gray, 90) # 10% piksel terang

markers[image_gray < low_limit] = 1
markers[image_gray > high_limit] = 2

# Terapkan Watershed
segmentation_watershed = segmentation.watershed(elevation_map, markers)

# Menghasilkan batas (boundaries) yang kontras (misal warna kuning [1, 1, 0])
watershed_boundaries = segmentation.mark_boundaries(image_rgb, segmentation_watershed, color=(1, 1, 0))

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