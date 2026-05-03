import matplotlib.pyplot as plt
from skimage import data, filters, segmentation, img_as_float, color
from sklearn.cluster import KMeans
import numpy as np
import warnings

# 1. Pilih satu citra untuk perbandingan (misal: camera)
image = data.camera()
image_float = img_as_float(image)

# 2. Lakukan beberapa metode segmentasi
# a) Otsu Thresholding
thresh_otsu = filters.threshold_otsu(image)
binary_otsu = image > thresh_otsu

# b) K-Means (misal K=3)
# Reshape untuk K-Means (1 fitur: intensitas)
rows, cols = image.shape
pixel_features = image_float.reshape(rows * cols, 1)
n_clusters = 3
kmeans = KMeans (n_clusters=n_clusters, random_state = 0, n_init = 10)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pixel_labels = kmeans.fit_predict(pixel_features)
segmented_kmeans_labels = pixel_labels.reshape(rows, cols)

# c) Watershed (gunakan marker sederhana dari Otsu)
elevation_map = filters.sobel (image)
markers = np.zeros_like (image)
markers [image <= thresh_otsu] = 1
markers [image > thresh_otsu] = 2
segmentation_watershed = segmentation.watershed (elevation_map, markers)

# 3. Visualisasi Perbandingan
fig, axes = plt.subplots (1, 4, figsize=(16, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap = plt.cm.gray)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(binary_otsu, cmap=plt.cm.gray)
ax[1].set_title('Otsu Thresholding')
ax[1].axis('off')

ax[2].imshow(segmented_kmeans_labels, cmap='viridis') # Gunakan cmap berbeda untuk label
ax[2].set_title(f'K-Means (K={n_clusters})')
ax[2].axis('off')

# Gunakan mark boundaries untuk Watershed agar lebih jelas
segmented_watershed_colored = segmentation.mark_boundaries (image_float, segmentation_watershed, color=(1,0,0)) # Batas merah
ax [3].imshow(segmented_watershed_colored)
ax [3].set_title('Watershed')
ax [3].axis('off')

plt.suptitle('Comparison of Segmentation Methods')
plt.tight_layout (rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
plt.show()