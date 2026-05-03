import matplotlib.pyplot as plt
from skimage import data, filters
from skimage.color import rgb2gray

# 1. Memuat citra data.page()
image_page = data.page()

# 2. Penerapan Thresholding Otsu (Global)
thresh_otsu = filters.threshold_otsu(image_page)
binary_otsu = image_page > thresh_otsu

# 3. Penerapan Thresholding Yen (Global)
thresh_yen = filters.threshold_yen(image_page)
binary_yen = image_page > thresh_yen

# 4. Penerapan Thresholding Local (Adaptive)
# block_size harus ganjil, menentukan area lokal untuk perhitungan ambang
block_size = 35
binary_local = filters.threshold_local(image_page, block_size, offset=10)
# Hasil threshold_local berupa map ambang, perlu dibandingkan dengan citra asli
binary_local_result = image_page > binary_local

# 5. Visualisasi Perbandingan
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ax = axes.ravel()

ax[0].imshow(image_page, cmap=plt.cm.gray)
ax[0].set_title('Original Image (data.page)')
ax[0].axis('off')

ax[1].imshow(binary_otsu, cmap=plt.cm.gray)
ax[1].set_title(f'Otsu Thresholding (Global)')
ax[1].axis('off')

ax[2].imshow(binary_yen, cmap=plt.cm.gray)
ax[2].set_title('Yen Thresholding (Global)')
ax[2].axis('off')

ax[3].imshow(binary_local_result, cmap=plt.cm.gray)
ax[3].set_title(f'Local Thresholding (Adaptive, block={block_size})')
ax[3].axis('off')

plt.tight_layout()
plt.show()

# Analisis Singkat:
# Otsu biasanya tidak memuaskan pada data.page() karena adanya degradasi pencahayaan (shadow).
# Thresholding Local/Adaptive jauh lebih baik karena menghitung ambang batas di setiap area kecil.