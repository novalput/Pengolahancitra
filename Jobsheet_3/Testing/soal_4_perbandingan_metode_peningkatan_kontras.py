from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np

# 1. Load citra kontras rendah (misal: citra moon)
img = img_as_float(data.moon())

# 2. Contrast Stretching (Rescale Intensity)
# Merentangkan 2% - 98% persentil agar lebih stabil terhadap outlier
p2, p98 = np.percentile(img, (2, 98))
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

# 3. Histogram Equalization
img_eq = exposure.equalize_hist(img)

# Plotting Citra dan Histogram
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Citra
axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title("Original")
axes[0, 1].imshow(img_rescale, cmap='gray'); axes[0, 1].set_title("Contrast Stretching")
axes[0, 2].imshow(img_eq, cmap='gray'); axes[0, 2].set_title("Histogram Equalization")

# Histogram
axes[1, 0].hist(img.ravel(), bins=256, color='black'); axes[1, 0].set_title("Hist Original")
axes[1, 1].hist(img_rescale.ravel(), bins=256, color='black'); axes[1, 1].set_title("Hist Rescale")
axes[1, 2].hist(img_eq.ravel(), bins=256, color='black'); axes[1, 2].set_title("Hist Equalized")

plt.tight_layout()
plt.show()