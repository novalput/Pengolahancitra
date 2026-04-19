from skimage import data, utils, img_as_float, filter
from scipy import ndimage as ndi
import numpy as np
import matplotlib.pyplot as plt

# 1. Load citra dan tambah noise salt-and-pepper
img_orig = img_as_float(data.camera())
img_noisy = util.random_noise(img_orig, mode='s&p', amount=0.05)

# 2. Filter Rata-rata (Mean Filter) menggunakan kernel 3x3
kernel = np.ones((3, 3)) / 9
img_mean = ndi.convolve(img_noisy, kernel)

# 3. Filter Median menggunakan disk/window 3x3
img_median = filters.median(img_noisy, np.ones((3, 3)))

# Plotting
fig, ax = plt.subplots(1, 3, figsize=(18, 6))
ax[0].imshow(img_noisy, cmap='gray'); ax[0].set_title("Noisy (Salt & Pepper)")
ax[1].imshow(img_mean, cmap='gray'); ax[1].set_title("Mean Filter (Blurred)")
ax[2].imshow(img_median, cmap='gray'); ax[2].set_title("Median Filter (Clean)")

for a in ax: a.axis('off')
plt.show()