import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float, color

# 1. Load citra dan konversi ke grayscale
img_cam =  img_as_float(color.rgb2gray(data.astronaut()))

# 2. Transformasi Logaritmik: s = c * log(1 + r)
c_log = 1.0
img_log = c_log * np.log(1 + img_cam)

# 3. Transformasi Gamma: s = c * r^gamma
gamma_low = 0.5   # Memperterang (gamma < 1)
gamma_high = 1.5  # Mempergelap (gamma > 1)
img_gamma_low = exposure.adjust_gamma(img_cam, gamma=gamma_low)
img_gamma_high = exposure.adjust_gamma(img_cam, gamma=gamma_high)

# Plotting
fig, ax = plt.subplots(1, 4, figsize=(20, 5))
ax[0].imshow(img_cam, cmap='gray'); ax[0].set_title("Original")
ax[1].imshow(img_log, cmap='gray'); ax[1].set_title("Log Transformation")
ax[2].imshow(img_gamma_low, cmap='gray'); ax[2].set_title(f"Gamma = {gamma_low}")
ax[3].imshow(img_gamma_high, cmap='gray'); ax[3].set_title(f"Gamma = {gamma_high}")
for a in ax: a.axis('off')
plt.show()