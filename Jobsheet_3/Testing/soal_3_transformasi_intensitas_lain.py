import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float, color

# 1. Persiapan Citra
# Citra 1: Camera (sudah grayscale)
img1 = img_as_float(data.camera())
# Citra 2: Astronaut (konversi ke grayscale)
img2 = img_as_float(color.rgb2gray(data.astronaut()))

images = [img1, img2]
titles = ["Camera", "Astronaut"]

# 2. Definisi Transformasi
gamma_low = 0.5   # Memperterang
gamma_high = 2.0  # Mempergelap
c_log = 1.0       # Konstanta Log

# 3. Plotting (2 Baris x 4 Kolom)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

for i, img in enumerate(images):
    # Original
    axes[i, 0].imshow(img, cmap='gray')
    axes[i, 0].set_title(f"{titles[i]} (Original)")
    
    # Logarithmic
    # Formula: s = c * log(1 + r)
    img_log = exposure.adjust_log(img, gain=c_log)
    axes[i, 1].imshow(img_log, cmap='gray')
    axes[i, 1].set_title("Logarithmic")
    
    # Gamma Low (Gamma < 1)
    img_gamma_l = exposure.adjust_gamma(img, gamma=gamma_low)
    axes[i, 2].imshow(img_gamma_l, cmap='gray')
    axes[i, 2].set_title(f"Gamma {gamma_low} (Bright)")
    
    # Gamma High (Gamma > 1)
    img_gamma_h = exposure.adjust_gamma(img, gamma=gamma_high)
    axes[i, 3].imshow(img_gamma_h, cmap='gray')
    axes[i, 3].set_title(f"Gamma {gamma_high} (Dark)")

for ax in axes.ravel():
    ax.axis('off')

plt.tight_layout()
plt.show()