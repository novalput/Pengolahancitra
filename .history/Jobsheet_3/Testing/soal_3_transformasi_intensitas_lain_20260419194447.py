import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float

# 1. Load citra kamera (grayscale)
img = img_as_float(data.camera())

# 2. Terapkan dua nilai Gamma yang berlawanan
# Gamma < 1: Memperlebar rentang intensitas rendah (citra jadi lebih terang)
img_bright = exposure.adjust_gamma(img, gamma=0.5)

# Gamma > 1: Menekan rentang intensitas rendah (citra jadi lebih gelap/kontras tinggi)
img_dark = exposure.adjust_gamma(img, gamma=2.0)

# 3. Plotting langsung berdampingan
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(img_bright, cmap='gray')
ax[0].set_title("Gamma 0.5 (Mencerahkan Bayangan)")
ax[0].axis('off')

ax[1].imshow(img_dark, cmap='gray')
ax[1].set_title("Gamma 2.0 (Memperdalam Gelap)")
ax[1].axis('off')

plt.tight_layout()
plt.show()