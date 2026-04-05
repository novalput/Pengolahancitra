import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

# Menambahkan 1 gambar dari absolute path custom
def get_project_root():
    return os.path.abspath(os.path.dirname(__file__))

# Pastikan Anda memiliki folder 'assets' dan file 'pompom.png' di dalamnya
path_gambar_custom = os.path.join(get_project_root(), "assets", "pompom.png")

if os.path.exists(path_gambar_custom):
    gambar_custom = io.imread(path_gambar_custom)
else:
    # Placeholder jika file tidak ditemukan agar kode tidak error
    gambar_custom = np.zeros((512, 512, 3), dtype=np.uint8)
    print(f"Peringatan: File di {path_gambar_custom} tidak ditemukan.")

# Fungsi untuk menampilkan informasi array NumPy dari gambar
def info_array_numpy(image, nama):
    print(f"\n=== {nama} ===")
    print(f"Tipe Data: {type(image)}")  # Harusnya <class 'numpy.ndarray'>
    print(f"Dimensi: {image.shape}")  # (height, width, channels jika RGB)
    print(f"Tipe Nilai Piksel: {image.dtype}")  # (uint8)

    # Menampilkan sebagian kecil dari array (5x5 piksel pertama)
    print("Contoh nilai piksel (5x5 pertama):")
    print(image[:5, :5] if len(image.shape) == 2 else image[:5, :5, :])

# Menampilkan informasi gambar custom
info_array_numpy(gambar_custom, "Pompom (Custom)")

# Visualisasi bagaimana array NumPy mewakili citra
# Menggunakan 1 subplot karena hanya ada 1 gambar
fig, ax = plt.subplots(1, 1, figsize=(6, 5))

ax.imshow(gambar_custom, cmap='gray' if len(gambar_custom.shape) == 2 else None)
ax.set_title(f"Pompom (Custom)\nArray Shape: {gambar_custom.shape}")
ax.axis('off')

plt.tight_layout()
plt.show()