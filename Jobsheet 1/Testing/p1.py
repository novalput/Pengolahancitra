import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io
import os

# 1. Mengambil 3 gambar default dari scikit-image
gambar_coins = data.coins()
gambar_camera = data.camera()
gambar_astronaut = data.astronaut()

# 2. Menambahkan 1 gambar tambahan dari absolute path custom
def get_project_root():
    return os.path.abspath(os.path.dirname(__file__))
path_gambar_custom = os.path.join(get_project_root(), "assets", "pompom.png")
if os.path.exists(path_gambar_custom):
    gambar_custom = io.imread(path_gambar_custom)
else:
    # Placeholder jika file tidak ditemukan agar kode tidak error
    gambar_custom = np.zeros((512, 512, 3), dtype=np.uint8)
    print(f"Peringatan: File di {path_gambar_custom} tidak ditemukan.")

# Fungsi untuk menghitung ukuran gambar dalam bit, byte, KB, MB
def hitung_ukuran(image):
    # Grayscale (2 dimensi) = 8 bit, RGB (3 dimensi) = 24 bit
    bit_per_pixel = 8 if len(image.shape) == 2 else 24 
    total_pixels = image.shape[0] * image.shape[1]
    total_bits = total_pixels * bit_per_pixel
    total_bytes = total_bits / 8
    total_kb = total_bytes / 1024
    total_mb = total_kb / 1024
    return total_bits, total_bytes, total_kb, total_mb

# Menyusun gambar dan ukurannya dalam satu baris (1 baris, 4 kolom)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

# Daftar gambar dan nama
gambar_list = [gambar_coins, gambar_camera, gambar_astronaut, gambar_custom]
nama_list = ["Coins", "Camera", "Astronaut", "Custom Image"]

for ax, img, nama in zip(axes, gambar_list, nama_list):
    bits, bytes_, kb, mb = hitung_ukuran(img)
    
    # Cek apakah gambar grayscale atau RGB untuk colormap
    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    
    ax.set_title(f"{nama}\n{bits} bit, {bytes_:.2f} B\n{kb:.2f} KB, {mb:.4f} MB")
    ax.axis('off')

plt.tight_layout()
plt.show()