import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

# Memuat gambar dari absolute path custom
def get_project_root():
    return os.path.abspath(os.path.dirname(__file__))

# Pastikan Anda memiliki folder 'assets' dan file 'pompom.png' di dalamnya
path_gambar_custom = os.path.join(get_project_root(), "assets", "pompom.png")

if os.path.exists(path_gambar_custom):
    gambar_sample = io.imread(path_gambar_custom)
else:
    # Placeholder jika file tidak ditemukan agar kode tidak error
    gambar_sample = np.zeros((512, 512, 3), dtype=np.uint8)
    print(f"Peringatan: File di {path_gambar_custom} tidak ditemukan.")

# Ukuran gambar (Ambil 2 elemen pertama saja untuk mengatasi gambar RGB maupun Grayscale)
height, width = gambar_sample.shape[:2]

# Buat plot
fig, ax = plt.subplots(figsize=(6, 6))

# Tampilkan gambar (cek apakah grayscale atau RGB)
ax.imshow(gambar_sample, cmap='gray' if len(gambar_sample.shape) == 2 else None)

# Tambahkan anotasi sumbu koordinat
ax.set_title(
    "Sistem Koordinat Kiri Atas dalam Citra Digital\n(Gambar Custom)",
    fontsize=12,
    fontweight="bold"
)
ax.set_xlabel("Sumbu X (Lebar) →", fontsize=10, color="blue")
ax.set_ylabel("↓ Sumbu Y (Tinggi)", fontsize=10, color="red")

# Tambahkan garis koordinat utama
ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label="Y = 0 (Atas)")
ax.axvline(x=0, color='blue', linestyle='--', linewidth=1, label="X = 0 (Kiri)")

# Tambahkan beberapa titik koordinat penting dengan warna kontras (cyan)
koordinat_titik = [
    (0, 0),
    (width - 1, 0),
    (0, height - 1),
    (width - 1, height - 1),
    (width // 2, height // 2)
]

for x, y in koordinat_titik:
    ax.scatter(x, y, color='cyan', s=50, edgecolors='black', linewidth=1.2)
    ax.text(
        x + 10, y + 10,
        f"({x}, {y})",
        color="cyan",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

# Tambahkan legenda
ax.legend(loc="upper right")

plt.show()