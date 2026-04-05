# Pengolahan Citra

Repositori ini berisi kumpulan tugas dan percobaan untuk mata kuliah Pengolahan Citra. Proyek ini dibagi ke dalam beberapa folder berdasarkan Jobsheet.

## Struktur Direktori

Pada setiap folder Jobsheet (misalnya `Jobsheet_1`), terdapat dua folder utama:
- **Final**: Folder ini berisi file Jupyter Notebook (`.ipynb`). Di sinilah kumpulan kode utuh dan laporan akhir untuk jobsheet tersebut berada.
- **Testing**: Folder ini digunakan untuk melakukan percobaan atau testing kode secara satuan menggunakan script Python (`.py`).

## Persiapan dan Cara Menjalankan Percobaan

Jika Anda ingin menjalankan atau mencoba kode pada folder **Testing**, Anda harus menginstal library yang dibutuhkan (requirements) terlebih dahulu. Berikut adalah langkah-langkahnya:

1. Buka Terminal atau Command Prompt.
2. Pindah (`cd`) ke dalam folder `Testing` pada Jobsheet yang ingin Anda coba. Sebagai contoh, untuk masuk ke Testing Jobsheet 1:
   ```bash
   cd Jobsheet_1/Testing
   ```
3. Lakukan instalasi dependensi yang dibutuhkan dengan menjalankan perintah pip berikut:
   ```bash
   pip install -r requirements.txt
   ```
4. Setelah instalasi berhasil, Anda sudah bisa menjalankan file percobaan, misalnya:
   ```bash
   python p1.py
   ```
