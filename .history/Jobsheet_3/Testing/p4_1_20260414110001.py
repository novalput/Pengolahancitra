import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# --- FUNGSI MATEMATIKA (Rotasi 3D) ---
def rot_x(theta):
    """Matriks rotasi sumbu X (Pitch / Menekuk Atas-Bawah)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(theta):
    """Matriks rotasi sumbu Y (Roll / Memutar Kanan-Kiri)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# --- KONFIGURASI ANATOMI ---
# Area 3: Lengan bawah (Statis di titik origin ke pergelangan)
forearm_start = np.array([0, -10, 0])
wrist_pos = np.array([0, 0, 0])

# Area 2: Punggung tangan (Dari pergelangan ke pangkal jari)
# Koordinat pangkal jari (Ibu jari, Telunjuk, Tengah, Manis, Kelingking)
base_fingers = [
    np.array([-2.5, 2.0, 0]),  # Ibu Jari
    np.array([-1.5, 4.0, 0]),  # Telunjuk
    np.array([ 0.0, 4.5, 0]),  # Tengah
    np.array([ 1.5, 4.0, 0]),  # Manis
    np.array([ 2.5, 3.0, 0])   # Kelingking
]

# Panjang ruas jari (3 ruas per jari)
phalanx_lengths = [1.5, 1.2, 0.8]

# --- SETUP PLOT MATPLOTLIB ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.35) # Menyisakan ruang untuk slider di bawah

# Elemen visual (Garis 3D)
lines_palm = [ax.plot([], [], [], 'b-', lw=3)[0] for _ in range(5)]
lines_fingers = [ax.plot([], [], [], 'r-', lw=2, marker='o', markersize=4)[0] for _ in range(5)]
line_forearm, = ax.plot([], [], [], 'k-', lw=5)

def update_plot(val):
    """Fungsi ini dipanggil setiap kali slider digeser"""
    # Ambil nilai dari slider dan ubah ke radian
    roll_angle = np.radians(slider_roll.val)    # Memutar Area 2
    pitch_angle = np.radians(slider_pitch.val)  # Menekuk Area 2
    bend_angle = np.radians(slider_finger.val)  # Menekuk Area 1

    # Matriks rotasi gabungan untuk Pergelangan Tangan (Area 2)
    R_wrist = rot_y(roll_angle) @ rot_x(pitch_angle)

    # 1. Gambar Area 3 (Lengan Bawah - Statis)
    line_forearm.set_data([forearm_start[0], wrist_pos[0]], [forearm_start[1], wrist_pos[1]])
    line_forearm.set_3d_properties([forearm_start[2], wrist_pos[2]])

    # 2. Hitung & Gambar Area 2 (Punggung Tangan) dan Area 1 (Jari-jari)
    for i in range(5):
        # Rotasi pangkal jari akibat pergerakan pergelangan
        rotated_base = R_wrist @ base_fingers[i]
        
        # Garis punggung tangan (dari pergelangan ke pangkal jari)
        lines_palm[i].set_data([wrist_pos[0], rotated_base[0]], [wrist_pos[1], rotated_base[1]])
        lines_palm[i].set_3d_properties([wrist_pos[2], rotated_base[2]])

        # Hitung posisi tiap ruas jari (Area 1)
        finger_points = [rotated_base]
        current_pos = rotated_base
        
        # Akumulasi rotasi untuk efek jari melingkar/menekuk
        current_bend = 0 
        
        for j in range(3): # 3 Ruas jari
            current_bend += bend_angle
            # Vektor ruas jari saat lurus (searah sumbu Y lokal punggung tangan)
            # Khusus ibu jari (index 0) arah naturalnya agak menyamping
            if i == 0:
                local_dir = np.array([-1, 1, 0])
            else:
                local_dir = np.array([0, 1, 0])
                
            local_dir = local_dir / np.linalg.norm(local_dir) * phalanx_lengths[j]
            
            # Terapkan tekukan jari (Rotasi X lokal), lalu ikuti rotasi pergelangan
            R_finger_bend = rot_x(-current_bend)
            dir_rotated = R_wrist @ (R_finger_bend @ local_dir)
            
            current_pos = current_pos + dir_rotated
            finger_points.append(current_pos)

        finger_points = np.array(finger_points)
        lines_fingers[i].set_data(finger_points[:, 0], finger_points[:, 1])
        lines_fingers[i].set_3d_properties(finger_points[:, 2])

    fig.canvas.draw_idle()

# --- SETUP SLIDER INTERAKTIF ---
axcolor = 'lightgoldenrodyellow'
ax_finger = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
ax_pitch = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_roll = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

# Slider (Area 1, Area 2)
slider_finger = Slider(ax_finger, 'Area 1: Tekuk Jari', 0.0, 90.0, valinit=0.0, valstep=1)
slider_pitch = Slider(ax_pitch, 'Area 2: Tekuk Atas/Bawah', -90.0, 90.0, valinit=0.0, valstep=1)
slider_roll = Slider(ax_roll, 'Area 2: Putar Kanan/Kiri', -90.0, 90.0, valinit=0.0, valstep=1)

slider_finger.on_changed(update_plot)
slider_pitch.on_changed(update_plot)
slider_roll.on_changed(update_plot)

# Konfigurasi batas tampilan 3D agar proporsional
ax.set_xlim([-10, 10])
ax.set_ylim([-12, 10])
ax.set_zlim([-10, 10])
ax.set_xlabel('Sumbu X (Kanan-Kiri)')
ax.set_ylabel('Sumbu Y (Panjang Lengan)')
ax.set_zlabel('Sumbu Z (Atas-Bawah)')
ax.set_title('Monitoring Pergerakan Lengan dan Jari 3D')

# Panggil update pertama kali untuk menggambar posisi awal
update_plot(0)

plt.show()