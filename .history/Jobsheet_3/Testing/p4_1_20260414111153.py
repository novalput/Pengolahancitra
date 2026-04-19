import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# --- FUNGSI MATEMATIKA (Rotasi 3D) ---
def rot_x(theta):
    """Pitch: Menekuk Atas-Bawah (Rotasi sumbu X)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def rot_y(theta):
    """Roll: Memutar Kanan-Kiri Lengan (Rotasi sumbu Y)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rot_z(theta):
    """Yaw: Melambai Kanan-Kiri (Rotasi sumbu Z)"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

# --- KONFIGURASI ANATOMI ---
# Area 3: Lengan bawah (Statis di titik origin ke pergelangan)
forearm_start = np.array([0, -10, 0])
wrist_pos = np.array([0, 0, 0])

# Area 2: Punggung tangan (Dari pergelangan ke pangkal jari)
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
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.45) # Ruang diperbesar untuk 4 slider

# Elemen visual
lines_palm = [ax.plot([], [], [], 'b-', lw=3)[0] for _ in range(5)]
lines_fingers = [ax.plot([], [], [], 'r-', lw=2, marker='o', markersize=4)[0] for _ in range(5)]
line_forearm, = ax.plot([], [], [], 'k-', lw=5)

def update_plot(val):
    roll_angle = np.radians(slider_roll.val)
    pitch_angle = np.radians(slider_pitch.val)
    yaw_angle = np.radians(slider_yaw.val)
    bend_angle = np.radians(slider_finger.val)

    # Matriks rotasi gabungan (Roll -> Yaw -> Pitch)
    # Urutan perkalian matriks menentukan bagaimana sumbu rotasi dipengaruhi
    R_wrist = rot_y(roll_angle) @ rot_z(yaw_angle) @ rot_x(pitch_angle)

    # 1. Gambar Area 3 (Lengan Bawah)
    line_forearm.set_data([forearm_start[0], wrist_pos[0]], [forearm_start[1], wrist_pos[1]])
    line_forearm.set_3d_properties([forearm_start[2], wrist_pos[2]])

    # 2. Gambar Area 2 dan Area 1
    for i in range(5):
        # Rotasi pangkal jari
        rotated_base = R_wrist @ base_fingers[i]
        
        # Garis punggung tangan
        lines_palm[i].set_data([wrist_pos[0], rotated_base[0]], [wrist_pos[1], rotated_base[1]])
        lines_palm[i].set_3d_properties([wrist_pos[2], rotated_base[2]])

        # Posisi tiap ruas jari
        finger_points = [rotated_base]
        current_pos = rotated_base
        current_bend = 0 
        
        for j in range(3):
            current_bend += bend_angle
            
            if i == 0:
                local_dir = np.array([-1, 1, 0]) # Ibu jari menyamping
            else:
                local_dir = np.array([0, 1, 0])  # Jari lain lurus
                
            local_dir = local_dir / np.linalg.norm(local_dir) * phalanx_lengths[j]
            
            # Tekukan jari dihitung relatif terhadap rotasi pergelangan total
            R_finger_bend = rot_x(-current_bend)
            dir_rotated = R_wrist @ (R_finger_bend @ local_dir)
            
            current_pos = current_pos + dir_rotated
            finger_points.append(current_pos)

        finger_points = np.array(finger_points)
        lines_fingers[i].set_data(finger_points[:, 0], finger_points[:, 1])
        lines_fingers[i].set_3d_properties(finger_points[:, 2])

    fig.canvas.draw_idle()

# --- SETUP SLIDER ---
axcolor = 'lightgoldenrodyellow'
ax_finger = plt.axes([0.15, 0.3, 0.65, 0.03], facecolor=axcolor)
ax_roll   = plt.axes([0.15, 0.22, 0.65, 0.03], facecolor=axcolor)
ax_pitch  = plt.axes([0.15, 0.16, 0.65, 0.03], facecolor=axcolor)
ax_yaw    = plt.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor) # Tambahan Slider Yaw

# Inisialisasi Slider
slider_finger = Slider(ax_finger, 'Jari: Tekuk', 0.0, 90.0, valinit=0.0, valstep=1)
slider_roll   = Slider(ax_roll, 'Wrist: Roll (Putar)', -90.0, 90.0, valinit=0.0, valstep=1)
slider_pitch  = Slider(ax_pitch, 'Wrist: Pitch (Atas/Bawah)', -90.0, 90.0, valinit=0.0, valstep=1)
slider_yaw    = Slider(ax_yaw, 'Wrist: Yaw (Kiri/Kanan)', -45.0, 45.0, valinit=0.0, valstep=1)

# Event Listener
slider_finger.on_changed(update_plot)
slider_roll.on_changed(update_plot)
slider_pitch.on_changed(update_plot)
slider_yaw.on_changed(update_plot) # Listener Yaw

# Konfigurasi 3D Plot
ax.set_xlim([-10, 10])
ax.set_ylim([-12, 10])
ax.set_zlim([-10, 10])
ax.set_xlabel('X (Yaw)')
ax.set_ylabel('Y (Roll)')
ax.set_zlabel('Z (Pitch)')
ax.set_title('Kinematika Tangan dengan Roll, Pitch, & Yaw')

update_plot(0)
plt.show()