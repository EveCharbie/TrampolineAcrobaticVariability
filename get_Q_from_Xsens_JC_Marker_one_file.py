import pickle
import matplotlib.pyplot as plt
import biorbd
import numpy as np
import bioviz
import os
import scipy

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from TrampolineAcrobaticVariability.Function.Function_build_model import (
    convert_marker_to_local_frame,
    calculer_rotation_et_angle,
)
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import find_index

parent_list_xsens_JC = [
    "Pelvis",  # 0
    "L5",  # delete
    "L3",  # delete
    "T12",  # delete
    "T8",  # delete
    "Neck",  # delete
    "Head",  # 1
    "ShoulderR",  # delete
    "UpperArmR",  # delete
    "LowerArmR",  # 2
    "HandR",  # 3
    "ShoulderL",  # delete
    "UpperArmL",  # delete
    "LowerArmL",  # 4
    "HandL",  # 5
    "UpperLegR",  # 6
    "LowerLegR",  # 7
    "FootR",  # 8
    "ToesR",  # delete
    "UpperLegL",  # 9
    "LowerLegL",  # 10
    "FootL",  # 11
    "ToesL",  # delete
]

chemin_fichier_pkl = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/GuSe/831</a4ff449f_0_0-120_283__831<__0__eyetracking_metrics.pkl"
with open(chemin_fichier_pkl, "rb") as fichier_pkl:
    # Charger les données à partir du fichier ".pkl"
    eye_tracking_metrics = pickle.load(fichier_pkl)

expertise = eye_tracking_metrics["subject_expertise"]
subject_name = eye_tracking_metrics["subject_name"]

Xsens_orientation_per_move = eye_tracking_metrics["Xsens_orientation_per_move"]
Xsens_position_rotated_per_move = eye_tracking_metrics["Xsens_position_rotated_per_move"]

n_frames = Xsens_position_rotated_per_move.shape[0]
Xsens_position = Xsens_position_rotated_per_move.reshape(n_frames, 23, 3).transpose(2, 1, 0)

##
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [])
def init():
    sc._offsets3d = ([], [], [])
    return sc,
def update(frame):
    x = Xsens_position[0, :, frame]
    y = Xsens_position[1, :, frame]
    z = Xsens_position[2, :, frame]
    sc._offsets3d = (x, y, z)
    return sc,
ani = FuncAnimation(fig, update, frames=range(Xsens_position.shape[2]), init_func=init, blit=False)
plt.show()
##

# Ne selectionner que les articulations necessaire
indices_a_supprimer = [1, 2, 3, 4, 5, 7, 8, 11, 12, 18, 22]

indices_total = range(Xsens_position.shape[1])
indices_a_conserver = [i for i in indices_total if i not in indices_a_supprimer]
Xsens_positions_complet = Xsens_position[:, indices_a_conserver, :]
parent_list_xsens_JC_complet = [jc for i, jc in enumerate(parent_list_xsens_JC) if i not in indices_a_supprimer]

indices_reels_colonnes_a_supprimer = []
for indice in indices_a_supprimer:
    indices_reels_colonnes_a_supprimer.extend(range(indice * 4, indice * 4 + 4))
mask_colonnes = np.ones(Xsens_orientation_per_move.shape[1], dtype=bool)
mask_colonnes[indices_reels_colonnes_a_supprimer] = False

Xsens_orientation_per_move_complet = Xsens_orientation_per_move[:, mask_colonnes]

nb_mat = Xsens_orientation_per_move_complet.shape[1]//4
Q = np.zeros((nb_mat * 3, n_frames))

n_markers = len(parent_list_xsens_JC_complet)

Jc_in_pelvis_frame = np.ndarray((3, n_markers, n_frames))

for i in range(n_frames):
    mid_hip_pos = (Xsens_positions_complet[:, find_index("UpperLegR", parent_list_xsens_JC_complet), i] +
                   Xsens_positions_complet[:, find_index("UpperLegL", parent_list_xsens_JC_complet), i]) / 2

    rot_mov = calculer_rotation_et_angle(find_index("Pelvis", parent_list_xsens_JC_complet),
                                         Xsens_orientation_per_move_complet[i, :])

    for idx, jcname in enumerate(parent_list_xsens_JC_complet):

        if idx == find_index("Pelvis", parent_list_xsens_JC_complet):
            Rotation_pelvis = biorbd.Rotation(
                rot_mov[0, 0],
                rot_mov[0, 1],
                rot_mov[0, 2],
                rot_mov[1, 0],
                rot_mov[1, 1],
                rot_mov[1, 2],
                rot_mov[2, 0],
                rot_mov[2, 1],
                rot_mov[2, 2],
            )
            Jc_in_pelvis_frame[:, idx, i] = biorbd.Rotation.toEulerAngles(
                Rotation_pelvis, "xyz").to_array()
            # Jc_in_pelvis_frame[:, idx, i] = mid_hip_pos

        else:
            P2_prime = convert_marker_to_local_frame(mid_hip_pos, rot_mov, Xsens_positions_complet[:, idx, i])
            Jc_in_pelvis_frame[:, idx, i] = P2_prime

# Jc_in_pelvis_frame[:, 0:3, :] = np.unwrap(Jc_in_pelvis_frame[:, 0:3, :], axis=2)
# Jc_in_pelvis_frame[:, 0, :] = np.unwrap(Jc_in_pelvis_frame[:, 0, :])
# Jc_in_pelvis_frame[:, 1, :] = np.unwrap(Jc_in_pelvis_frame[:, 1, :])
# Jc_in_pelvis_frame[:, 2, :] = np.unwrap(Jc_in_pelvis_frame[:, 2, :])
Jc_in_pelvis_frame = np.unwrap(Jc_in_pelvis_frame)


colors = ['r', 'g', 'b']
n_rows = int(np.ceil(Jc_in_pelvis_frame.shape[1] / 4))
plt.figure(figsize=(20, 3 * n_rows))

for idx, jcname in enumerate(parent_list_xsens_JC_complet):
    ax = plt.subplot(n_rows, 4, idx + 1)
    for j in range(Jc_in_pelvis_frame.shape[0]):
        ax.plot(Jc_in_pelvis_frame[j, idx, :], color=colors[j], label=f'Composante {["X", "Y", "Z"][j]}')
    ax.set_title(f'Graphique {jcname}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Valeur')
    if idx == 0:
        ax.legend()
plt.tight_layout()
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [])
def init():
    sc._offsets3d = ([], [], [])
    return sc,
def update(frame):
    x = Jc_in_pelvis_frame[0, 1:, frame]
    y = Jc_in_pelvis_frame[1, 1:, frame]
    z = Jc_in_pelvis_frame[2, 1:, frame]
    sc._offsets3d = (x, y, z)
    return sc,
ani = FuncAnimation(fig, update, frames=range(Xsens_position.shape[2]), init_func=init, blit=False)
plt.show()
