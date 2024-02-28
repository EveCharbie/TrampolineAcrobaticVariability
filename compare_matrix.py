import biorbd
import numpy as np
import matplotlib.pyplot as plt
from Function_Class_Graph import get_all_matrice

model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod")
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/"

file_intervals = [
    (file_path_c3d + "Sa_821_seul_2.c3d", (3431, 3736)),
]

relax_intervals = [(file_path_c3d + "Relax.c3d", (0, 50))]

results_list = []
relax_list = []

for file_path, interval in file_intervals:
    rot_mat, articular_joint_center = get_all_matrice(file_path, interval, model)
    results_list.append(rot_mat)

for file_path, interval in relax_intervals:
    rot_mat_relax, relax_articular_joint_center = get_all_matrice(file_path, interval, model)
    relax_list.append(rot_mat_relax)

relax_matrix = np.mean(relax_list[0], axis=1)


nb_frames = results_list[0].shape[1]
nb_mat = results_list[0].shape[0]
Q = np.zeros((nb_mat * 3, nb_frames))

movement_mat = results_list[0]


for i_segment in range(nb_mat):
    for i_frame in range(nb_frames):
        RotMat = relax_matrix[i_segment, :, :]
        RotMat_current = movement_mat[i_segment, i_frame, :, :]

        RotMat_between = np.linalg.inv(RotMat) @ RotMat_current
        RotMat_between = biorbd.Rotation(
            RotMat_between[0, 0],
            RotMat_between[0, 1],
            RotMat_between[0, 2],
            RotMat_between[1, 0],
            RotMat_between[1, 1],
            RotMat_between[1, 2],
            RotMat_between[2, 0],
            RotMat_between[2, 1],
            RotMat_between[2, 2],
        )
        Q[i_segment * 3: (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
            RotMat_between, "xyz"
        ).to_array()

Q_corrected = np.unwrap(Q, axis=1)
Q_degrees = np.degrees(Q_corrected)

for i in range(nb_mat):
    plt.figure(figsize=(5, 3))
    for axis in range(3):
        plt.plot(Q_corrected[i*3+axis, :], label=f'{["X", "Y", "Z"][axis]}')
    plt.title(f'Segment {i+1}')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.show()
