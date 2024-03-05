import biorbd
import numpy as np
import bioviz
import matplotlib.pyplot as plt
from Function.Function_build_model import get_all_matrice

model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod")
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/"

file_intervals = [
    # (file_path_c3d + "Sa_bras_volant_1.c3d", (3349, 3450)),
    (file_path_c3d + "Sa_821_seul_2.c3d", (3431, 3736)),

]

relax_intervals = [(file_path_c3d + "Relax.c3d", (0, 50))]

results_list = []
relax_list = []
pelv_trans_list = []

for file_path, interval in file_intervals:
    rot_mat, articular_joint_center, pos_mov = get_all_matrice(file_path, interval, model)
    results_list.append(rot_mat)
    pelv_trans_list.append(articular_joint_center[0])

for file_path, interval in relax_intervals:
    rot_mat_relax, relax_articular_joint_center, pos_relax = get_all_matrice(file_path, interval, model)
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
        if i_segment in (3, 4, 6, 7):
            Q[i_segment * 3: (i_segment + 1) * 3-1, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "zy").to_array()
        elif i_segment in (10, 13):
            Q[i_segment * 3: (i_segment + 1) * 3-2, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "x").to_array()
        else:
            Q[i_segment * 3: (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "xyz").to_array()


Q_corrected = np.unwrap(Q, axis=1)
Q_degrees = np.degrees(Q_corrected)
Q_complet = np.concatenate((pelv_trans_list[0].T, Q_corrected), axis=0)

ligne_a_supprimer = np.all(Q_complet == 0, axis=1)

# Suppression des colonnes où tous les éléments sont zéro
Q_complet_good_DOF = Q_complet[~ligne_a_supprimer, :]


for i in range(nb_mat):
    plt.figure(figsize=(5, 3))
    for axis in range(3):
        plt.plot(Q_complet[i*3+axis, :], label=f'{["X", "Y", "Z"][axis]}')
    plt.title(f'Segment {i+1}')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.show()

chemin_fichier_modifie = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/NewSarahModel.s2mMod"
model = biorbd.Model(chemin_fichier_modifie)
b = bioviz.Viz(loaded_model=model)
b.load_movement(Q_complet_good_DOF)
b.load_experimental_markers(pos_mov[:, :, :])


b.exec()
