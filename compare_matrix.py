import biorbd
import numpy as np
import bioviz
from TrampolineAcrobaticVariability.Function.Function_build_model import get_all_matrice

model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod")
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/"

file_intervals = [
    (file_path_c3d + "Sa_821_822_2.c3d", (3289, 3596)),
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
        Q[i_segment * 3: (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
            RotMat_between, "xyz"
        ).to_array()

Q_corrected = np.unwrap(Q, axis=1)
Q_degrees = np.degrees(Q_corrected)
Q_complet = np.concatenate((pelv_trans_list[0].T, Q), axis=0)

# Indices des lignes à supprimer
indices_a_supprimer = [16, 20, 25, 28, 34, 35, 37, 43, 44, 46]

# Suppression des lignes
# Nous utilisons une compréhension de liste pour créer une liste d'indices à conserver,
# puis nous indexons l'array original avec cette liste.
indices_a_conserver = [i for i in range(Q_complet.shape[0]) if i not in indices_a_supprimer]
Q_good_DoF = Q_complet[indices_a_conserver]


# for i in range(nb_mat):
#     plt.figure(figsize=(5, 3))
#     for axis in range(3):
#         plt.plot(Q_corrected[i*3+axis, :], label=f'{["X", "Y", "Z"][axis]}')
#     plt.title(f'Segment {i+1}')
#     plt.xlabel('Frame')
#     plt.ylabel('Angle (rad)')
#     plt.legend()
#     plt.show()

chemin_fichier_modifie = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/NewSarahModel.s2mMod"
model = biorbd.Model(chemin_fichier_modifie)
b = bioviz.Viz(loaded_model=model)
b.load_movement(Q_good_DoF)
# b.load_experimental_markers(pos_mov[:, :, :])


b.exec()
