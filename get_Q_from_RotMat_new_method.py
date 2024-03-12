import biorbd
import numpy as np
import os
import bioviz
import scipy
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_build_model import get_all_matrice, average_rotation_matrix, calculer_rotation_relative
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import parent_list_marker, check_matrix_orthogonality
# from pyorerun import BiorbdModel, PhaseRerun
# import rerun as rr
# import pyorerun as prr

model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod")
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Q/"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_intervals = [
    (file_path_c3d + "Sa_bras_volant_1.c3d", (3349, 3950)),
    (file_path_c3d + "Sa_821_seul_2.c3d", (3431, 3736)),
    (file_path_c3d + "Sa_831_831_6.c3d", (4710, 5009)),

]

relax_intervals = [(file_path_c3d + "Relax.c3d", (0, 50))]

file_path_relax, interval_relax = relax_intervals[0]
rot_mat_relax, relax_articular_joint_center, pos_relax = get_all_matrice(file_path_relax, interval_relax, model)

for file_path, interval in file_intervals:
    movement_matrix, articular_joint_center, pos_mov = get_all_matrice(file_path, interval, model)
    pelv_trans_list = articular_joint_center[0]

    file_name, _ = os.path.splitext(os.path.basename(file_path))

    nb_frames = movement_matrix.shape[1]
    nb_mat = movement_matrix.shape[0]
    Q = np.zeros((nb_mat * 3, nb_frames))

    # Calcul de la matrice de rotation moyenne pour chaque articulation
    relax_matrix = np.zeros((nb_mat, 3, 3))
    for i in range(nb_mat):
        matrices = rot_mat_relax[i]
        relax_matrix[i] = average_rotation_matrix(matrices)

    # movement_matrix = results_list[0]

    for i_frame in range(nb_frames):
        RotMat_between_total = []
        for i_segment in range(nb_mat):
            RotMat = relax_matrix[i_segment, :, :]
            RotMat_current = movement_matrix[i_segment, i_frame, :, :]
            check_matrix_orthogonality(RotMat, "RotMat")
            check_matrix_orthogonality(RotMat_current, "RotMat_current")

            index_to_key = {i: key for i, key in enumerate(parent_list_marker.keys())}
            key_for_given_index = index_to_key[i_segment]
            info_for_given_index = parent_list_marker[key_for_given_index]

            if info_for_given_index is not None:
                parent_index, parent_name = info_for_given_index

            RotMat_between_relax = calculer_rotation_relative(relax_matrix[parent_index], relax_matrix[i_segment]) \
                if info_for_given_index is not None else relax_matrix[i_segment]
            RotMat_between_mvt = calculer_rotation_relative(movement_matrix[parent_index][i_frame], movement_matrix[i_segment][i_frame]) \
                if info_for_given_index is not None else movement_matrix[i_segment][i_frame]

            RotMat_between = np.linalg.inv(RotMat_between_relax) @ RotMat_between_mvt
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
            RotMat_between_total.append(RotMat_between)

            if i_segment in (5, 8, 11, 14):
                Q[i_segment * 3: (i_segment + 1) * 3-1, i_frame] = biorbd.Rotation.toEulerAngles(
                    RotMat_between, "zy").to_array()
            elif i_segment in (10, 13):
                Q[i_segment * 3: (i_segment + 1) * 3-2, i_frame] = biorbd.Rotation.toEulerAngles(
                    RotMat_between, "x").to_array()
            else:
                Q[i_segment * 3: (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
                    RotMat_between, "xyz").to_array()

    Q_corrected = np.unwrap(Q, axis=1)

    # Ajouter ou soustraire 2 pi si necessaire
    for i in range(Q_corrected.shape[0]):
        subtract_pi = False
        add_pi = False
        for j in range(Q_corrected.shape[1]):
            if Q_corrected[i, j] > 5:
                subtract_pi = True
                break
            if Q_corrected[i, j] < -5:
                add_pi = True
                break
        if subtract_pi:
            Q_corrected[i] = Q_corrected[i] - 2 * np.pi
        if add_pi:
            Q_corrected[i] = Q_corrected[i] + 2 * np.pi

    Q_degrees = np.degrees(Q)
    Q_complet = np.concatenate((pelv_trans_list.T, Q_corrected), axis=0)

    # Suppression des colonnes où tous les éléments sont zéro
    ligne_a_supprimer = np.all(Q_complet == 0, axis=1)
    Q_ready_to_use = Q_complet[~ligne_a_supprimer, :]

    # Création d'un dictionnaire pour le stockage
    mat_data = {
        "Q_ready_to_use": Q_ready_to_use,
        "Q_complet": Q_complet,
        "Q_original": Q
    }
    folder_and_file_name_path = folder_path + f"{file_name}.mat"
    # Enregistrement dans un fichier .mat
    scipy.io.savemat(folder_and_file_name_path, mat_data)


# for i in range(nb_mat+1):
#     plt.figure(figsize=(5, 3))
#     for axis in range(3):
#         plt.plot(Q_complet[i*3+axis, :], label=f'{["X", "Y", "Z"][axis]}')
#     plt.title(f'Segment {i+1}')
#     plt.xlabel('Frame')
#     plt.ylabel('Angle (rad)')
#     plt.legend()
# plt.show()

    # chemin_fichier_modifie = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/NewSarahModel.s2mMod"
    # model = biorbd.Model(chemin_fichier_modifie)
    # b = bioviz.Viz(loaded_model=model)
    # b.load_movement(Q_ready_to_use)
    # b.load_experimental_markers(pos_mov[:, :, :])
    #
    # b.exec()




# from pyorerun import BiorbdModel, PhaseRerun
#
# nb_frames = Q.shape[1]
# nb_seconds = 10
# t_span = np.linspace(0, nb_seconds, nb_frames)
# # loading biorbd model
# biorbd_model = BiorbdModel(chemin_fichier_modifie)
#
# # running the animation
# rerun_biorbd = PhaseRerun(t_span)
# rerun_biorbd.add_animated_model(biorbd_model, Q)
#
# rerun_biorbd.rerun("yoyo")
