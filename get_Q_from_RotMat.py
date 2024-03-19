import biorbd
import numpy as np
import os
import bioviz
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_build_model import (
    get_all_matrice,
    average_rotation_matrix,
    calculer_rotation_relative
)
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import parent_list_marker, check_matrix_orthogonality

home_path = "/home/lim/Documents/StageMathieu/DataTrampo/"
is_y_up = False

csv_path = f"{home_path}Labelling_trampo.csv"
interval_name_tab = pd.read_csv(csv_path, sep=';', usecols=['Participant', 'Analyse', 'Essai', 'Debut', 'Fin', 'Durée'])
valide = ['O']
interval_name_tab = interval_name_tab[interval_name_tab["Analyse"] == 'O']
interval_name_tab.loc[:, 'Essai'] = interval_name_tab['Essai'] + '.c3d'

# Obtenir la liste des participants
participant_names = interval_name_tab['Participant'].unique()

for name in participant_names:
    essai_by_name = interval_name_tab[interval_name_tab["Participant"] == name].copy()
    essai_by_name.loc[:, 'Interval'] = essai_by_name.apply(lambda row: (row['Debut'], row['Fin']), axis=1)
    folder_path = f"{home_path}{name}/Q/"
    model_path = f"{home_path}{name}/{name}.s2mMod"
    model = biorbd.Model(model_path)

    file_path_c3d = f"{home_path}{name}/Tests/"
    file_intervals = []

    for index, row in essai_by_name.iterrows():
        c3d_file = row['Essai']
        interval = row['Interval']
        file_path_complet = f"{file_path_c3d}{c3d_file}"
        file_intervals.append((file_path_complet, interval))

#
    model_path = f"{home_path}{name}/New{name}Model.s2mMod"
    model_kalman = f"{home_path}{name}/{name}.s2mMod"

    model_kalman = biorbd.Model(model_kalman)

    file_path_c3d = f"{home_path}{name}/Tests/"
    file_path_relax = f"{home_path}{name}/Score/"

    folder_path = f"{home_path}{name}/Q/"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    select_dof = "FullDof" in model_path

    relax_intervals = [(file_path_relax + "Relax.c3d", (0, 50))]

    file_path_relax, interval_relax = relax_intervals[0]
    rot_mat_relax, relax_articular_joint_center, pos_relax = get_all_matrice(file_path_relax, interval_relax, model_kalman, is_y_up)

    for file_path, interval in file_intervals:
        movement_matrix, articular_joint_center, pos_mov = get_all_matrice(file_path, interval, model_kalman, is_y_up)
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

        for i_frame in range(nb_frames):
            RotMat_between_total = []
            euler_sequences = {}
            for i_segment in range(nb_mat):
                RotMat = relax_matrix[i_segment, :, :]
                RotMat_current = movement_matrix[i_segment, i_frame, :, :]
                check_matrix_orthogonality(RotMat, i_segment, "RotMat")
                check_matrix_orthogonality(RotMat_current, i_segment, "RotMat_current")

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

                if select_dof is True:
                    euler_sequence = "xyz"
                    euler_sequences[i_segment] = euler_sequence
                    Q[i_segment * 3: (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
                            RotMat_between, euler_sequence).to_array()
                else:
                    if i_segment in (3, 6):
                        euler_sequence = "zyx"
                    elif i_segment in (10, 13):
                        euler_sequence = "x"
                    else:
                        euler_sequence = "xyz"

                    euler_sequences[i_segment] = euler_sequence

                    if euler_sequence == "x":  # Special handling for single axis
                        Q[i_segment * 3: (i_segment + 1) * 3 - 2, i_frame] = biorbd.Rotation.toEulerAngles(
                            RotMat_between, euler_sequence).to_array()
                    else:  # General case for three axes
                        Q[i_segment * 3: (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
                            RotMat_between, euler_sequence).to_array()

        Q_corrected = np.unwrap(Q, axis=1)

        # Ajouter ou soustraire 2 pi si necessaire
        for i in range(3, Q_corrected.shape[0]):
            subtract_pi = False
            add_pi = False
            for j in range(Q_corrected.shape[1]):
                if Q_corrected[i, j] > 2 * np.pi:
                    subtract_pi = True
                    break
                if Q_corrected[i, j] < -2 * np.pi:
                    add_pi = True
                    break
            if subtract_pi:
                Q_corrected[i] = Q_corrected[i] - 2 * np.pi
            if add_pi:
                Q_corrected[i] = Q_corrected[i] + 2 * np.pi

        Q_degrees = np.degrees(Q_corrected)

        # Ajout pelvis trans
        Q_complet = np.concatenate((pelv_trans_list.T, Q_corrected), axis=0)
        euler_sequences_complet = {key+1: value for key, value in euler_sequences.items()}
        euler_sequences_complet[0] = 'xyz'

        names = ["PelvisTranslation", "PelvisRotation", "Thorax", "Head", "RightShoulder",
                 "RightElbow", "RightWrist", "LeftShoulder", "LeftElbow", "LeftWrist",
                 "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee", "LeftAnkle"]
        # Update the dictionary to include names
        named_euler_sequences = [(names[key], euler_sequences_complet[key]) for key in sorted(euler_sequences_complet)]

        # Suppression des colonnes où tous les éléments sont zéro
        ligne_a_supprimer = np.all(Q_complet == 0, axis=1)
        Q_ready_to_use = Q_complet[~ligne_a_supprimer, :]

        # Création d'un dictionnaire pour le stockage
        mat_data = {
            "Q_ready_to_use": Q_ready_to_use,
            "Q_complet": Q_complet,
            "Q_original": Q,
            "Euler_Sequence": named_euler_sequences
        }
        folder_and_file_name_path = folder_path + f"{file_name}.mat"
        # Enregistrement dans un fichier .mat
        scipy.io.savemat(folder_and_file_name_path, mat_data)

        # axis_colors = {'X': 'blue', 'Y': 'green', 'Z': 'red'}
        # rows = (nb_mat + 1) // 4 + int((nb_mat + 1) % 4 > 0)
        # plt.figure(figsize=(25, 4 * rows))
        # for i in range(nb_mat + 1):
        #     plt.subplot(rows, 4, i + 1)
        #     segment_name, euler_sequence = named_euler_sequences[i]
        #     axis_labels = list(euler_sequence.upper())
        #     for axis, axis_label in enumerate(axis_labels):
        #         plt.plot(Q_complet[i * 3 + axis, :], label=axis_label, color=axis_colors[axis_label])
        #     plt.title(f'Segment {i + 1}: {segment_name}')
        #     plt.legend()
        # plt.tight_layout()
        # plt.show()
        #
        # model = biorbd.Model(model_path)
        # b = bioviz.Viz(loaded_model=model)
        # b.load_movement(Q_ready_to_use)
        # b.load_experimental_markers(pos_mov[:, :, :])
        # b.exec()


    # from pyorerun import BiorbdModelNoMesh, PhaseRerun
    #
    # nb_frames = Q.shape[1]
    # nb_seconds = 10
    # t_span = np.linspace(0, nb_seconds, nb_frames)
    # # loading biorbd model
    # biorbd_model = BiorbdModelNoMesh(model_path)
    #
    # # running the animation
    # rerun_biorbd = PhaseRerun(t_span)
    # # rerun_biorbd.add_animated_model(biorbd_model, Q_ready_to_use)
    # rerun_biorbd.rerun("yoyo")
