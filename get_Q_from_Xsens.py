import pickle
import matplotlib.pyplot as plt
import biorbd
import numpy as np
import bioviz
from TrampolineAcrobaticVariability.Function.Function_build_model import (calculer_rotation_et_angle,
                                                                          calculer_rotation_relative)
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import parent_list_xsens

chemin_fichier_pkl = "/home/lim/disk/Eye-tracking/Results_831/SaMi/43/31a5eaac_0_0-64_489__43__0__eyetracking_metrics.pkl"
model_path = "/home/lim/Documents/StageMathieu/Xsens_Model.bioMod"

# select_dof = "FullDof" in model_path
select_dof = True
with open(chemin_fichier_pkl, "rb") as fichier_pkl:
    # Charger les données à partir du fichier ".pkl"
    eye_tracking_metrics = pickle.load(fichier_pkl)

expertise = eye_tracking_metrics["subject_expertise"]
subject_name = eye_tracking_metrics["subject_name"]

Xsens_jointAngle_per_move = eye_tracking_metrics["Xsens_jointAngle_per_move"]
Xsens_orientation_per_move = eye_tracking_metrics["Xsens_orientation_per_move"]

Xsens_global_JCS_positions = eye_tracking_metrics["Xsens_global_JCS_positions"]
Xsens_global_JCS_orientations = eye_tracking_metrics["Xsens_global_JCS_orientations"]

Xsens_global_JCS_positions_reshape = Xsens_global_JCS_positions.reshape(23, 3)

# Ne selectionner que les articulations necessaire
indices_a_supprimer = [1, 2, 3, 5, 7, 11, 18, 22]

indices_total = range(Xsens_global_JCS_positions_reshape.shape[0])
indices_a_conserver = [i for i in indices_total if i not in indices_a_supprimer]
Xsens_global_JCS_positions_complet = Xsens_global_JCS_positions_reshape[indices_a_conserver, :]

indices_reels_colonnes_a_supprimer = []
for indice in indices_a_supprimer:
    indices_reels_colonnes_a_supprimer.extend(range(indice * 4, indice * 4 + 4))
mask_colonnes = np.ones(Xsens_global_JCS_orientations.shape[1], dtype=bool)
mask_colonnes[indices_reels_colonnes_a_supprimer] = False

# Appliquer le masque pour conserver uniquement les colonnes désirées
Xsens_global_JCS_orientations_modifie = Xsens_global_JCS_orientations[:, mask_colonnes]
Xsens_orientation_per_move_modifie = Xsens_orientation_per_move[:, mask_colonnes]


nb_frames = Xsens_orientation_per_move_modifie.shape[0]
nb_mat = Xsens_orientation_per_move_modifie.shape[1]//4
Q = np.zeros((nb_mat * 3, nb_frames))
pelvis_trans = Xsens_jointAngle_per_move[:, :3]

z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi / 2]), "z").to_array()

for i_frame in range(nb_frames):
    RotMat_between_total = []
    RotMat_neutre = []
    RotMat_mov = []
    euler_sequences = {}

    # Calcul initial des matrices de rotation pour neutre et mouvement
    for i_segment in range(nb_mat):
        rot_neutre = calculer_rotation_et_angle(i_segment, Xsens_global_JCS_orientations_modifie[0, :], z_rotation)
        rot_mov = calculer_rotation_et_angle(i_segment, Xsens_orientation_per_move_modifie[i_frame, :], z_rotation)
        RotMat_neutre.append(rot_neutre)
        RotMat_mov.append(rot_mov)

    for i_segment in range(nb_mat):
        index_to_key = {i: key for i, key in enumerate(parent_list_xsens.keys())}
        key_for_given_index = index_to_key[i_segment]
        info_for_given_index = parent_list_xsens[key_for_given_index]

        if info_for_given_index is not None:
            parent_index, parent_name = info_for_given_index

        # RotMat_between_neutre = calculer_rotation_relative(RotMat_neutre[parent_index], RotMat_neutre[i_segment]) \
        #     if info_for_given_index is not None else RotMat_neutre[i_segment]
        # RotMat_between_mov = calculer_rotation_relative(RotMat_mov[parent_index], RotMat_mov[i_segment]) \
        #     if info_for_given_index is not None else RotMat_mov[i_segment]

        RotMat_between = np.linalg.inv(RotMat_mov[parent_index]) @ RotMat_mov[i_segment] \
            if info_for_given_index is not None else RotMat_mov[i_segment]
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

Q_complet = np.concatenate((pelvis_trans.T, Q_corrected), axis=0)

euler_sequences_complet = {key+1: value for key, value in euler_sequences.items()}
euler_sequences_complet[0] = 'xyz'

names = ["PelvisTranslation", "PelvisRotation", "Thorax", "Head", "RightShoulder",
         "RightElbow", "RightWrist", "LeftShoulder", "LeftElbow", "LeftWrist",
         "RightHip", "RightKnee", "RightAnkle", "LeftHip", "LeftKnee", "LeftAnkle"]
# Update the dictionary to include names
named_euler_sequences = [(names[key], euler_sequences_complet[key]) for key in sorted(euler_sequences_complet)]


# for i in range(nb_mat+1):
#     plt.figure(figsize=(5, 3))
#     for axis in range(3):
#         plt.plot(Q_complet[i*3+axis, :], label=f'{["X", "Y", "Z"][axis]}')
#     plt.title(f'Segment {i+1}')
#     plt.xlabel('Frame')
#     plt.ylabel('Angle (rad)')
#     plt.legend()
# plt.show()

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

# new_folder_file = file_name.rsplit('_', 1)[0]
#
# new_folder_path = f"{folder_path}{new_folder_file}/"
# folder_and_file_name_path = new_folder_path + f"{file_name}.mat"
#
# if not os.path.exists(new_folder_path):
#     os.makedirs(new_folder_path)
# # Enregistrement dans un fichier .mat
# scipy.io.savemat(folder_and_file_name_path, mat_data)

axis_colors = {'X': 'blue', 'Y': 'green', 'Z': 'red'}
rows = (nb_mat + 1) // 4 + int((nb_mat + 1) % 4 > 0)
plt.figure(figsize=(25, 4 * rows))
for i in range(nb_mat + 1):
    plt.subplot(rows, 4, i + 1)
    segment_name, euler_sequence = named_euler_sequences[i]
    axis_labels = list(euler_sequence.upper())
    for axis, axis_label in enumerate(axis_labels):
        plt.plot(Q_complet[i * 3 + axis, :], label=axis_label, color=axis_colors[axis_label])
    plt.title(f'Segment {i + 1}: {segment_name}')
    plt.legend()
plt.tight_layout()
plt.show()


model = biorbd.Model(model_path)
b = bioviz.Viz(loaded_model=model)
b.load_movement(Q_ready_to_use)

b.exec()

