import pickle
import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import biorbd
import os
import numpy as np
import bioviz
from TrampolineAcrobaticVariability.Function.Function_build_model import get_all_matrice, convert_to_local_frame
# from TrampolineAcrobaticVariability.Function.Function_Class_Basics import parent_list_xsens

parent_list_xsens = {
    "Pelvis": None,  # 0
    # "L5": [0, "Pelvis"],  # delete
    # "L3": [1, "L5"],  # delete
    # "T12": [2, "L3"],  # delete
    "T8": [0, "Pelvis"],  # 1
    # "Neck": [4, "T8"],  # delete
    "Head": [1, "T8"],  # 2
    # "ShoulderR": [4, "T8"],  # delete
    "UpperArmR": [1, "T8"],  # 3
    "LowerArmR": [3, "UpperArmR"],  # 4
    "HandR": [4, "LowerArmR"],  # 5
    # "ShoulderL": [4, "T8"],  # delete
    "UpperArmL": [1, "T8"],  # 6
    "LowerArmL": [6, "UpperArmR"],  # 7
    "HandL": [7, "LowerArmR"],  # 8
    "UpperLegR": [0, "Pelvis"],  # 9
    "LowerLegR": [9, "UpperLegR"],  # 10
    "FootR": [10, "LowerLegR"],  # 11
    # "ToesR": [17, "FootR"],  # delete
    "UpperLegL": [0, "Pelvis"],  # 12
    "LowerLegL": [12, "UpperLegL"],  # 13
    "FootL": [13, "LowerLegL"],  # 14
    # "ToesL": [21, "FootL"],  # delete
}

chemin_fichier_pkl = "/home/lim/disk/Eye-tracking/Results_831/SaMi/4-/31a5eaac_0_0-64_489__4-__0__eyetracking_metrics.pkl"

with open(chemin_fichier_pkl, "rb") as fichier_pkl:
    # Charger les données à partir du fichier ".pkl"
    eye_tracking_metrics = pickle.load(fichier_pkl)

expertise = eye_tracking_metrics["subject_expertise"]
subject_name = eye_tracking_metrics["subject_name"]

Xsens_jointAngle_per_move = eye_tracking_metrics["Xsens_jointAngle_per_move"]
Xsens_orientation_per_move = eye_tracking_metrics["Xsens_orientation_per_move"]

Xsens_global_JCS_positions = eye_tracking_metrics["Xsens_global_JCS_positions"]
Xsens_global_JCS_orientations = eye_tracking_metrics["Xsens_global_JCS_orientations"]

Xsens_global_JCS_positions = Xsens_global_JCS_positions.reshape(23, 3)

# Ne selectionner que les articulations necessaire
indices_a_supprimer = [1, 2, 3, 5, 7, 11, 18, 22]

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
rotation_matrices = np.zeros((23, nb_frames, 3, 3))
pelvis_trans = Xsens_jointAngle_per_move[:, :3]

for i_frame in range(nb_frames):
    RotMat_between_total = []
    RotMat_neutre = []
    RotMat_mov = []

## Pour la position neutre
    for i_segment in range(nb_mat):
        z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi / 2]), "z").to_array()

        Quat_normalized_neutre = Xsens_global_JCS_orientations_modifie[0, i_segment * 4 : (i_segment + 1) * 4] / np.linalg.norm(
            Xsens_global_JCS_orientations_modifie[0, i_segment * 4 : (i_segment + 1) * 4]
        )
        Quat_neutre = biorbd.Quaternion(Quat_normalized_neutre[0], Quat_normalized_neutre[1], Quat_normalized_neutre[2], Quat_normalized_neutre[3])
        RotMat_neutre_segment = biorbd.Quaternion.toMatrix(Quat_neutre).to_array()
        RotMat_neutre_segment = z_rotation @ RotMat_neutre_segment
        RotMat_neutre.append(RotMat_neutre_segment)

        Quat_normalized_mov = Xsens_orientation_per_move_modifie[i_frame, i_segment * 4: (i_segment + 1) * 4] / np.linalg.norm(
            Xsens_orientation_per_move_modifie[i_frame, i_segment * 4: (i_segment + 1) * 4]
        )
        Quat_mov = biorbd.Quaternion(Quat_normalized_mov[0], Quat_normalized_mov[1], Quat_normalized_mov[2], Quat_normalized_mov[3])
        RotMat_mov_segment = biorbd.Quaternion.toMatrix(Quat_mov).to_array()
        RotMat_mov_segment = z_rotation @ RotMat_mov_segment
        RotMat_mov.append(RotMat_mov_segment)

    for i_segment in range(nb_mat):
        index_to_key = {i: key for i, key in enumerate(parent_list_xsens.keys())}
        key_for_given_index = index_to_key[i_segment]
        info_for_given_index = parent_list_xsens[key_for_given_index]

        if info_for_given_index is not None:
            parent_index, parent_name = info_for_given_index
            RotMat_between_neutre = np.linalg.inv(RotMat_neutre[parent_index]) @ RotMat_neutre[i_segment]
            RotMat_between_neutre = biorbd.Rotation(
                RotMat_between_neutre[0, 0],
                RotMat_between_neutre[0, 1],
                RotMat_between_neutre[0, 2],
                RotMat_between_neutre[1, 0],
                RotMat_between_neutre[1, 1],
                RotMat_between_neutre[1, 2],
                RotMat_between_neutre[2, 0],
                RotMat_between_neutre[2, 1],
                RotMat_between_neutre[2, 2],
            )
            RotMat_between_mov = np.linalg.inv(RotMat_mov[parent_index]) @ RotMat_mov[i_segment]
            RotMat_between_mov = biorbd.Rotation(
                RotMat_between_mov[0, 0],
                RotMat_between_mov[0, 1],
                RotMat_between_mov[0, 2],
                RotMat_between_mov[1, 0],
                RotMat_between_mov[1, 1],
                RotMat_between_mov[1, 2],
                RotMat_between_mov[2, 0],
                RotMat_between_mov[2, 1],
                RotMat_between_mov[2, 2],
            )
        else:
            RotMat_between_neutre = RotMat_neutre[i_segment]
            RotMat_between_mov = RotMat_mov[i_segment]

        if info_for_given_index is not None:
            RotMat_between_neutre = RotMat_between_neutre.to_array()
            RotMat_between_mov = RotMat_between_mov.to_array()

        RotMat_between = np.linalg.inv(RotMat_between_neutre) @ RotMat_between_mov
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
            Q[i_segment * 3: (i_segment + 1) * 3 - 1, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "zy").to_array()
        elif i_segment in (10, 13):
            Q[i_segment * 3: (i_segment + 1) * 3 - 2, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "x").to_array()
        else:
            Q[i_segment * 3: (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "xyz").to_array()

Q_corrected = np.unwrap(Q, axis=1)

for i in range(nb_mat):
    plt.figure(figsize=(5, 3))
    for axis in range(3):
        plt.plot(Q_corrected[i*3+axis, :], label=f'{["X", "Y", "Z"][axis]}')
    plt.title(f'Segment {i+1}')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
plt.show()

# Suppression des colonnes où tous les éléments sont zéro
ligne_a_supprimer = np.all(Q_corrected == 0, axis=1)
Q_complet_good_DOF = Q_corrected[~ligne_a_supprimer, :]

chemin_fichier_modifie = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/NewSarahModel.s2mMod"
model = biorbd.Model(chemin_fichier_modifie)
b = bioviz.Viz(loaded_model=model)
b.load_movement(Q_complet_good_DOF)

b.exec()