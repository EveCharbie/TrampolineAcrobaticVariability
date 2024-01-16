import pickle
import ezc3d
import os
import scipy.io
import biorbd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_q(Xsens_orientation_per_move):
    """
    This function returns de generalized coordinates in the sequence XYZ (biorbd) from the quaternion of the orientation
    of the Xsens segments.
    The translation is left empty as it has to be computed otherwise.
    I am not sure if I would use this for kinematics analysis, but for visualisation it is not that bad.
    """

    parent_idx_list = {"Pelvis": None,  # 0
                       "L5": [0, "Pelvis"],  # 1
                       "L3": [1, "L5"],  # 2
                       "T12": [2, "L3"],  # 3
                       "T8": [3, "T12"],  # 4
                       "Neck": [4, "T8"],  # 5
                       "Head": [5, "Neck"],  # 6
                       "ShoulderR": [4, "T8"],  # 7
                       "UpperArmR": [7, "ShoulderR"],  # 8
                       "LowerArmR": [8, "UpperArmR"],  # 9
                       "HandR": [9, "LowerArmR"],  # 10
                       "ShoulderL": [4, "T8"],  # 11
                       "UpperArmL": [11, "ShoulderR"],  # 12
                       "LowerArmL": [12, "UpperArmR"],  # 13
                       "HandL": [13, "LowerArmR"],  # 14
                       "UpperLegR": [0, "Pelvis"],  # 15
                       "LowerLegR": [15, "UpperLegR"],  # 16
                       "FootR": [16, "LowerLegR"],  # 17
                       "ToesR": [17, "FootR"],  # 18
                       "UpperLegL": [0, "Pelvis"],  # 19
                       "LowerLegL": [19, "UpperLegL"],  # 20
                       "FootL": [20, "LowerLegL"],  # 21
                       "ToesL": [21, "FootL"],  # 22
                       }

    nb_frames = Xsens_orientation_per_move.shape[0]
    Q = np.zeros((23*3, nb_frames))
    rotation_matrices = np.zeros((23, nb_frames, 3, 3))
    for i_segment, key in enumerate(parent_idx_list):
        for i_frame in range(nb_frames):
            Quat_normalized = Xsens_orientation_per_move[i_frame, i_segment*4: (i_segment+1)*4] / np.linalg.norm(
                Xsens_orientation_per_move[i_frame, i_segment*4: (i_segment+1)*4]
            )
            Quat = biorbd.Quaternion(Quat_normalized[0],
                                     Quat_normalized[1],
                                     Quat_normalized[2],
                                     Quat_normalized[3])

            RotMat_current = biorbd.Quaternion.toMatrix(Quat).to_array()
            z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi/2]), 'z').to_array()
            RotMat_current = z_rotation @ RotMat_current

            if parent_idx_list[key] is None:
                RotMat = np.eye(3)
            else:
                RotMat = rotation_matrices[parent_idx_list[key][0], i_frame, :, :]

            RotMat_between = np.linalg.inv(RotMat) @ RotMat_current
            RotMat_between = biorbd.Rotation(RotMat_between[0, 0], RotMat_between[0, 1], RotMat_between[0, 2],
                            RotMat_between[1, 0], RotMat_between[1, 1], RotMat_between[1, 2],
                            RotMat_between[2, 0], RotMat_between[2, 1], RotMat_between[2, 2])
            Q[i_segment*3:(i_segment+1)*3, i_frame] = biorbd.Rotation.toEulerAngles(RotMat_between, 'xyz').to_array()

            rotation_matrices[i_segment, i_frame, :, :] = RotMat_current
    return Q


class MyData:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __getitem__(self, key):
        # Trouver toutes les colonnes qui commencent par le préfixe 'key'
        matching_columns = [col for col in self.dataframe.columns if col.startswith(key)]

        if not matching_columns:
            raise KeyError(f"Variable {key} not found.")

        # Retourner un dictionnaire où chaque élément est une colonne trouvée
        return {col.split('_')[-1]: self.dataframe[col].tolist() for col in matching_columns}


file_path_mat = '/home/lim/Documents/StageMathieu/Data_propre/SaMi/Q/'
file_name_mat = 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat'

folder_path = "/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/Sa_821_822_2_MOD200.00_GenderF"

data_loaded = scipy.io.loadmat(file_path_mat+file_name_mat)


if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# print(data_loaded.keys())
data_info_reloaded = {key: str(type(data_loaded[key])) for key in data_loaded.keys() if not key.startswith('__')}
# print(data_info_reloaded)
q2_data = data_loaded['Q2']
# print(q2_data)

column_names = [
    "PelvisTranslation_X", "PelvisTranslation_Y", "PelvisTranslation_Z",
    "Pelvis_X", "Pelvis_Y", "Pelvis_Z",
    "Thorax_X", "Thorax_Y", "Thorax_Z",
    "Tete_X", "Tete_Y", "Tete_Z",
    "EpauleD_Y", "EpauleD_Z",
    "BrasD_X", "BrasD_Y", "BrasD_Z",
    "AvBrasD_X", "AvBrasD_Z",
    "MainD_X", "MainD_Y",
    "EpauleG_Y", "EpauleG_Z",
    "BrasG_X", "BrasG_Y", "BrasG_Z",
    "AvBrasG_X", "AvBrasG_Z",
    "MainG_X", "MainG_Y",
    "CuisseD_X", "CuisseD_Y", "CuisseD_Z",
    "JambeD_X",
    "PiedD_X", "PiedD_Z",
    "CuisseG_X", "CuisseG_Y", "CuisseG_Z",
    "JambeG_X",
    "PiedG_X", "PiedG_Z"
]

DataFrame_with_colname = pd.DataFrame(q2_data)
# Inversez les lignes et les colonnes
DataFrame_with_colname = DataFrame_with_colname.T
DataFrame_with_colname.columns = column_names

my_data = MyData(DataFrame_with_colname)

# print(my_data["PiedD"]["X"])

selected_data = my_data.dataframe.iloc[3299:3591]

# Identifier les groupes de membres
member_groups = set([name.split('_')[0] for name in column_names])

for group in member_groups:
    # Filtrer les colonnes pour le groupe actuel, mais exclure les translations si nécessaire
    if group == "Pelvis":
        group_columns = [col for col in column_names if col.startswith(group) and not col.startswith("PelvisTranslation")]
    else:
        group_columns = [col for col in column_names if col.startswith(group)]

    group_data = selected_data[group_columns]

    plt.figure(figsize=(10, 6))
    for col in group_columns:
        plt.plot(group_data[col], label=col)

    plt.title(f"Graphique pour {group}")
    plt.xlabel("Index")
    plt.ylabel("Valeur")
    plt.legend()

    # Nom du fichier image
    file_name = f"{group}_graph.png"
    file_path = os.path.join(folder_path, file_name)

    # Enregistrer le graphique dans le dossier spécifié
    #plt.savefig(file_path)
    plt.close()  # Fermer le graphique après l'avoir enregistré

