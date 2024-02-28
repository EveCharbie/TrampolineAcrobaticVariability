import biorbd
import numpy as np
import matplotlib.pyplot as plt
from Function_Class_Graph import get_all_matrice, convert_to_local_frame

model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod")
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/"

file_intervals = [
    (file_path_c3d + "Relax.c3d", (0, 50)),
    (file_path_c3d + "Sa_821_seul_2.c3d", (3431, 3736)),
]

joint_center_mov = []
rotate_matrice_mov = []

joint_center_relax = []
rotate_matrice_relax = []

for file_path, interval in file_intervals:
    rot_mat, articular_joint_center = get_all_matrice(file_path, interval, model)

    if "Relax" in file_path:
        joint_center_relax.append(articular_joint_center)
        rotate_matrice_relax.append(rot_mat)
    else:
        joint_center_mov.append(articular_joint_center)
        rotate_matrice_mov.append(rot_mat)


relax_matrix = np.mean(rotate_matrice_relax[0], axis=1)
relax_joint_center = np.mean(joint_center_relax[0], axis=1)

parent_list = {
    "Pelvis": None,  # 0
    "HipR": [0, "Pelvis"],  # 1
    "KneeR": [1, "HipR"],  # 2
    "AnkleR": [2, "KneeR"],  # 3
    "HipL": [0, "Pelvis"],  # 4
    "KneeL": [4, "HipL"],  # 5
    "AnkleL": [5, "KneeL"],  # 6
    "Thorax": [0, "Pelvis"],  # 7
    "Head": [7, "Thorax"],  # 8
    "UpperArmR": [7, "Thorax"],  # 9
    "LowerArmR": [9, "UpperArmR"],  # 10
    "HandR": [10, "LowerArmR"],  # 11
    "UpperArmL": [7, "Thorax"],  # 12
    "LowerArmL": [12, "UpperArmL"],  # 13
    "HandL": [13, "LowerArmL"],  # 14
}

matrix_in_parent_frame = []
joint_center_in_parent_frame = []
rot_trans_matrix = []

for index, (joint, parent_info) in enumerate(parent_list.items()):
    if parent_info is not None:  # Vérifie si parent_info n'est pas None
        parent_index, parent_name = parent_info
        P2_in_P1, R2_in_R1 = convert_to_local_frame(relax_joint_center[parent_index], relax_matrix[parent_index],
                                                    relax_joint_center[index], relax_matrix[index])
        matrix_in_parent_frame.append(R2_in_R1)
        joint_center_in_parent_frame.append(P2_in_P1)
        RT_mat = np.eye(4)
        RT_mat[:3, :3] = R2_in_R1
        RT_mat[:3, 3] = P2_in_P1
        rot_trans_matrix.append(RT_mat)

    else:
        matrix_in_parent_frame.append(relax_matrix[index])
        joint_center_in_parent_frame.append(relax_joint_center[index])
        RT_mat = np.eye(4)
        RT_mat[:3, :3] = relax_matrix[index]
        RT_mat[:3, 3] = [0.0, 0.0, 0.0]
        rot_trans_matrix.append(RT_mat)


chemin_fichier_original = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/SarahModelTest (copy).s2mMod"
chemin_fichier_modifie = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/test1.s2mMod"

with open(chemin_fichier_original, 'r') as fichier:
    lignes = fichier.readlines()

a = 0

with open(chemin_fichier_modifie, 'w') as fichier_modifie:
    i = 0
    while i < len(lignes):
        if "RT" in lignes[i]:
            fichier_modifie.write(lignes[i])  # Écrit la ligne RT
            i += 1  # Passe à la ligne suivante
            # Convertir chaque ligne de la matrice NumPy en chaîne avec des tabulations entre les éléments
            for ligne in rot_trans_matrix[a]:
                ligne_formattee = "\t".join(f"{val:.10f}" for val in ligne)
                fichier_modifie.write("\t\t\t" + ligne_formattee + "\n")
            i += 4  # Saute les 4 lignes de l'ancienne matrice
            a += 1
        else:
            fichier_modifie.write(lignes[i])
            i += 1
