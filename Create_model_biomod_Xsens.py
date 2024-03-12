import biorbd
import numpy as np
import bioviz
import pickle
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_build_model import get_all_matrice, convert_to_local_frame
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import parent_list_marker

chemin_fichier_pkl = "/home/lim/disk/Eye-tracking/Results_831/SaMi/4-/31a5eaac_0_0-64_489__4-__0__eyetracking_metrics.pkl"

with open(chemin_fichier_pkl, "rb") as fichier_pkl:
    # Charger les données à partir du fichier ".pkl"
    eye_tracking_metrics = pickle.load(fichier_pkl)

expertise = eye_tracking_metrics["subject_expertise"]
subject_name = eye_tracking_metrics["subject_name"]

Xsens_global_JCS_positions_full = eye_tracking_metrics["Xsens_global_JCS_positions"]
Xsens_global_JCS_orientations_full = eye_tracking_metrics["Xsens_global_JCS_orientations"]

Xsens_global_JCS_positions_reshape = Xsens_global_JCS_positions_full.reshape(23, 3)
Xsens_global_JCS_orientations_reshape = Xsens_global_JCS_orientations_full.reshape(23, 4)

# Ne selectionner que les articulations necessaire
indices_a_supprimer = [1, 2, 3, 5, 7, 11, 18, 22]

# Calculer les indices à conserver pour le tableau NumPy
indices_total = range(Xsens_global_JCS_positions_reshape.shape[0])
indices_a_conserver = [i for i in indices_total if i not in indices_a_supprimer]

# Sélectionner les lignes à conserver dans le tableau
Xsens_global_JCS_positions = Xsens_global_JCS_positions_reshape[indices_a_conserver, :]
Xsens_global_JCS_orientations = Xsens_global_JCS_orientations_reshape[indices_a_conserver, :]

# Extraction des coordonnées x, y, z
x = Xsens_global_JCS_positions_reshape[:, 0]
y = Xsens_global_JCS_positions_reshape[:, 1]
z = Xsens_global_JCS_positions_reshape[:, 2]

# Création d'une figure et d'un axe 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Ajout des points à l'axe
ax.scatter(x, y, z)
# Ajout des étiquettes à chaque point
for i, (px, py, pz) in enumerate(zip(x, y, z)):
    ax.text(px, py, pz, f'{i}', color='blue')  # Remplacez '{i}' par toute autre chaîne que vous souhaitez utiliser comme étiquette

# Étiquetage des axes
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

# Affichage de la figure
plt.show()


nb_mat = Xsens_global_JCS_positions.shape[0]


for i_segment in range(nb_mat):
    z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi / 2]), "z").to_array()

    Quat_normalized_neutre = Xsens_global_JCS_orientations_modifie[0, i_segment * 4 : (i_segment + 1) * 4] / np.linalg.norm(
        Xsens_global_JCS_orientations_modifie[0, i_segment * 4 : (i_segment + 1) * 4]
    )
    Quat_neutre = biorbd.Quaternion(Quat_normalized_neutre[0], Quat_normalized_neutre[1], Quat_normalized_neutre[2], Quat_normalized_neutre[3])
    RotMat_neutre_segment = biorbd.Quaternion.toMatrix(Quat_neutre).to_array()
    RotMat_neutre_segment = z_rotation @ RotMat_neutre_segment
    RotMat_neutre.append(RotMat_neutre_segment)



matrix_in_parent_frame = []
joint_center_in_parent_frame = []
rot_trans_matrix = []

for index, (joint, parent_info) in enumerate(parent_list_marker.items()):
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


chemin_fichier_original = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/SarahModelTest.s2mMod"
chemin_fichier_modifie = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/NewSarahModel.s2mMod"

model = biorbd.Model(chemin_fichier_original)
desired_order = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]

with open(chemin_fichier_original, 'r') as fichier:
    lignes = fichier.readlines()


with open(chemin_fichier_modifie, 'w') as fichier_modifie:
    a = 0
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
#

#
# informations_marqueurs = []
# nouvelles_lignes = []  # Liste pour stocker les nouvelles lignes avec les positions mises à jour
# # Parcourir les lignes du fichier pour extraire les informations
# i = 0  # Index pour parcourir les lignes
# while i < len(lignes):
#     if lignes[i].strip().startswith("marker"):  # Vérifie si la ligne commence par "marker"
#         nouvelles_lignes.append(lignes[i])
#         nouvelles_lignes.append(lignes[i+1])
#         nom_marqueur = lignes[i].split()[1]  # Extrait le nom du marqueur
#         nom_parent = lignes[i+1].split()[1]  # Extrait le nom du parent à la ligne suivante
#         id_parent = trouver_index_parent(nom_parent)
#         mat_parent_marker = relax_matrix[id_parent]
#         pos_parent_marker = relax_joint_center[id_parent]
#         index_marker = find_index(nom_marqueur, desired_order)
#         marker_global_pos = pos_marker_relax[0][:, index_marker, :]
#         mean_marker_global_pos = np.mean(marker_global_pos, axis=1)
#
#         marker_local_pos = convert_marker_to_local_frame(pos_parent_marker, mat_parent_marker, mean_marker_global_pos)
#
#         pos_str = "\t\t" + "position" + "\t\t" + "\t".join(f"{coord:.6f}" for coord in marker_local_pos) + "\n"
#         print(pos_str)
#
#         # Ajoute la nouvelle position à la liste des nouvelles lignes
#         nouvelles_lignes.append(pos_str)
#
#         # Incrémente i pour passer les lignes déjà traitées, en supposant que 'position' est la ligne i+2
#         i += 3
#     else:
#         # Ajoute les lignes non relatives aux markers directement à nouvelles_lignes
#         nouvelles_lignes.append(lignes[i])
#         i += 1
#
#
# with open(chemin_fichier_modifie, 'w') as fichier_modifie:
#     fichier_modifie.writelines(nouvelles_lignes)

#

model = biorbd.Model(chemin_fichier_modifie)
b = bioviz.Viz(loaded_model=model)
b.exec()

goodmodel = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod"
model = biorbd.Model(goodmodel)
b = bioviz.Viz(loaded_model=model)
b.exec()
