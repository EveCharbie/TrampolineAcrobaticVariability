import pickle
import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import biorbd
import os
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import get_q

dossier_graphiques = "/Users/mathieubourgeois/Desktop/Dossier_Graphique"

biorbd_model_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod"
# chemin_fichier_pkl = "/Users/mathieubourgeois/Desktop/PickleFiles/a62d4691_0_0-45_796__41__0__eyetracking_metrics.pkl"
chemin_fichier_pkl = "/home/lim/disk/Eye-tracking/Results/AnBe/42/55d81c96_0_0-49_552__42__0__eyetracking_metrics.pkl"

with open(chemin_fichier_pkl, "rb") as fichier_pkl:
    # Charger les données à partir du fichier ".pkl"
    eye_tracking_metrics = pickle.load(fichier_pkl)

expertise = eye_tracking_metrics["subject_expertise"]
subject_name = eye_tracking_metrics["subject_name"]
move_orientation = eye_tracking_metrics["move_orientation"]
gaze_position_temporal_evolution_projected = eye_tracking_metrics["gaze_position_temporal_evolution_projected"]
gaze_position_temporal_evolution_projected_facing_front_wall = eye_tracking_metrics[
    "gaze_position_temporal_evolution_projected_facing_front_wall"
]
wall_index = eye_tracking_metrics["wall_index"]
wall_index_facing_front_wall = eye_tracking_metrics["wall_index_facing_front_wall"]
fixation_index = eye_tracking_metrics["fixation_index"]
anticipatory_index = eye_tracking_metrics["anticipatory_index"]
compensatory_index = eye_tracking_metrics["compensatory_index"]
spotting_index = eye_tracking_metrics["spotting_index"]
movement_detection_index = eye_tracking_metrics["movement_detection_index"]
blinks_index = eye_tracking_metrics["blinks_index"]
fixation_positions = eye_tracking_metrics["fixation_positions"]
Xsens_head_position_calculated = eye_tracking_metrics["Xsens_head_position_calculated"]
eye_position = eye_tracking_metrics["eye_position"]
gaze_orientation = eye_tracking_metrics["gaze_orientation"]
EulAngles_head_global = eye_tracking_metrics["EulAngles_head_global"]
EulAngles_neck = eye_tracking_metrics["EulAngles_neck"]
eye_angles = eye_tracking_metrics["eye_angles"]
Xsens_orthogonal_thorax_position = eye_tracking_metrics["Xsens_orthogonal_thorax_position"]
Xsens_orthogonal_head_position = eye_tracking_metrics["Xsens_orthogonal_head_position"]
Xsens_position_no_level_CoM_corrected_rotated_per_move = eye_tracking_metrics[
    "Xsens_position_no_level_CoM_corrected_rotated_per_move"
]
Xsens_jointAngle_per_move = eye_tracking_metrics["Xsens_jointAngle_per_move"]
Xsens_orientation_per_move = eye_tracking_metrics["Xsens_orientation_per_move"]
Xsens_CoM_per_move = eye_tracking_metrics["Xsens_CoM_per_move"]
time_vector_pupil_per_move = eye_tracking_metrics["time_vector_pupil_per_move"]

print(subject_name)
# Créer le graphique
# plt.figure(figsize=(8, 6))
# plt.plot(gaze_position_temporal_evolution_projected, marker='o', linestyle='-')
# plt.title('Évolution temporelle de la position du regard')
# plt.xlabel('Temps')
# plt.ylabel('Position du regard')
# plt.grid(True)

# Afficher le graphique
# plt.show()


q = get_q(Xsens_orientation_per_move)
# print(q)

# Définir la liste des membres
parent_idx_list = {
    "Pelvis": None,
    "L5": [0, "Pelvis"],
    "L3": [1, "L5"],
    "T12": [2, "L3"],
    "T8": [3, "T12"],
    "Neck": [4, "T8"],
    "Head": [5, "Neck"],
    "ShoulderR": [4, "T8"],
    "UpperArmR": [7, "ShoulderR"],
    "LowerArmR": [8, "UpperArmR"],
    "HandR": [9, "LowerArmR"],
    "ShoulderL": [4, "T8"],
    "UpperArmL": [11, "ShoulderR"],
    "LowerArmL": [12, "UpperArmR"],
    "HandL": [13, "LowerArmR"],
    "UpperLegR": [0, "Pelvis"],
    "LowerLegR": [15, "UpperLegR"],
    "FootR": [16, "LowerLegR"],
    "ToesR": [17, "FootR"],
    "UpperLegL": [0, "Pelvis"],
    "LowerLegL": [19, "UpperLegL"],
    "FootL": [20, "LowerLegL"],
    "ToesL": [21, "FootL"],
}

# Supposer que 'q' contient les données que vous souhaitez tracer
nb_frames = q.shape[1]
time = np.arange(nb_frames)  # Créez une séquence de temps pour l'axe x

q = np.unwrap(q, axis=1)

# Créer le dossier s'il n'existe pas
# if not os.path.exists(dossier_graphiques):
#     os.makedirs(dossier_graphiques)

# Parcourez chaque membre de la liste et tracez un graphique
for member, parent_info in parent_idx_list.items():
    i_segment = parent_info[0] if parent_info is not None else 0
    start_idx = i_segment * 3
    end_idx = (i_segment + 1) * 3

    # Convertir les données en degrés
    q_degrees = np.degrees(q[start_idx:end_idx, :])

    # Créez un subplot pour chaque membre
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 1, 1)

    # Tracez les données converties en degrés
    plt.plot(time, q_degrees.T, label=["X", "Y", "Z"])
    plt.title(f"Orientation pour {member}")
    plt.xlabel("Frames")
    plt.ylabel("Orientation (degrés)")

    # Affichez une légende
    plt.legend(["X", "Y", "Z"])

    # Enregistrez le graphique dans le dossier spécifié
    # nom_fichier = os.path.join(dossier_graphiques, f"{member}_orientation_degrees.png")
    # plt.savefig(nom_fichier)

    # Affichez le graphique
    plt.show()
