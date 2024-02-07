import pickle
import ezc3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import biorbd
import os

dossier_graphiques = "/Users/mathieubourgeois/Desktop/Dossier_Graphique"


filepath = "/Users/mathieubourgeois/Documents/GitHub/Stage2024/2019-08-29/JeCh/tests/Je_833_1.c3d"
# filepath = "/Users/mathieubourgeois/Library/Mobile Documents/com~apple~CloudDocs/Desktop/MASTER/M1/STAGE_M1_TRAITEMENT/Analyse/C3DFiles/Sujet_01/Sujet_01_velo_60.c3d"


chemin_fichier_pkl = "/Users/mathieubourgeois/Desktop/PickleFiles/a62d4691_0_0-45_796__41__0__eyetracking_metrics.pkl"
chemin_fichier_pkl = "/Users/mathieubourgeois/Desktop/PickleFiles/a62d4691_0_0-45_796__41__1__eyetracking_metrics.pkl"
chemin_fichier_pkl = "/Users/mathieubourgeois/Desktop/PickleFiles/a62d4691_0_0-45_796__41__2__eyetracking_metrics.pkl"

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


def get_q(Xsens_orientation_per_move):
    """
    This function returns de generalized coordinates in the sequence XYZ (biorbd) from the quaternion of the orientation
    of the Xsens segments.
    The translation is left empty as it has to be computed otherwise.
    I am not sure if I would use this for kinematics analysis, but for visualisation it is not that bad.
    """

    parent_idx_list = {
        "Pelvis": None,  # 0
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
    Q = np.zeros((23 * 3, nb_frames))
    rotation_matrices = np.zeros((23, nb_frames, 3, 3))
    for i_segment, key in enumerate(parent_idx_list):
        for i_frame in range(nb_frames):
            Quat_normalized = Xsens_orientation_per_move[i_frame, i_segment * 4 : (i_segment + 1) * 4] / np.linalg.norm(
                Xsens_orientation_per_move[i_frame, i_segment * 4 : (i_segment + 1) * 4]
            )
            Quat = biorbd.Quaternion(Quat_normalized[0], Quat_normalized[1], Quat_normalized[2], Quat_normalized[3])

            RotMat_current = biorbd.Quaternion.toMatrix(Quat).to_array()
            z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi / 2]), "z").to_array()
            RotMat_current = z_rotation @ RotMat_current

            if parent_idx_list[key] is None:
                RotMat = np.eye(3)
            else:
                RotMat = rotation_matrices[parent_idx_list[key][0], i_frame, :, :]

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
            Q[i_segment * 3 : (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "xyz"
            ).to_array()

            rotation_matrices[i_segment, i_frame, :, :] = RotMat_current
    return Q


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


# Créer le dossier s'il n'existe pas
if not os.path.exists(dossier_graphiques):
    os.makedirs(dossier_graphiques)

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
    nom_fichier = os.path.join(dossier_graphiques, f"{member}_orientation_degrees.png")
    plt.savefig(nom_fichier)

    # Affichez le graphique
    # plt.show()
