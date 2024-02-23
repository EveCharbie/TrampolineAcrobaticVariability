import numpy as np
import matplotlib.pyplot as plt
import os
import biorbd
import ezc3d
import bioviz
import pandas as pd

from Function_Class_Graph import (find_index, calculate_rmsd, get_orientation_knee_left, get_orientation_knee_right,
                                  dessiner_vecteurs, predictive_hip_joint_center_location)
from matplotlib.animation import FuncAnimation


def recons_kalman_v2(n_frames, num_markers, markers_xsens, model, initial_guess):
    # Préparation comme avant
    markersOverFrames = []
    for i in range(n_frames):
        node_segment = []
        for j in range(num_markers):
            node_segment.append(biorbd.NodeSegment(markers_xsens[:, j, i].T))
        markersOverFrames.append(node_segment)

    freq = 200
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)
    kalman.setInitState(initial_guess[0], initial_guess[1], initial_guess[2])

    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)
    q_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    qdot_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    markers_recons = np.ndarray(
        (3, num_markers, len(markersOverFrames)))  # Pour stocker les positions reconstruites des marqueurs

    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()

        # Nouveau : Calculer et stocker les positions reconstruites des marqueurs pour ce cadre
        markers_reconstructed = model.markers(Q)
        for m, marker_recons in enumerate(markers_reconstructed):
            markers_recons[:, m, i] = marker_recons.to_array()

    return q_recons, qdot_recons, markers_recons


model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod")
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/"

file_intervals = [
    (file_path_c3d + "Sa_821_contact_1.c3d", (3029, 3325)),
]

##
for file_path, interval in file_intervals:
    file_name = os.path.basename(file_path).split(".")[0]
    print(f"{file_name} is running")
    c = ezc3d.c3d(file_path)
    point_data = c["data"]["points"][:, :, interval[0] : interval[1]]
    n_markers = point_data.shape[1]
    nf_mocap = point_data.shape[2]
    f_mocap = c["parameters"]["POINT"]["RATE"]["value"][0]
    point_labels = c["parameters"]["POINT"]["LABELS"]
    # Extraire les noms de marqueurs utiles de 'point_labels'
    useful_labels = [label for label in point_labels["value"] if not label.startswith("*")]

    sample_label = useful_labels[0]
    typical_dimensions = point_data[0][find_index(sample_label, point_labels["value"])].shape[0]

    desired_order = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]

    n_markers_desired = len(desired_order)
    reordered_point_data = np.full(
        (4, n_markers_desired, typical_dimensions), np.nan
    )

    for i, marker in enumerate(desired_order):
        marker_found = False
        for label in useful_labels:
            if marker in label:  # Vérifie si marker est une sous-chaîne de label.
                original_index = find_index(label, point_labels["value"])
                reordered_point_data[:, i, :] = point_data[:, original_index, :]
                marker_found = True
                break

        if not marker_found:
            print(f"Le marqueur '{marker}' n'a pas été trouvé et a été initialisé avec NaN.")
            pass

    n_markers_reordered = reordered_point_data.shape[1]

    # Ne prendre que les 3 premiere colonnes et divise par 1000
    markers = np.zeros((3, n_markers_reordered, nf_mocap))
    for i in range(nf_mocap):
        for j in range(n_markers_reordered):
            markers[:, j, i] = reordered_point_data[:3, j, i]
            # markers[:, j, nf_mocap-1-i] = reordered_point_data[:3, j, i]
    markers = markers / 1000

    frame_index = 0
    # frame_index = nf_mocap-1
    start_frame = markers[:, :, frame_index: frame_index + 1]
    if start_frame.shape != (3, n_markers_reordered, 1):
        raise ValueError(
            f"Dimension incorrecte pour 'specific_frame'. Attendu: (3, {n_markers_reordered}, 1), Obtenu: "
            f"{start_frame.shape}")

    ik = biorbd.InverseKinematics(model, start_frame)
    ik.solve("only_lm")
    Q = ik.q
    # Assurez-vous que Q est un vecteur 1D
    Q_1d = Q.flatten() if Q.ndim > 1 else Q

    # Initialiser Qdot et Qddot en tant que vecteurs 1D
    Qdot_1d = np.zeros(Q_1d.shape)
    Qddot_1d = np.zeros(Q_1d.shape)

    # Créer initial_guess avec ces vecteurs 1D
    initial_guess = (Q_1d, Qdot_1d, Qddot_1d)

    q_recons, qdot_recons, pos_recons = recons_kalman_v2(nf_mocap, n_markers_reordered, markers, model, initial_guess)
    # b = bioviz.Viz(loaded_model=model)
    # b.load_movement(q_recons)
    # b.load_experimental_markers(markers[:, :, :])
    # b.exec()

    rmsd_by_frame = calculate_rmsd(markers, pos_recons)

    matrices_rotation_left, mid_cond_left = get_orientation_knee_left(pos_recons, desired_order)
    matrices_rotation_right, mid_cond_right = get_orientation_knee_right(pos_recons, desired_order)
    hjc, pelvic_origin, matrices_rotation_pelvic = predictive_hip_joint_center_location(pos_recons, desired_order)

    # Création de la figure et de l'axe 3D
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Limites de l'axe pour une bonne visualisation
    # Ajustez ces limites en fonction de vos données spécifiques
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])


    # Initialisation de l'animation en nettoyant les axes
    def init():
        ax.clear()
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])

    # Mise à jour de l'animation pour chaque frame
    def update(frame):
        ax.clear()
        # Origines pour les genoux gauche et droit
        origin_left = mid_cond_left[frame]
        origin_right = mid_cond_right[frame]
        origin_pelvic = pelvic_origin[frame]
        origin_hjc = hjc[frame]

        dessiner_vecteurs(ax, origin_left, matrices_rotation_left[frame][:, 0], matrices_rotation_left[frame][:, 1],
                          matrices_rotation_left[frame][:, 2])

        dessiner_vecteurs(ax, origin_right, matrices_rotation_right[frame][:, 0], matrices_rotation_right[frame][:, 1],
                          matrices_rotation_right[frame][:, 2])

        dessiner_vecteurs(ax, origin_pelvic, matrices_rotation_pelvic[frame][:, 0], matrices_rotation_pelvic[frame][:, 1],
                          matrices_rotation_pelvic[frame][:, 2])

        dessiner_vecteurs(ax, origin_hjc, matrices_rotation_pelvic[frame][:, 0], matrices_rotation_pelvic[frame][:, 1],
                          matrices_rotation_pelvic[frame][:, 2])

        # Affichage des points pour tous les marqueurs avec une couleur fixe, par exemple bleu ('b')
        for m in range(pos_recons.shape[1]):
            x, y, z = pos_recons[:, m, frame]
            ax.scatter(x, y, z, s=10, c='b')  # Utiliser 'c' pour spécifier la couleur

        # Ajouter une légende seulement pour le premier frame pour éviter les répétitions
        if frame == 0:
            ax.legend()

        # Création de l'animation
    ani = FuncAnimation(fig, update, frames=range(pos_recons.shape[2]), init_func=init, blit=False)

    plt.show()

    # markers_by_segment = {}
    # for i in range(model.nbMarkers()):
    #     marker_name = model.markerNames()[i].to_string()
    #     marker_indice = model.marker(i)
    #     parent_index = marker_indice.parentId()
    #     # Vérifier que l'indice du parent est valide
    #     if 0 <= parent_index < model.nbSegment():
    #         segment_name = model.segment(parent_index).name().to_string()
    #         if segment_name not in markers_by_segment:
    #             markers_by_segment[segment_name] = []
    #         markers_by_segment[segment_name].append(marker_name)
    #     else:
    #         print(f"Indice de segment invalide pour le marqueur {marker_name}: {parent_index}")
    # # Afficher les marqueurs regroupés par segment
    # for segment, markers in markers_by_segment.items():
    #     print(f"Segment: {segment}, Marqueurs: {markers}")

