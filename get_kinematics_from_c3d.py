"""
The goal of this program is to reconstruct the kinematics of the motion capture data using a Kalman filter.
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import biorbd
import scipy
import ezc3d
import bioviz
from Function_Class_Graph import (column_names)
import pandas as pd


model = biorbd.Model('/home/lim/Documents/StageMathieu/Data_propre/SaMi/SaMi.bioMod')

# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = '/home/lim/Documents/StageMathieu/Data_propre/SaMi/Mvt_c3d/'

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/Data_propre/SaMi/QBis/"


# Liste des tuples (chemin du fichier, intervalle)
# file_intervals = [
#     (file_path_c3d + 'Sa_831_831_1.c3d', (3466, 3747)),
#     (file_path_c3d + 'Sa_831_831_3.c3d', (4138, 4427)),
#     (file_path_c3d + 'Sa_831_831_4.c3d', (3754, 4047)),
#     (file_path_c3d + 'Sa_831_831_5.c3d', (1632, 1928)),
#     (file_path_c3d + 'Sa_831_831_6.c3d', (4710, 5009)),
# ]


file_intervals = [
    (file_path_c3d + 'Sa_821_seul_1.c3d', (3357, 3665)),
    (file_path_c3d + 'Sa_821_seul_2.c3d', (3431, 3736)),
    (file_path_c3d + 'Sa_821_seul_3.c3d', (3209, 3520)),
    (file_path_c3d + 'Sa_821_seul_4.c3d', (3311, 3620)),
    (file_path_c3d + 'Sa_821_seul_5.c3d', (2696, 3000)),

]

def recons_kalman(n_frames, num_markers, markers_xsens, model,initial_guess):
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
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()
    return q_recons, qdot_recons


def find_index(name, list):
    return list.index(name)

for file_path, interval in file_intervals:
    file_name = os.path.basename(file_path).split('.')[0]
    print(f"{file_name} is running")
    c = ezc3d.c3d(file_path)
    point_data = c['data']['points'][:, :, interval[0]:interval[1]]
    n_markers = point_data.shape[1]
    nf_mocap = point_data.shape[2]
    f_mocap = c['parameters']['POINT']['RATE']['value'][0]
    point_labels = c['parameters']['POINT']['LABELS']
    # Extraire les noms de marqueurs utiles de 'point_labels'
    useful_labels = [label for label in point_labels['value'] if not label.startswith('*')]

    sample_label = useful_labels[0]
    typical_dimensions = point_data[0][find_index(sample_label, point_labels["value"])].shape[0]

    desired_order = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]

    ## Deuxieme methode pour remettre les marqueurs dans l'ordre
    n_markers_desired = len(desired_order)
    reordered_point_data = np.full((4, n_markers_desired, typical_dimensions), np.nan)  # 4 pour x, y, z, et la confidence

    # Remplir le tableau avec les données existantes ou NaN
    for i, marker in enumerate(desired_order):
        if marker in useful_labels:
            original_index = find_index(marker, point_labels["value"])
            reordered_point_data[:, i, :] = point_data[:, original_index, :]
        else:
        # Si le marqueur n'est pas trouvé, imprimer un message indiquant que le marqueur est manquant
            print(f"Le marqueur '{marker}' n'a pas été trouvé et a été initialisé avec NaN.")
    ##
    n_markers_reordered = reordered_point_data.shape[1]

    markers = np.zeros((3, n_markers_reordered, nf_mocap))
    for i in range(nf_mocap):
        for j in range(n_markers_reordered):
            markers[:, j, i] = reordered_point_data[:3, j, i]
    markers = markers / 1000

    frame_index = 0
    start_frame = markers[:, :, frame_index:frame_index + 1]

    if start_frame.shape != (3, n_markers_reordered, 1):
        raise ValueError(f"Dimension incorrecte pour 'specific_frame'. Attendu: (3, {n_markers_reordered}, 1), Obtenu: {start_frame.shape}")

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

    q_recons, qdot_recons = recons_kalman(nf_mocap, n_markers_reordered, markers, model, initial_guess)
    b = bioviz.Viz(loaded_model=model)
    b.load_movement(q_recons)
    b.load_experimental_markers(markers[:, :, :])
    b.exec()

    Q = pd.DataFrame(Q.transpose(), columns=column_names)
    Q_pred = q_recons[:, frame_index]
    error = Q_pred-Q
    # print(error.transpose())

    # Création d'un dictionnaire pour le stockage
    mat_data = {'Q2': q_recons}

    folder_and_file_name_path = folder_path + f"{file_name}.mat"
    # Enregistrement dans un fichier .mat
    scipy.io.savemat(folder_and_file_name_path, mat_data)
