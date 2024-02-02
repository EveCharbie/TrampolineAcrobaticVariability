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

# start_frame = 3754
# end_frame = 4047
model = biorbd.Model('/home/lim/Documents/StageMathieu/Data_propre/SaMi/SaMi.bioMod')

# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = '/home/lim/Documents/StageMathieu/Data_propre/SaMi/Mvt_c3d/821_seul/'

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/Data_propre/SaMi/Q/"


# Liste des tuples (chemin du fichier, intervalle)
# file_intervals = [
#     (file_path_c3d + 'Sa_831_831_1.c3d', (3466, 3747)),
#     (file_path_c3d + 'Sa_831_831_3.c3d', (4138, 4427)),
#     (file_path_c3d + 'Sa_831_831_4.c3d', (3754, 4047)),
#     (file_path_c3d + 'Sa_831_831_5.c3d', (1632, 1928)),
#     (file_path_c3d + 'Sa_831_831_6.c3d', (4710, 5009)),
# ]


file_intervals = [
    (file_path_c3d + 'Sa_821_seul_1.c3d', (3349, 3650)),
    (file_path_c3d + 'Sa_821_seul_2.c3d', (3429, 3740)),
    (file_path_c3d + 'Sa_821_seul_3.c3d', (3209, 3520)),
    (file_path_c3d + 'Sa_821_seul_4.c3d', (3309, 3620)),
    (file_path_c3d + 'Sa_821_seul_5.c3d', (2689, 3000)),

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


for file_path, interval in file_intervals:
    file_name = os.path.basename(file_path).split('.')[0]

    c = ezc3d.c3d(file_path)
    point_data = c['data']['points'][:, :, interval[0]:interval[1]]
    n_markers = point_data.shape[1]
    nf_mocap = point_data.shape[2]
    f_mocap = c['parameters']['POINT']['RATE']['value'][0]
    point_labels = c['parameters']['POINT']['LABELS']
    # Extraire les noms de marqueurs utiles de 'point_labels'
    useful_labels = [label for label in point_labels['value'] if not label.startswith('*')]
    # Liste des noms de marqueurs dans l'ordre souhaité
    desired_order = ['EIASD', 'CID', 'EIPSD', 'EIPSG', 'CIG', 'EIASG', 'MANU', 'MIDSTERNUM', 'XIPHOIDE', 'C7', 'D3',
                     'D10', 'ZYGD', 'TEMPD', 'GLABELLE', 'TEMPG', 'ZYGG', 'CLAV1D', 'CLAV2D', 'CLAV3D', 'ACRANTD',
                     'ACRPOSTD', 'SCAPD', 'DELTD', 'BICEPSD', 'TRICEPSD', 'EPICOND', 'EPITROD', 'OLE1D', 'OLE2D',
                     'BRACHD', 'BRACHANTD', 'ABRAPOSTD', 'ABRASANTD', 'ULNAD', 'RADIUSD', 'METAC5D', 'METAC2D',
                     'MIDMETAC3D', 'CLAV1G', 'CLAV2G', 'CLAV3G', 'ACRANTG', 'ACRPOSTG', 'SCAPG', 'DELTG', 'BICEPSG',
                     'TRICEPSG', 'EPICONG', 'EPITROG', 'OLE1G', 'OLE2G', 'BRACHG', 'BRACHANTG', 'ABRAPOSTG', 'ABRANTG',
                     'ULNAG', 'RADIUSG', 'METAC5G', 'METAC2G', 'MIDMETAC3G', 'ISCHIO1D', 'TFLD', 'ISCHIO2D', 'CONDEXTD',
                     'CONDINTD', 'CRETED', 'JAMBLATD', 'TUBD', 'ACHILED', 'MALEXTD', 'MALINTD', 'CALCD', 'MIDMETA4D',
                     'MIDMETA1D', 'SCAPHOIDED', 'METAT5D', 'METAT1D', 'ISCHIO1G', 'TFLG', 'ISCHIO2G', 'CONEXTG',
                     'CONDINTG', 'CRETEG', 'JAMBLATG', 'TUBG', 'ACHILLEG', 'MALEXTG', 'MALINTG', 'CALCG', 'MIDMETA4G',
                     'MIDMETA1G', 'SCAPHOIDEG', 'METAT5G', 'METAT1G']

    indices = [useful_labels.index(marker) for marker in desired_order if marker in useful_labels]

    # Vérifier si tous les marqueurs de 'desired_order' ont été trouvés
    if len(indices) != len(desired_order):
        missing_markers = [elem for elem in desired_order if elem not in useful_labels]
        raise ValueError(f"Certains marqueurs de 'desired_order' ne sont pas trouvés dans 'point_labels': {missing_markers}")

    reordered_point_data = point_data[:, indices, :]

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
    # b = bioviz.Viz(loaded_model=model)
    # b.load_movement(q_recons)
    # b.load_experimental_markers(markers[:, :, :])
    # b.exec()


    # Création d'un dictionnaire pour le stockage
    mat_data = {'Q2': q_recons}

    folder_and_file_name_path = folder_path + f"{file_name}.mat"
    # Enregistrement dans un fichier .mat
    scipy.io.savemat(folder_and_file_name_path, mat_data)
