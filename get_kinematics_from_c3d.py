import numpy as np
import os
import biorbd
import scipy
import ezc3d
import bioviz
from Function.Function_Class_Basics import recons_kalman, find_index
import pandas as pd
from TrampolineAcrobaticVariability.Function.Function_draw import column_names

model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Guillaume/Guillaume.s2mMod")
b = bioviz.Viz(loaded_model=model)
b.exec()
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Guillaume/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Guillaume/"

file_intervals = [
    (file_path_c3d + "Gui_822_mvtPrep_1.c3d", (3164, 3451)),
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

    # # Remplir le tableau avec les données existantes ou NaN
    # for i, marker in enumerate(desired_order):
    #     if marker in useful_labels:
    #         original_index = find_index(marker, point_labels["value"])
    #         reordered_point_data[:, i, :] = point_data[:, original_index, :]
    #     else:
    #         print(f"Le marqueur '{marker}' n'a pas été trouvé et a été initialisé avec NaN.")

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

    q_recons, qdot_recons = recons_kalman(nf_mocap, n_markers_reordered, markers, model, initial_guess)
    b = bioviz.Viz(loaded_model=model)
    b.load_movement(q_recons)
    b.load_experimental_markers(markers[:, :, :])
    b.exec()

    Q = pd.DataFrame(Q.transpose(), columns=column_names)
    Q_pred = q_recons[:, frame_index]
    error = Q_pred - Q
    # print(error.transpose())

    # Création d'un dictionnaire pour le stockage
    mat_data = {"Q2": q_recons}

    folder_and_file_name_path = folder_path + f"{file_name}.mat"
    # Enregistrement dans un fichier .mat
    scipy.io.savemat(folder_and_file_name_path, mat_data)
