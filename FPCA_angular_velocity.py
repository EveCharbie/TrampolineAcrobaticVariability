import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_draw import plot_adjusted_fd
from Function.Function_Class_Basics import calculate_scores
import os
import scipy
import pickle
from Function.Function_Class_Basics import get_q, normaliser_essai
# from skfda import FDataGrid
# from skfda.preprocessing.dim_reduction import FPCA
# from skfda.representation.basis import (BSplineBasis)

angular_data_elite = []
angular_data_subelite = []

home = "/home/lim/Documents/XsensResults/"

names = os.listdir(home)
for name in enumerate(names):

    file_path = f"/home/lim/Documents/XsensResults/{name[1]}/43/"

    data_elite = []
    data_subelite = []
    elements = os.listdir(file_path)
    filtered_elements = [file for file in elements if file.endswith('eyetracking_metrics.pkl')]
    for files in enumerate(filtered_elements):
        file_path_complete = f"{file_path}{files[1]}"

        with open(file_path_complete, "rb") as fichier_pkl:
            # Charger les données à partir du fichier ".pkl"
            eye_tracking_metrics = pickle.load(fichier_pkl)

        Xsens_orientation_per_move = eye_tracking_metrics["Xsens_orientation_per_move"]
        expertise = eye_tracking_metrics["subject_expertise"]
        q = get_q(Xsens_orientation_per_move)
        df = np.degrees(q[37])

        if expertise == "Elite":
            data_elite.append(df)
        else:
            data_subelite.append(df)

    if data_elite:
        angular_data_elite.append(data_elite)

    if data_subelite:
        angular_data_subelite.append(data_subelite)

nombre_points = 100

# Configuration initiale du graphique
plt.figure(figsize=(10, 6))

# Tracer les données Elite après normalisation
for participant in angular_data_elite:
    for essai in participant:
        essai_normalise = normaliser_essai(essai, nombre_points)
        plt.plot(essai_normalise, color='blue', alpha=0.5)

# Tracer les données Sub-Elite après normalisation
for participant in angular_data_subelite:
    for essai in participant:
        essai_normalise = normaliser_essai(essai, nombre_points)
        plt.plot(essai_normalise, color='red', alpha=0.5)

# Ajout des légendes, titres et étiquettes d'axe
plt.title('Comparaison des performances Elite vs Sub-Elite (Normalisées)')
plt.xlabel('Point interpolé')
plt.ylabel('Valeurs normalisées')
plt.legend(['Elite', 'Sub-Elite'], loc='best')

# Afficher le graphique
plt.show()

# for file in range(len(alldata)):
#     mydata = alldata[file]
#
#     for dof in range(nb_art):
#         plt.figure(figsize=(5, 3))
#         plt.plot(mydata[:, dof], label=f'Segment {dof+1}')
#         plt.title(f'Segment {dof+1}')
#         plt.xlabel('Frame')
#         plt.ylabel('Angle (rad)')
#         plt.legend()
#         plt.show()
