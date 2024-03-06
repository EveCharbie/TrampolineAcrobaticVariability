import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Function.Function_Class_Basics import get_q, normaliser_essai
import pickle
import os
import FDApy as fda

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
        df = []
        for dof in range(len(q)):
            df_dof = np.degrees(q[dof])
            df.append(df_dof)

        if expertise == "Elite":
            data_elite.append(df)
        else:
            data_subelite.append(df)

    if data_elite:
        angular_data_elite.append(data_elite)

    if data_subelite:
        angular_data_subelite.append(data_subelite)

nombre_points = 100
freq_acquisition = 200
interval = 1 / freq_acquisition


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


grid_points = np.arange(0, nombre_points) * interval

data_elite_normalisees = [[normaliser_essai(degre_liberte, nombre_points)
                           for degre_liberte in essai]
                          for participant in angular_data_elite
                          for essai in participant]
data_subelite_normalisees = [[normaliser_essai(degre_liberte, nombre_points)
                              for degre_liberte in essai]
                             for participant in angular_data_subelite
                             for essai in participant]


ids_elite = ['Elite' for _ in range(len(data_elite_normalisees))]
ids_subelite = ['Sub-Elite' for _ in range(len(data_subelite_normalisees))]

data_totale = np.vstack([data_elite_normalisees, data_subelite_normalisees])
ids_totales = np.array(ids_elite + ids_subelite)
