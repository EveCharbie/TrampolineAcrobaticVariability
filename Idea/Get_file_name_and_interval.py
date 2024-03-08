import numpy as np
import os
import pandas as pd
import biorbd

csv_path = "/home/lim/Documents/StageMathieu/DataTrampo/Labelling_trampo.csv"
interval_name_tab = pd.read_csv(csv_path, sep=';', usecols=['Participant', 'Analyse', 'Essai', 'Debut', 'Fin', 'Durée'])
valide = ['O']
interval_name_tab = interval_name_tab[interval_name_tab["Analyse"] == 'O']
interval_name_tab['Essai'] = interval_name_tab['Essai'] + '.c3d'

# Obtenir la liste des participants
participant_name = interval_name_tab['Participant'].unique()

for name in participant_name:
    essai_by_name = interval_name_tab[interval_name_tab["Participant"] == name].copy()  # Modifier ici
    essai_by_name.loc[:, 'Interval'] = essai_by_name.apply(lambda row: (row['Debut'], row['Fin']), axis=1)
    folder_path = f"/home/lim/Documents/StageMathieu/DataTrampo/{name}/Q/"
    model_path = f"/home/lim/Documents/StageMathieu/DataTrampo/{name}/{name}.s2mMod"
    model = biorbd.Model(model_path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_path = f"/home/lim/Documents/StageMathieu/DataTrampo/{name}/Tests/"
    file_intervals = []

    for index, row in essai_by_name.iterrows():
        c3d_file = row['Essai']
        interval = row['Interval']
        file_path_complet = f"{file_path}{c3d_file}"
        file_intervals.append((file_path_complet, interval))
    print(file_intervals)
