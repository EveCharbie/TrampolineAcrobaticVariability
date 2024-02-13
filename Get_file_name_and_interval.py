import numpy as np
import os
import pandas as pd

csv_path = "/home/lim/Documents/StageMathieu/DataTrampo/Labelling_trampo.csv"
df = pd.read_csv(csv_path, sep=';', usecols=['Participant', 'Analyse', 'Essai', 'Debut', 'Fin', 'Durée'])
valide = ['O']
df = df[df["Analyse"] == 'O']
df['Essai'] = df['Essai'] + '.c3d'

# Obtenir la liste unique des participants
participant_name = df['Participant'].unique()

# Afficher la liste
print(participant_name)

for name in participant_name:
    essai_by_name = df[df["Participant"] == name].copy()  # Modifier ici
    essai_by_name.loc[:, 'Interval'] = essai_by_name.apply(lambda row: (row['Debut'], row['Fin']), axis=1)
    folder_path = f"/home/lim/Documents/StageMathieu/DataTrampo/{name}/Q/"
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
