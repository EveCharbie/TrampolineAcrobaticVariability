
####
import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


class MyData:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Mapping des indices aux suffixes attendus
        self.index_suffix_map = {
            0: 'X', 1: 'Y', 2: 'Z'
        }

    def __getitem__(self, key):
        matching_columns = [col for col in self.dataframe.columns if col.startswith(key)]
        if not matching_columns:
            raise KeyError(f"Variable {key} not found.")
        return self.dataframe[matching_columns]

    def get_column_by_index(self, key, index):
        # Vérifie si l'index est valide
        if index not in self.index_suffix_map:
            raise KeyError(f"Invalid index {index}.")

        expected_suffix = self.index_suffix_map[index]
        column_name = f"{key}_{expected_suffix}"

        if column_name not in self.dataframe.columns:
            raise KeyError(f"Column {column_name} does not exist.")

        return self.dataframe[column_name]


# Chemin du dossier contenant les fichiers .mat
file_path_mat = '/home/lim/Documents/StageMathieu/Data_propre/SaMi/Q/'

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/MeanSD"

# Assurez-vous que le dossier de sauvegarde existe
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

file_intervals = [
    (file_path_mat + 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat', (3299, 3591)),
    (file_path_mat + 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat', (3139, 3440)),
    # Autres fichiers et intervalles si nécessaire
]


def load_and_interpolate(file, interval, num_points=1000):
    # Charger les données
    data = scipy.io.loadmat(file)
    df = pd.DataFrame(data['Q2']).T
    column_names = [
        "PelvisTranslation_X", "PelvisTranslation_Y", "PelvisTranslation_Z",
        "Pelvis_X", "Pelvis_Y", "Pelvis_Z",
        "Thorax_X", "Thorax_Y", "Thorax_Z",
        "Tete_X", "Tete_Y", "Tete_Z",
        "EpauleD_Y", "EpauleD_Z",
        "BrasD_X", "BrasD_Y", "BrasD_Z",
        "AvBrasD_X", "AvBrasD_Z",
        "MainD_X", "MainD_Y",
        "EpauleG_Y", "EpauleG_Z",
        "BrasG_X", "BrasG_Y", "BrasG_Z",
        "AvBrasG_X", "AvBrasG_Z",
        "MainG_X", "MainG_Y",
        "CuisseD_X", "CuisseD_Y", "CuisseD_Z",
        "JambeD_X",
        "PiedD_X", "PiedD_Z",
        "CuisseG_X", "CuisseG_Y", "CuisseG_Z",
        "JambeG_X",
        "PiedG_X", "PiedG_Z"
    ]
    df.columns = column_names

    # Sélectionner les données dans l'intervalle spécifié
    df_selected = df.iloc[interval[0]:interval[1]]

    # Interpoler chaque colonne pour avoir un nombre de points uniforme
    df_interpolated = df_selected.apply(lambda x: np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(x)), x))

    # Créer une instance de MyData avec les données interpolées
    my_data_instance = MyData(df_interpolated)
    return my_data_instance

# Charger et interpoler chaque essai


my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]

data_first_file = my_data_instances[0]  # Première instance de MyData

# Exemple acces aux données de 'Pelvis_X' en utilisant l'index
# pelvis_x_data = data_first_file.get_column_by_index("Pelvis", 0)


# data_first_file = my_data_instances[0]  # Première instance de MyData
# data_second_file = my_data_instances[1]  # Deuxième instance de MyData
#
# # Accéder aux données de 'Pelvis_X' pour les deux instances
# pelvis_x_data_first = data_first_file.get_column_by_index("Pelvis", 0)
# pelvis_x_data_second = data_second_file.get_column_by_index("Pelvis", 0)
#
#
# # Calculer la moyenne des deux ensembles de données
# mean_pelvis_x = np.mean([pelvis_x_data_first, pelvis_x_data_second], axis=0)
#
# # Calculer l'écart-type
# std_dev_pelvis_x = np.std([pelvis_x_data_first, pelvis_x_data_second], axis=0)
#
# # Tracer le graphique de la moyenne
# plt.figure(figsize=(10, 6))
# plt.plot(mean_pelvis_x, label='Moyenne - Pelvis X')
#
# # Ajouter la zone d'écart-type
# plt.fill_between(range(len(mean_pelvis_x)), mean_pelvis_x - std_dev_pelvis_x, mean_pelvis_x + std_dev_pelvis_x, color='gray', alpha=0.5)
#
# plt.title('Moyenne des Données Pelvis X pour les Première et Deuxième Instances avec Écart-Type')
# plt.xlabel('Index')
# plt.ylabel('Valeurs')
# plt.legend()
# plt.show()


# Liste des membres (à adapter en fonction de vos données)
members = ["Pelvis", "Thorax", "Tete", "EpauleD", "BrasD", "AvBrasD", "MainD", "EpauleG", "BrasG", "AvBrasG", "MainG",
           "CuisseD", "JambeD", "PiedD", "CuisseG", "JambeG", "PiedG"]

# Axes (X, Y, Z)
axes = [0, 1, 2]  # 0 pour X, 1 pour Y, 2 pour Z

for member in members:
    for axis in axes:
        try:
            # Accéder aux données pour chaque membre et axe pour les deux instances
            data_first = my_data_instances[0].get_column_by_index(member, axis)
            data_second = my_data_instances[1].get_column_by_index(member, axis)

            # Calculer la moyenne et l'écart-type
            mean_data = np.mean([data_first, data_second], axis=0)
            std_dev_data = np.std([data_first, data_second], axis=0)

            # Tracer le graphique de la moyenne avec la zone d'écart-type
            plt.figure(figsize=(10, 6))
            plt.plot(mean_data, label=f'Moyenne - {member} {["X", "Y", "Z"][axis]}')
            plt.fill_between(range(len(mean_data)), mean_data - std_dev_data, mean_data + std_dev_data, color='gray', alpha=0.5)
            plt.title(f'Moyenne des Données {member} {["X", "Y", "Z"][axis]} avec Écart-Type')
            plt.xlabel('Index')
            plt.ylabel('Valeurs')
            plt.legend()
            # plt.show()
            # Nom du fichier image
            file_name = f"{member}_{['X', 'Y', 'Z'][axis]}_graph.png"
            file_path = os.path.join(folder_path, file_name)

            # Enregistrer le graphique dans le dossier spécifié
            plt.savefig(file_path)
            plt.close()




        except KeyError:
            # Gérer le cas où une combinaison membre-axe n'existe pas
            print(f"Le membre {member} avec l'axe {['X', 'Y', 'Z'][axis]} n'existe pas.")
