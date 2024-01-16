
####
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
folder_path = "/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/test"

# Liste des fichiers avec les intervals de frames spécifiques
file_intervals = [(file_path_mat + 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat', (3299, 3591)),
                 (file_path_mat + 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat', (3139, 3440)),
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


file_intervals = [
    (file_path_mat + 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat', (3299, 3591)),
    (file_path_mat + 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat', (3139, 3440)),
    # Autres fichiers et intervalles si nécessaire
]

# Charger et interpoler chaque essai
my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]


# # Exemple pour accéder aux données du premier fichier
# data_first_file = my_data_instances[1].dataframe
#
# print(data_first_file["Pelvis"]["X"])
#

data_first_file = my_data_instances[0]  # Première instance de MyData

# Accéder aux données de 'Pelvis_X' en utilisant l'index
pelvis_x_data = data_first_file.get_column_by_index("Pelvis", 0)


# Assumons que my_data_instances[0] et my_data_instances[1] sont déjà des instances de MyData
data_first_file = my_data_instances[0]  # Première instance de MyData
data_second_file = my_data_instances[1]  # Deuxième instance de MyData

# Accéder aux données de 'Pelvis_X' pour les deux instances
pelvis_x_data_first = data_first_file.get_column_by_index("Pelvis", 0)
pelvis_x_data_second = data_second_file.get_column_by_index("Pelvis", 0)

# Tracer les graphiques
plt.figure(figsize=(10, 6))
plt.plot(pelvis_x_data_first, label='Première Instance - Pelvis X')
plt.plot(pelvis_x_data_second, label='Deuxième Instance - Pelvis X')
plt.title('Données Pelvis X pour les Première et Deuxième Instances')
plt.xlabel('Index')
plt.ylabel('Valeurs')
plt.legend()
plt.show()
