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


def load_and_interpolate(file, interval, num_points=100):
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


def calculate_mean_std(data_instances, member, axis):
    """
    Calcule la moyenne et l'écart-type pour un membre et un axe donnés
    sur toutes les instances de données.
    """
    data_arrays = [instance.get_column_by_index(member, axis) for instance in data_instances]
    mean_data = np.mean(data_arrays, axis=0)
    std_dev_data = np.std(data_arrays, axis=0)
    return mean_data, std_dev_data


# Charger et interpoler chaque essai
my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]


# Exemple acces aux données de 'Pelvis_X' en utilisant l'index
# data_first_file = my_data_instances[0]  # Première instance de MyData
# pelvis_x_data = data_first_file.get_column_by_index("Pelvis", 0)


# Liste des membres (à adapter en fonction de vos données)
members = ["Pelvis", "Thorax", "Tete", "EpauleD", "BrasD", "AvBrasD", "MainD", "EpauleG", "BrasG", "AvBrasG", "MainG",
           "CuisseD", "JambeD", "PiedD", "CuisseG", "JambeG", "PiedG"]

####### UNE COMPOSANTE PAR GRAPHIQUE #######
axes = [0, 1, 2]  # 0 pour X, 1 pour Y, 2 pour Z

for member in members:
    for axis in axes:
        try:
            # Calculer la moyenne et l'écart-type pour chaque membre et axe
            mean_data, std_dev_data = calculate_mean_std(my_data_instances, member, axis)

            # Tracer le graphique de la moyenne avec la zone d'écart-type
            plt.figure(figsize=(10, 6))
            plt.plot(mean_data, label=f'{member} {["X", "Y", "Z"][axis]}')
            plt.fill_between(range(len(mean_data)), mean_data - std_dev_data, mean_data + std_dev_data, color='gray', alpha=0.5)
            plt.title(f'Moyenne des Données {member} {["X", "Y", "Z"][axis]} avec Écart-Type')
            plt.xlabel('Time (%)')
            plt.ylabel('Value')
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


####### TOUTES LES COMPOSANTES PAR GRAPHIQUE #######

# Définir les couleurs pour les axes X, Y, et Z
colors = ['red', 'green', 'blue']
dpi = 300  # Haute résolution
desired_width_px = 333  # Largeur souhaitée en pixels
desired_height_px = 200  # Hauteur souhaitée en pixels
fig_width = desired_width_px / dpi  # Largeur en pouces
fig_height = desired_height_px / dpi  # Hauteur en pouces
legend_created = False  # Un indicateur pour vérifier si la légende a déjà été créée

for member in members:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    for axis in axes:
        try:
            # Calculer la moyenne et l'écart-type pour chaque membre et axe
            mean_data, std_dev_data = calculate_mean_std(my_data_instances, member, axis)

            # Obtenir la couleur correspondante pour l'axe actuel
            color = colors[axis]

            # Tracer le graphique de la moyenne avec la zone d'écart-type pour chaque axe
            plt.plot(mean_data, label=f'{["X", "Y", "Z"][axis]}', color=color, linewidth=0.3)
            plt.fill_between(range(len(mean_data)), mean_data - std_dev_data, mean_data + std_dev_data, alpha=0.4,
                             color=color, edgecolor='none')

        except KeyError:
            # Gérer le cas où une combinaison membre-axe n'existe pas
            print(f"Le membre {member} avec l'axe {['X', 'Y', 'Z'][axis]} n'existe pas.")

    # Configurer le graphique
    plt.xlabel('Time (%)', fontsize=4)
    plt.ylabel('Value', fontsize=4)
    plt.tick_params(axis='both', labelsize=3, width=0.3, length=1.5)

    # Rendre les traits de l'axe plus fins
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.3)

    # plt.legend()
    # plt.tick_params(axis='x', labelsize=3)  # Modifiez 'labelsize' selon vos besoins

    # Enregistrer le graphique
    file_name = f"{member}_all_axes_graph.png"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format='png', bbox_inches='tight')
    # Créer et enregistrer la légende uniquement pour le membre 'tête'
    if member == 'Tete' and not legend_created:
        # Récupérer les handles et les labels pour la légende
        handles, labels = ax.get_legend_handles_labels()

        # Créer une nouvelle figure pour la légende avec la taille souhaitée
        fig_leg = plt.figure(figsize=(2, 2), dpi=100)  # Dimensions en pouces, dpi pour la résolution
        ax_leg = fig_leg.add_subplot(111)

        # Ajouter la légende à la nouvelle figure avec une grande taille de police et en gras
        legend = ax_leg.legend(handles, labels, loc='center',
                               prop={'size': 20, 'weight': 'bold'})  # Ajustez la taille de police selon vos besoins

        # Augmenter l'épaisseur des traits de couleur dans la légende
        legend.get_lines()[0].set_linewidth(4)  # Augmentez l'épaisseur du trait pour "X"
        legend.get_lines()[1].set_linewidth(4)  # Augmentez l'épaisseur du trait pour "Y"
        legend.get_lines()[2].set_linewidth(4)  # Augmentez l'épaisseur du trait pour "Z"

        ax_leg.axis('off')

        # Enregistrer l'image de la légende avec les dimensions souhaitées
        leg_file_name = "legend.png"
        leg_file_path = os.path.join(folder_path, leg_file_name)
        fig_leg.savefig(leg_file_path, format='png', bbox_inches='tight',
                        pad_inches=0)  # Utilisation de pad_inches=0 pour supprimer les marges
        plt.close(fig_leg)  # Fermer la figure pour libérer la mémoire

        legend_created = True

    plt.close('all')