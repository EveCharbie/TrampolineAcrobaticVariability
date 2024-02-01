import pickle
import ezc3d
import os
import scipy.io
import biorbd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Function_Class_Graph import OrderMatData, column_names


file_path_mat = '/home/lim/Documents/StageMathieu/Data_propre/SaMi/'
file_name_mat = 'fichier5.mat'
folder_path = f"/home/lim/Documents/StageMathieu/Data_propre/SaMi/test5/"

data_loaded = scipy.io.loadmat(file_path_mat+file_name_mat)


if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# print(data_loaded.keys())
data_info_reloaded = {key: str(type(data_loaded[key])) for key in data_loaded.keys() if not key.startswith('__')}
# print(data_info_reloaded)
q2_data = data_loaded['Q2']
# print(q2_data)


DataFrame_with_colname = pd.DataFrame(q2_data)
# Inversez les lignes et les colonnes
DataFrame_with_colname = DataFrame_with_colname.T
DataFrame_with_colname.columns = column_names

my_data = OrderMatData(DataFrame_with_colname)

selected_data = my_data.dataframe.iloc[0:281]

# Identifier les groupes de membres
member_groups = set([name.split('_')[0] for name in column_names])


for group in member_groups:
    group_columns = [col for col in column_names if col.startswith(group)]

    # Utilisation de la nouvelle méthode pour récupérer les données
    group_data = selected_data[group_columns]

    plt.figure(figsize=(10, 6))
    for col in group_columns:
        plt.plot(group_data[col], label=col)

    plt.title(f"Graphique pour {group}")
    plt.xlabel("Index")
    plt.ylabel("Valeur")
    plt.legend()
    # plt.show()

    file_name = f"{group}_graph.png"
    file_path = os.path.join(folder_path, file_name)

    plt.savefig(file_path)
    plt.close()