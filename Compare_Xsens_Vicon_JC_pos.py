import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_point,
)

home_path_xsens = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/GuSe/Q/831<"

fichiers_mat_xsens = []
for root, dirs, files in os.walk(home_path_xsens):
    for file in files:
        if file.endswith(".mat"):
            full_path = os.path.join(root, file)
            fichiers_mat_xsens.append(full_path)

data_xsens = [load_and_interpolate_for_point(file) for file in fichiers_mat_xsens]

home_path_vicon = "/home/lim/Documents/StageMathieu/DataTrampo/Guillaume/Q/Gui_831_contact"

fichiers_mat_vicon = []
for root, dirs, files in os.walk(home_path_vicon):
    for file in files:
        if file.endswith(".mat"):
            full_path = os.path.join(root, file)
            fichiers_mat_vicon.append(full_path)

data_vicon = [load_and_interpolate_for_point(file) for file in fichiers_mat_vicon]

name_columns = data_vicon[0].get_column_names()
n_columns = len(name_columns)

plt.figure(figsize=(24, 24))

max_columns = min(n_columns, len(name_columns))

for i in range(max_columns):
    plt.subplot(6, 6, i + 1)
    col_name = name_columns[i]

    for df in data_vicon:
        plt.plot(df.dataframe.iloc[:, i], label=f"Vicon Essai {data_vicon.index(df) + 1}", alpha=0.7, linewidth=1)

    for df in data_xsens:
        plt.plot(df.dataframe.iloc[:, i], label=f"Xsens Essai {data_xsens.index(df) + 1}", alpha=0.7, linewidth=1)

    plt.title(f"{col_name}")
    plt.xlabel("Frame")
    plt.ylabel("Valeur")

    if i == 0:
        plt.legend(loc='upper right')

plt.tight_layout()
plt.show()