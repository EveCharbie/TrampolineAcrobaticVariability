import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_point,
    calculate_means_for_xyz,
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
columns_to_exclude = [18, 19, 20, 27, 28, 29]

## Plot all try
for i in range(max_columns):
    if i in columns_to_exclude:
        continue
    plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
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


## Plot SD all axes

sd_xsens = []
sd_vicon = []
plt.figure(figsize=(24, 24))
for i in range(max_columns):
    if i in columns_to_exclude:
        continue

    plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
    col_name = name_columns[i]

    data_vicon_col = [df.dataframe.iloc[:, i].values for df in data_vicon]
    std_vicon = np.std(data_vicon_col, axis=0)
    sd_vicon.append(std_vicon)

    data_xsens_col = [df.dataframe.iloc[:, i].values for df in data_xsens]
    std_xsens = np.std(data_xsens_col, axis=0)
    sd_xsens.append(std_xsens)

    plt.plot(std_vicon, label=f"Vicon - {col_name}", alpha=0.7, linewidth=1)
    plt.plot(std_xsens, label=f"Xsens - {col_name}", alpha=0.7, linewidth=1)

    plt.title(f"SD - {col_name}")
    plt.xlabel("Frame")
    plt.ylabel("SD")
    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()

sd_xsens = np.stack(sd_xsens)
sd_vicon = np.stack(sd_vicon)

means_vicon = calculate_means_for_xyz(data_vicon)
means_xsens = calculate_means_for_xyz(data_xsens)

columns_to_excludev2 = [6, 9]
members = [
    "Pelvis",
    "Tete",
    "AvBrasD",
    "MainD",
    "AvBrasG",
    "MainG",
    "CuisseD",
    "JambeD",
    "PiedD",
    "CuisseG",
    "JambeG",
    "PiedG",
]


## Plot SD on mean axis
num_points = means_vicon[0].shape[1]
plt.figure(figsize=(20, num_points * 3))

for point_index in range(num_points):
    if point_index in columns_to_excludev2:
        continue
    col_name = members[point_index]
    point_data_vicon = np.array([means[:, point_index] for means in means_vicon])
    std_dev_vicon = np.std(point_data_vicon, axis=0)

    point_data_xsens = np.array([means[:, point_index] for means in means_xsens])
    std_dev_xsens = np.std(point_data_xsens, axis=0)

    plt.subplot(num_points // 3 + 1, 3, point_index + 1)
    plt.plot(std_dev_vicon, label=f"Vicon Point {col_name}", linewidth=1)
    plt.plot(std_dev_xsens, label=f"Xsens Point {col_name}", linewidth=1)
    plt.title(f"Écart type - Point {col_name}")
    plt.xlabel("Frame")
    plt.ylabel("Écart Type")
    if point_index == 0:
        plt.legend(loc='upper right')
plt.tight_layout()


## Moyennes de l ecart type pour les 3 axes
result_xsens = np.zeros((10, 100))
result_vicon = np.zeros((10, 100))

for i in range(10):
    start_index = i * 3
    end_index = start_index + 3
    result_xsens[i] = np.mean(sd_xsens[start_index:end_index], axis=0)
for i in range(10):
    start_index = i * 3
    end_index = start_index + 3
    result_vicon[i] = np.mean(sd_vicon[start_index:end_index], axis=0)

fig, axs = plt.subplots(5, 2, figsize=(10, 20))

for i in range(10):
    row = i // 2
    col = i % 2
    axs[row, col].plot(result_vicon[i])
    axs[row, col].plot(result_xsens[i])
    axs[row, col].set_title(f'Plot {members[i]}')
    axs[row, col].set_xlabel('Time')
    axs[row, col].set_ylabel('Value')
    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

