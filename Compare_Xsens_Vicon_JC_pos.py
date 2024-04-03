import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_point,
)

home_path_xsens = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/GuSe/Pos_JC/831<"

fichiers_mat_xsens = []
for root, dirs, files in os.walk(home_path_xsens):
    for file in files:
        if file.endswith(".mat"):
            full_path = os.path.join(root, file)
            fichiers_mat_xsens.append(full_path)

data_xsens = [load_and_interpolate_for_point(file) for file in fichiers_mat_xsens]

home_path_vicon = "/home/lim/Documents/StageMathieu/DataTrampo/Guillaume/Pos_JC/Gui_831_contact"

fichiers_mat_vicon = []
for root, dirs, files in os.walk(home_path_vicon):
    for file in files:
        if file.endswith(".mat"):
            full_path = os.path.join(root, file)
            fichiers_mat_vicon.append(full_path)

data_vicon = [load_and_interpolate_for_point(file) for file in fichiers_mat_vicon]

joint_center_name_all_axes = data_vicon[0].get_column_names()
n_columns_all_axes = len(joint_center_name_all_axes)

plt.figure(figsize=(30, 30))

columns_to_exclude = [18, 19, 20, 27, 28, 29]
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


## Plot all try
for i in range(n_columns_all_axes):
    if i in columns_to_exclude:
        continue
    plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
    jc_name = joint_center_name_all_axes[i]

    for df in data_vicon:
        plt.plot(df.dataframe.iloc[:, i], label=f"Vicon Essai {data_vicon.index(df) + 1}", alpha=0.7, linewidth=1)

    for df in data_xsens:
        plt.plot(df.dataframe.iloc[:, i], label=f"Xsens Essai {data_xsens.index(df) + 1}", alpha=0.7, linewidth=1)

    plt.title(f"{jc_name}")
    plt.ylabel("Position")

    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()


## Plot SD all axes

std_xsens_all_data = []
std_vicon_all_data = []
plt.figure(figsize=(30, 30))
for i in range(n_columns_all_axes):
    if i in columns_to_exclude:
        continue

    plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
    col_name = joint_center_name_all_axes[i]

    data_vicon_col = [df.dataframe.iloc[:, i].values for df in data_vicon]
    std_vicon = np.std(data_vicon_col, axis=0)
    std_vicon_all_data.append(std_vicon)

    data_xsens_col = [df.dataframe.iloc[:, i].values for df in data_xsens]
    std_xsens = np.std(data_xsens_col, axis=0)
    std_xsens_all_data.append(std_xsens)

    plt.plot(std_vicon, label=f"Vicon - {col_name}", alpha=0.7, linewidth=1)
    plt.plot(std_xsens, label=f"Xsens - {col_name}", alpha=0.7, linewidth=1)

    plt.title(f"SD - {col_name}")
    plt.xlabel("Frame")
    plt.ylabel("SD")
    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()

std_xsens_all_data = np.stack(std_xsens_all_data)
std_vicon_all_data = np.stack(std_vicon_all_data)


## Moyennes de l ecart type pour les 3 axes

result_xsens = np.zeros((len(members), 100))
result_vicon = np.zeros((len(members), 100))

fig, axs = plt.subplots(5, 2, figsize=(10, 20))

for i in range(10):
    start_index = i * 3
    end_index = start_index + 3
    result_xsens[i] = np.mean(std_xsens_all_data[start_index:end_index], axis=0)
    result_vicon[i] = np.mean(std_vicon_all_data[start_index:end_index], axis=0)

    row = i // 2
    col = i % 2
    axs[row, col].plot(result_vicon[i])
    axs[row, col].plot(result_xsens[i])
    axs[row, col].set_title(f'Plot {i}')
    axs[row, col].set_ylabel('SD')
    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

