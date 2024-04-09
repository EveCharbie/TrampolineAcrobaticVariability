import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_point,
)

home_path_xsens = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/GuSe/Pos_JC/831<"
n_points = 100
alpha = 0.05

fichiers_mat_xsens = []
for root, dirs, files in os.walk(home_path_xsens):
    for file in files:
        if file.endswith(".mat"):
            full_path = os.path.join(root, file)
            fichiers_mat_xsens.append(full_path)

data_xsens = [load_and_interpolate_for_point(file, n_points) for file in fichiers_mat_xsens]

home_path_vicon = "/home/lim/Documents/StageMathieu/DataTrampo/Guillaume/Pos_JC/Gui_831_contact"

fichiers_mat_vicon = []
for root, dirs, files in os.walk(home_path_vicon):
    for file in files:
        if file.endswith(".mat"):
            full_path = os.path.join(root, file)
            fichiers_mat_vicon.append(full_path)

data_vicon = [load_and_interpolate_for_point(file, n_points) for file in fichiers_mat_vicon]

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
    "JambeD",
    "PiedD",
    "JambeG",
    "PiedG",
]

## Plot all try
colors_vicon = plt.cm.Reds(np.linspace(0.5, 1, len(data_vicon)))
colors_xsens = plt.cm.Blues(np.linspace(0.5, 1, len(data_xsens)))

for i in range(n_columns_all_axes):
    if i in columns_to_exclude:
        continue
    plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
    jc_name = joint_center_name_all_axes[i]

    # Plot Vicon data
    for idx, df in enumerate(data_vicon):
        plt.plot(df.dataframe.iloc[:, i], label=f"Vicon Trial {idx + 1}", alpha=0.7, linewidth=1, color=colors_vicon[idx])

    # Plot Xsens data
    for idx, df in enumerate(data_xsens):
        plt.plot(df.dataframe.iloc[:, i], label=f"Xsens Trial {idx + 1}", alpha=0.7, linewidth=1, color=colors_xsens[idx])

    plt.title(f"{jc_name}")
    plt.ylabel("Position")

    if i == 0:
        plt.legend(loc='upper right')

plt.tight_layout()


## Plot SD all axes

std_xsens_all_data = []
std_vicon_all_data = []
plt.figure(figsize=(30, 24))
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

    plt.plot(std_vicon, label=f"Vicon - {col_name}", alpha=0.7, linewidth=1, color="red")
    plt.plot(std_xsens, label=f"Xsens - {col_name}", alpha=0.7, linewidth=1, color="blue")

    plt.title(f"SD - {col_name}")
    plt.xlabel("Frame")
    plt.ylabel("SD")
    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()

std_xsens_all_data = np.stack(std_xsens_all_data)
std_vicon_all_data = np.stack(std_vicon_all_data)


## Moyennes de l ecart type pour les 3 axes

result_xsens = np.zeros((len(members), n_points))
result_vicon = np.zeros((len(members), n_points))

fig, axs = plt.subplots(5, 2, figsize=(14, 16))

for i in range(10):
    start_index = i * 3
    end_index = start_index + 3
    result_xsens[i] = np.mean(std_xsens_all_data[start_index:end_index], axis=0)
    result_vicon[i] = np.mean(std_vicon_all_data[start_index:end_index], axis=0)

    row = i // 2
    col = i % 2
    axs[row, col].plot(result_vicon[i], color="red")
    axs[row, col].plot(result_xsens[i], color="blue")
    axs[row, col].set_title(f'{members[i]}')
    axs[row, col].set_ylabel('SD')
    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()


## SPM plot

all_data_vicon = []
all_data_xsens = []

for i in range(len(data_xsens)):
    all_data_vicon.append(data_xsens[i].to_numpy_array())
    all_data_xsens.append(data_vicon[i].to_numpy_array())

all_data_vicon = np.array(all_data_vicon)
all_data_xsens = np.array(all_data_xsens)

total_dofs = all_data_vicon.shape[2]

n_cols = 6
n_rows = int(np.ceil(total_dofs / n_cols))

# Préparation de la figure et des axes pour les sous-graphes
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 4 * n_rows))
axes = axes.flatten()

current_ax_index = 0

for dof in range(total_dofs):
    if dof in columns_to_exclude:
        continue

    mean_total_vicon = np.mean(all_data_vicon[:, :, dof], axis=0)
    std_vicon = np.std(all_data_vicon[:, :, dof], axis=0)
    mean_total_xsens = np.mean(all_data_xsens[:, :, dof], axis=0)
    std_xsens = np.std(all_data_xsens[:, :, dof], axis=0)
    col_name = joint_center_name_all_axes[dof]

    p_value = np.array([ttest_ind(all_data_vicon[:, i, dof], all_data_xsens[:, i, dof]).pvalue for i in range(n_points)])
    p_value_ordo = np.sort(p_value)
    correction = np.array([alpha * (i + 1) / n_points for i in range(n_points)])
    p_value_threshold = p_value_ordo[p_value_ordo < correction].max()

    significatif = p_value < p_value_threshold
    indices_significatifs = np.where(significatif)[0]

    ax = axes[current_ax_index]
    ax.plot(mean_total_vicon, label=f"Vicon {col_name}", color="red")
    ax.fill_between(range(n_points), mean_total_vicon - std_vicon, mean_total_vicon + std_vicon, color="red", alpha=0.4)
    ax.plot(mean_total_xsens, label=f"Xsens {col_name}", color="blue")
    ax.fill_between(range(n_points), mean_total_xsens - std_xsens, mean_total_xsens + std_xsens, color="blue", alpha=0.4)

    if len(indices_significatifs) > 0:
        breaks = np.where(np.diff(indices_significatifs) > 1)[0] + 1
        start_points = np.insert(indices_significatifs[breaks], 0, indices_significatifs[0])
        end_points = np.append(indices_significatifs[breaks - 1], indices_significatifs[-1])
        for start, end in zip(start_points, end_points):
            ax.axvspan(start - 0.5, end + 0.5, color='gray', alpha=0.5)

    ax.set_title(f"{col_name}")
    if dof == 0:
        ax.legend(loc='upper right')

    current_ax_index += 1

# Masquer les axes non utilisés
for ax in axes[current_ax_index:]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
