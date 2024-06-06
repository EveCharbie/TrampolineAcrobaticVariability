import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
# import spm1d
import matplotlib.lines as mlines
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_point,
    find_index
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

joint_center_name_all_axes = data_xsens[0].columns
n_columns_all_axes = len(joint_center_name_all_axes)

plt.figure(figsize=(30, 30))

name_to_exclude = ['UpperLegR_X', 'UpperLegR_Y', 'UpperLegR_Z',
                   'UpperLegL_X', 'UpperLegL_Y', 'UpperLegL_Z',
                   'UpperArmR_X', 'UpperArmR_Y', 'UpperArmR_Z',
                   'UpperArmL_X', 'UpperArmL_Y', 'UpperArmL_Z',
                   'Head_X', 'Head_Y', 'Head_Z',
                   ]

columns_to_exclude = []
for column_name in name_to_exclude:
    column_index = find_index(column_name, joint_center_name_all_axes.tolist())
    if column_index is not None:
        columns_to_exclude.append(column_index)

members = [
    "Pelvis",
    # "Head",
    "Elbow R",
    "Wrist R",
    "Elbow L",
    "Wrist L",
    "Knee R",
    "Ankle R",
    "Knee L",
    "Ankle L",
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
        plt.plot(df.iloc[:, i], label=f"Vicon Trial {idx + 1}", alpha=0.7, linewidth=1, color=colors_vicon[idx])

    # Plot Xsens data
    for idx, df in enumerate(data_xsens):
        plt.plot(df.iloc[:, i], label=f"Xsens Trial {idx + 1}", alpha=0.7, linewidth=1, color=colors_xsens[idx])

    plt.title(f"{jc_name}")
    if i in (0, 1, 2):
        plt.ylabel("Rotation (rad)")
    else:
        plt.ylabel("Position (m)")

    if i == 0:
        legend_vicon = mlines.Line2D([], [], color='red', markersize=15, label='Vicon')
        legend_xsens = mlines.Line2D([], [], color='blue', markersize=15, label='Xsens')

        plt.legend(handles=[legend_vicon, legend_xsens], loc='upper right')

plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.5)
plt.close()

## Plot SD all axes

std_xsens_all_data = []
std_vicon_all_data = []
plt.figure(figsize=(30, 24))
for i in range(n_columns_all_axes):
    if i in columns_to_exclude:
        continue

    plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
    col_name = joint_center_name_all_axes[i]

    data_vicon_col = [df.iloc[:, i].values for df in data_vicon]
    std_vicon = np.std(data_vicon_col, axis=0)
    std_vicon_all_data.append(std_vicon)

    data_xsens_col = [df.iloc[:, i].values for df in data_xsens]
    std_xsens = np.std(data_xsens_col, axis=0)
    std_xsens_all_data.append(std_xsens)

    plt.plot(std_vicon, label=f"Vicon - {col_name}", alpha=0.7, linewidth=1, color="red")
    plt.plot(std_xsens, label=f"Xsens - {col_name}", alpha=0.7, linewidth=1, color="blue")

    plt.title(f"SD - {col_name}")

    if i in (0, 1, 2):
        plt.ylabel("SD (rad)")
    else:
        plt.ylabel("SD (m)")

    if i == 0:
        plt.legend(loc='upper right')
plt.tight_layout()
plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.5)
plt.close()

std_xsens_all_data = np.stack(std_xsens_all_data)
std_vicon_all_data = np.stack(std_vicon_all_data)


## Moyennes de l ecart type pour les 3 axes

result_xsens = np.zeros((len(members), n_points))
result_vicon = np.zeros((len(members), n_points))

fig, axs = plt.subplots(3, 3, figsize=(13, 17))

for i in range(len(members)):
    start_index = i * 3
    end_index = start_index + 3
    result_xsens[i] = np.mean(std_xsens_all_data[start_index:end_index], axis=0)
    result_vicon[i] = np.mean(std_vicon_all_data[start_index:end_index], axis=0)

    row = i // 3
    col = i % 3
    # S'assurer que les étiquettes pour la légende sont définies
    line1, = axs[row, col].plot(result_vicon[i], color="red", label="Vicon")
    line2, = axs[row, col].plot(result_xsens[i], color="blue", label="Xsens")
    axs[row, col].set_title(f'{members[i]}')
    axs[row, col].set_ylabel('SD')

    # Pour inclure la légende sur chaque sous-plot ou juste pour i == 1 (exclusif)
    if i == 0:  # Cette condition peut être changée pour `True` si vous souhaitez la légende sur tous les graphiques
        axs[row, col].legend(loc='upper right')

        # Vérifiez si le sous-graphe est dans la dernière ligne
    if row == 2:  # La dernière ligne a row == 2 dans une grille 3x3
        axs[row, col].set_xlabel("Time (%)")



plt.subplots_adjust(top=0.975, bottom=0.09, left=0.05, right=0.995, hspace=0.2, wspace=0.19)
plt.close()



fig = plt.figure(figsize=(15, 20))
gs = fig.add_gridspec(5, 2)  # 5 lignes, 2 colonnes
y_ticks = np.arange(0, 0.13, 0.03)
# Premier graphique en haut centré
ax_top = fig.add_subplot(gs[0, :])
start_index = 0
end_index = 3
result_xsens[0] = np.mean(std_xsens_all_data[start_index:end_index], axis=0)
result_vicon[0] = np.mean(std_vicon_all_data[start_index:end_index], axis=0)
ax_top.plot(result_vicon[0], color="red", label="Vicon")
ax_top.plot(result_xsens[0], color="blue", label="Xsens")
ax_top.set_title(f'{members[0]}')
ax_top.set_ylabel('SDtotal (rad)')
ax_top.legend(loc='upper right')

# Ordre personnalisé pour les graphiques restants
# custom_order = [1, 3, 2, 4, 5, 7, 6, 8]
custom_order = [3, 1, 4, 2, 7, 5, 8, 6]

# Graphiques restants en 2 colonnes selon l'ordre personnalisé
for i, idx in enumerate(custom_order, start=1):
    start_index = idx * 3
    end_index = start_index + 3
    result_xsens[idx] = np.mean(std_xsens_all_data[start_index:end_index], axis=0)
    result_vicon[idx] = np.mean(std_vicon_all_data[start_index:end_index], axis=0)

    row = (i - 1) // 2 + 1  # Commencer à la deuxième ligne
    col = (i - 1) % 2
    ax = fig.add_subplot(gs[row, col])
    ax.plot(result_vicon[idx], color="red", label="Vicon")
    ax.plot(result_xsens[idx], color="blue", label="Xsens")
    ax.set_title(f'{members[idx]}')
    ax.set_ylim(0, 0.13)  # Limite de l'axe y
    ax.set_yticks(y_ticks)  # Définir les ticks de l'axe y

    # Ajouter l'étiquette de l'axe x pour la dernière ligne
    if row == 4:
        ax.set_xlabel("Time (%)")

        # Ajouter l'étiquette de l'axe x pour la dernière ligne
    if col == 0:
        ax.set_ylabel("SDtotal (m)")

    # if col == 1:
    #     ax.set_yticklabels([])

# Ajout d'un titre global et ajustement des espaces entre les sous-graphiques
# fig.suptitle('Comparaison des résultats Xsens et Vicon')
plt.subplots_adjust(top=0.977,
                    bottom=0.049,
                    left=0.075,
                    right=0.99,
                    hspace=0.329,
                    wspace=0.143
                    )

plt.show()










# ## SPM plot

all_data_vicon = []
all_data_xsens = []

for i in range(len(data_xsens)):
    all_data_vicon.append(data_xsens[i])

for i in range(len(data_vicon)):
    all_data_xsens.append(data_vicon[i])

all_data_vicon = np.array(all_data_vicon)
all_data_xsens = np.array(all_data_xsens)

total_dofs = all_data_vicon.shape[2]


# plt.figure(figsize=(30, 24), dpi=100)
#
# for dof in range(total_dofs):
#     ax = plt.subplot(6, 6, dof + 1)
#
#     # T-test pour données indépendantes
#     t = spm1d.stats.ttest2(all_data_vicon[:, :, dof], all_data_xsens[:, :, dof])
#     ti = t.inference(alpha, two_tailed=True, interp=True)
#
#     # Plot des résultats
#     ti.plot(ax=ax)
#     ti.plot_threshold_label(ax=ax, fontsize=8)  # Ajoute le seuil de signification sur le graphique
#     ti.plot_p_values(ax=ax, size=8, offsets=[(0, 0.5)])  # Affiche les valeurs p sur le graphique
#     plt.title(f"{joint_center_name_all_axes[dof]}")
#
#     # Cacher les noms des axes pour les graphiques intérieurs
#     if dof % 6 != 0:  # Cacher les étiquettes de l'axe y sauf pour la première colonne
#         plt.setp(ax.get_yticklabels(), visible=False)
#     if dof < 30:  # Cacher les étiquettes de l'axe x sauf pour la dernière ligne
#         plt.setp(ax.get_xticklabels(), visible=False)
#
# # Ajustement de l'espacement et affichage
# plt.tight_layout()
# plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.5, bottom=0.05)
plt.show()