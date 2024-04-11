import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import spm1d
import statsmodels.api as sm
from statsmodels.formula.api import ols
import math
import matplotlib.lines as mlines
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_pointv2,
)

n_points = 100
alpha = 0.05
next_index = 0

home_path = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/"

liste_name = [name for name in os.listdir(home_path) if os.path.isdir(os.path.join(home_path, name))]
noms_colonnes = ['ID', 'Expertise', 'Timing', 'Std']
anova_tab = np.zeros((len(liste_name) * 2, 4), dtype=object)
anova_tab[:] = noms_colonnes

for id_name, name in enumerate(liste_name):
    home_path_subject1 = f"{home_path}{name}/Pos_JC/43"

    fichiers_mat_subject1 = []
    for root, dirs, files in os.walk(home_path_subject1):
        for file in files:
            if file.endswith(".mat"):
                full_path = os.path.join(root, file)
                fichiers_mat_subject1.append(full_path)
    data_subject1 = [load_and_interpolate_for_pointv2(file, n_points) for file in fichiers_mat_subject1]
    subject_expertise = "Elite"
    laterality = "D"
    joint_center_name_all_axes = data_subject1[0].columns
    n_columns_all_axes = len(joint_center_name_all_axes)

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

    ################ Plot all try ################
    plt.figure(figsize=(30, 30))

    colors_subject1 = plt.cm.Blues(np.linspace(0.5, 1, len(data_subject1)))

    for i in range(n_columns_all_axes):
        if i in columns_to_exclude:
            continue
        plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
        jc_name = joint_center_name_all_axes[i]

        # Plot subject1 data
        for idx, trial in enumerate(data_subject1):
            plt.plot(trial.iloc[:, i], label=f"subject1 Trial {idx + 1}", alpha=0.7, linewidth=1, color=colors_subject1[idx])

        plt.title(f"{jc_name}")
        if i in (0, 1, 2):
            plt.ylabel("Rotation (rad)")
        else:
            plt.ylabel("Position")

        if i == 0:
            legend_subject1 = mlines.Line2D([], [], color='blue', markersize=15, label='subject1')

            plt.legend(handles=[legend_subject1], loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.5)


    ################ Plot SD all axes ################

    std_subject1_all_data = []
    plt.figure(figsize=(30, 24))
    for i in range(n_columns_all_axes):
        if i in columns_to_exclude:
            continue

        plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
        col_name = joint_center_name_all_axes[i]

        data_subject1_col = [trial.iloc[:, i].values for trial in data_subject1]
        std_subject1 = np.std(data_subject1_col, axis=0)
        std_subject1_all_data.append(std_subject1)

        plt.plot(std_subject1, label=f"subject1 - {col_name}", alpha=0.7, linewidth=1, color="blue")

        plt.title(f"SD - {col_name}")
        plt.ylabel("SD")
        if i == 0:
            plt.legend(loc='upper right')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.5)


    std_subject1_all_data = np.stack(std_subject1_all_data)

    ################ Moyennes de l ecart type pour les 3 axes ################

    result_subject1 = np.zeros((len(members), n_points))

    fig, axs = plt.subplots(5, 2, figsize=(14, 16))

    for i in range(len(members)):
        start_index = i * 3
        end_index = start_index + 3
        result_subject1[i] = np.mean(std_subject1_all_data[start_index:end_index], axis=0)

        row = i // 2
        col = i % 2
        axs[row, col].plot(result_subject1[i], color="blue")
        axs[row, col].set_title(f'{members[i]}')
        axs[row, col].set_ylabel('SD')
        if i == 0:
            plt.legend(loc='upper right')
    plt.tight_layout()
    # plt.show()


    all_data_subject1 = []
    for i in range(len(data_subject1)):
        all_data_subject1.append(data_subject1[i])
    all_data_subject1 = np.array(all_data_subject1)

    total_dofs = all_data_subject1.shape[2]

    timestramp_treshold_subject1 = []
    for trials in range(all_data_subject1.shape[0]):
        initial_rot = all_data_subject1[trials, 0, 2]
        for timestamp in range(n_points):
            if laterality =="D":
                threshold = initial_rot - 2.25 * math.pi
                if threshold > all_data_subject1[trials, timestamp, 2]:
                    timestramp_treshold_subject1.append(timestamp)
                    break
            else:
                threshold = initial_rot + 2.25 * math.pi
                if threshold < all_data_subject1[trials, timestamp, 2]:
                    timestramp_treshold_subject1.append(timestamp)
                    break

    # treshold_3_4 = round(np.mean(timestramp_treshold_subject1))
    #
    # std_at_3_4 = []
    # std_at_landing = []
    #
    # std_3_4 = result_subject1[0, treshold_3_4]
    # std_landing = result_subject1[0, n_points-1]
    #
    # std_at_3_4.append(std_3_4)
    # std_at_landing.append(std_landing)
    #
    # print(std_at_3_4)
    # print(std_at_landing)

    ################
    std_axes_subject1_3_4 = []
    std_axes_subject1_landing = []

    for axes in range(n_columns_all_axes):
        values_to_std_3_4 = []
        values_to_std_landing = []

        for trials in range(all_data_subject1.shape[0]):

            value_to_std_3_4 = all_data_subject1[trials, timestramp_treshold_subject1[trials], axes]
            values_to_std_3_4.append(value_to_std_3_4)

            value_to_std_landing = all_data_subject1[trials, n_points-1, axes]
            values_to_std_landing.append(value_to_std_landing)

        std_axe_3_4 = np.std(values_to_std_3_4, axis=0)
        std_axes_subject1_3_4.append(std_axe_3_4)

        std_axe_landing = np.std(values_to_std_landing, axis=0)
        std_axes_subject1_landing.append(std_axe_landing)

    mean_std_subject1_3_4 = np.zeros((len(members)))
    mean_std_subject1_landing = np.zeros((len(members)))

    for i in range(len(members)):
        start_index = i * 3
        end_index = start_index + 3
        mean_std_subject1_3_4[i] = np.mean(std_axes_subject1_3_4[start_index:end_index], axis=0)
        mean_std_subject1_landing[i] = np.mean(std_axes_subject1_landing[start_index:end_index], axis=0)

    ################

    print(mean_std_subject1_3_4[0])
    print(mean_std_subject1_landing[0])

    anova_tab[next_index] = [id_name, str(subject_expertise), "75%", mean_std_subject1_3_4[0]]
    next_index += 1
    anova_tab[next_index] = [id_name, str(subject_expertise), "landing", mean_std_subject1_landing[0]]
    next_index += 1


df = pd.DataFrame(anova_tab)
modele = ols("Std ~ C(Expertise) * C(Timing)", data=df).fit()
result_anova = sm.stats.anova_lm(modele, typ=2)
print(result_anova)