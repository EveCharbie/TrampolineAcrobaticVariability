import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, shapiro, levene
from scipy.integrate import simpson
# import spm1d
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
import math
import mplcursors
import matplotlib.lines as mlines
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_point,
)

n_points = 100
next_index = 0
time_values = np.linspace(0, n_points-1, num=n_points)

home_path = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/"

liste_name = [name for name in os.listdir(home_path) if os.path.isdir(os.path.join(home_path, name))]

movement_to_analyse = ['41', '42', '43']

columns_names_area = ['ID', 'Expertise'] + movement_to_analyse
area_df = pd.DataFrame(columns=columns_names_area)

if "ArMa" in liste_name:
    liste_name.remove("ArMa")
    liste_name.remove("MaBo")


# n_demi_vrille = 3

for id_mvt, mvt_name in enumerate(movement_to_analyse):

    columns_names_anova = ['ID', 'Expertise', 'Timing', 'Std']
    anova_df = pd.DataFrame(columns=columns_names_anova)

    n_demi_vrille = int(mvt_name[-1])

    for id_name, name in enumerate(liste_name):
        print(f"{name} is running")
        home_path_subject1 = f"{home_path}{name}/Pos_JC/{mvt_name}"

        fichiers_mat_subject1 = []
        for root, dirs, files in os.walk(home_path_subject1):
            for file in files:
                if file.endswith(".mat"):
                    full_path = os.path.join(root, file)
                    fichiers_mat_subject1.append(full_path)

        data_subject1 = []
        subject_info_dict = {}
        for file in fichiers_mat_subject1:
            (data_subject,
             subject_expertise,
             laterality) = load_and_interpolate_for_point(file, include_expertise_laterality=True)
            data_subject1.append(data_subject)

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
            ax = plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
            jc_name = joint_center_name_all_axes[i]

            # Plot subject1 data
            lines = []
            for idx, trial_data in enumerate(data_subject1):
                trial_name = fichiers_mat_subject1[idx]  # Nom du fichier correspondant à ce trial
                trial = trial_data.iloc[:, i]  # Sélectionne la colonne correspondant à i
                line, = ax.plot(trial, label=f"subject1 Trial {trial_name}", alpha=0.7, linewidth=1,
                                color=colors_subject1[idx])
                lines.append(line)

            plt.title(f"{jc_name}")
            if i in (0, 1, 2):
                plt.ylabel("Rotation (rad)")
            else:
                plt.ylabel("Position")

            if i == 0:
                legend_subject1 = mlines.Line2D([], [], color='blue', markersize=15, label='subject1')
                plt.legend(handles=[legend_subject1], loc='upper right')

            mplcursors.cursor(lines, hover=True).connect("add", lambda sel: sel.annotation.set_text(
                f"File: {sel.artist.get_label()}"))

        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.5)
        # plt.show()
        plt.close()


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
        plt.close()


        std_subject1_all_data = np.stack(std_subject1_all_data)

        ################ Mean STD for the 3 axes ################

        result_subject1 = np.zeros((len(members), n_points))

        fig, axs = plt.subplots(5, 2, figsize=(14, 16))

        for i in range(len(members)):
            start_index = i * 3
            end_index = start_index + 3
            result_subject1[i] = np.sum(std_subject1_all_data[start_index:end_index], axis=0)  # sum or mean

            row = i // 2
            col = i % 2
            axs[row, col].plot(result_subject1[i], color="blue")
            axs[row, col].set_title(f'{members[i]}')
            axs[row, col].set_ylabel('SD')
            if i == 0:
                plt.legend(loc='upper right')
        plt.tight_layout()
        # plt.show()
        plt.close()

        ################ Get 75% of twist ################

        all_data_subject1 = []
        for i in range(len(data_subject1)):
            all_data_subject1.append(data_subject1[i])
        all_data_subject1 = np.array(all_data_subject1)

        total_dofs = all_data_subject1.shape[2]

        timestramp_treshold_subject1 = []
        for trials in range(all_data_subject1.shape[0]):
            initial_rot = all_data_subject1[trials, 0, 2]
            for timestamp in range(n_points):
                if laterality[0] == "D":
                    threshold = initial_rot - n_demi_vrille * 0.75 * math.pi
                    if threshold > all_data_subject1[trials, timestamp, 2]:
                        timestramp_treshold_subject1.append(timestamp)
                        break
                else:
                    threshold = initial_rot + n_demi_vrille * 0.75 * math.pi
                    if threshold < all_data_subject1[trials, timestamp, 2]:
                        timestramp_treshold_subject1.append(timestamp)
                        break

        treshold_3_4 = round(np.mean(timestramp_treshold_subject1))

        std_at_takeoff = []
        std_at_3_4 = []
        std_at_landing = []

        std_takeoff = result_subject1[0, 0]
        std_3_4 = result_subject1[0, treshold_3_4]
        std_landing = result_subject1[0, n_points-1]

        std_at_takeoff.append(std_takeoff)
        std_at_3_4.append(std_3_4)
        std_at_landing.append(std_landing)

        # auc = np.trapz(result_subject1[0], x=time_values)
        # print("Aire sous la courbe np :", auc)
        auc = simpson(result_subject1[0], x=time_values)
        print("Aire sous la courbe simps :", auc)

        print(std_at_takeoff)
        print(std_at_3_4)
        print(std_at_landing)

        ################
        # std_axes_subject1_3_4 = []
        # std_axes_subject1_landing = []
        #
        # for axes in range(n_columns_all_axes):
        #     values_to_std_3_4 = []
        #     values_to_std_landing = []
        #
        #     for trials in range(all_data_subject1.shape[0]):
        #
        #         value_to_std_3_4 = all_data_subject1[trials, timestramp_treshold_subject1[trials], axes]
        #         values_to_std_3_4.append(value_to_std_3_4)
        #
        #         value_to_std_landing = all_data_subject1[trials, n_points-1, axes]
        #         values_to_std_landing.append(value_to_std_landing)
        #
        #     std_axe_3_4 = np.std(values_to_std_3_4, axis=0)
        #     std_axes_subject1_3_4.append(std_axe_3_4)
        #
        #     std_axe_landing = np.std(values_to_std_landing, axis=0)
        #     std_axes_subject1_landing.append(std_axe_landing)
        #
        # mean_std_subject1_3_4 = np.zeros((len(members)))
        # mean_std_subject1_landing = np.zeros((len(members)))
        #
        # for i in range(len(members)):
        #     start_index = i * 3
        #     end_index = start_index + 3
        #     mean_std_subject1_3_4[i] = np.mean(std_axes_subject1_3_4[start_index:end_index], axis=0)
        #     mean_std_subject1_landing[i] = np.mean(std_axes_subject1_landing[start_index:end_index], axis=0)
        #
        # print(mean_std_subject1_3_4[0])
        # print(mean_std_subject1_landing[0])

        ################
        # anova_df.loc[next_index] = [name, str(subject_expertise[0]), "Takeoff", std_at_takeoff[0]]
        # next_index += 1
        anova_df.loc[next_index] = [name, str(subject_expertise[0]), "75%", std_at_3_4[0]]
        next_index += 1
        anova_df.loc[next_index] = [name, str(subject_expertise[0]), "landing", std_at_landing[0]]
        next_index += 1

        # area_df.loc[id_name] = [name, str(subject_expertise[0])]
    ##########
        area_df.at[id_name, 'ID'] = name
        area_df.at[id_name, 'Expertise'] = str(subject_expertise[0])
        area_df.at[id_name, mvt_name] = auc
    ##########

    expertises = anova_df["Expertise"].unique()
    timings = anova_df["Timing"].unique()

    print(anova_df)
    anova_df.to_csv(f'/home/lim/Documents/StageMathieu/results_4{n_demi_vrille}.csv', index=False)


    for expertise in expertises:
        for timing in timings:
            data_Shapiro = anova_df[(anova_df["Expertise"] == expertise) & (anova_df["Timing"] == timing)]["Std"]
            stat_Shapiro, p_value_Shapiro = shapiro(data_Shapiro)
            print(f"Groupe {expertise}, Moment {timing}, p_value Shapiro: {p_value_Shapiro}")

    data_Levene = [anova_df[(anova_df["Expertise"] == expertise) & (anova_df["Timing"] == timing)]["Std"]
                   for expertise in expertises for timing in timings]
    stat_Levene, p_value_Levene = levene(*data_Levene)
    print(f"p-value Levene:{p_value_Levene}")

    modele = ols("Std ~ C(Expertise) * C(Timing)", data=anova_df).fit()
    result_anova = sm.stats.anova_lm(modele, typ=2)
    print(result_anova)

print(area_df)
area_df.to_csv(f'/home/lim/Documents/StageMathieu/results_area_under_curve.csv', index=False)
