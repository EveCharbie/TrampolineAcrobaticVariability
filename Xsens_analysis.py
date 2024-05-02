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
nombre_lignes_minimum = 10
n_points = 100
next_index = 0
time_values = np.linspace(0, n_points-1, num=n_points)

home_path = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/"
mean_length_member = np.loadtxt('/home/lim/Documents/StageMathieu/mean_total_length.csv', delimiter=',', skiprows=1)
movement_to_analyse = ['41', '42', '43', '41o', '4-', '4-o', '8--o', '8-1<', '8-1o', '8-3<', '811<', '822', '831<']
# movement_to_analyse = ['831<']

half_twists_per_movement = {
    '4-': 0,
    '4-o': 0,
    '41': 1,
    '42': 2,
    '43': 3,
    '41o': 1,
    '8-1<': 1,
    '8-1o': 1,
    '8-3<': 3,
    '8--o': 0,
    '811<': 1,
    '822': 2,
    '831<': 3,

}


members = ["Pelvis", "Tete", "AvBrasD", "MainD", "AvBrasG", "MainG", "JambeD", "PiedD", "JambeG", "PiedG"]
columns_names_anova_rotation = ['ID', 'Expertise', 'Timing', 'Std']
columns_names_anova_position = ['ID', 'Expertise', 'Timing'] + members[2:]
liste_name = [name for name in os.listdir(home_path) if os.path.isdir(os.path.join(home_path, name))]

columns_names_area = ['ID', 'Expertise'] + movement_to_analyse
area_df = pd.DataFrame(columns=columns_names_area, index=liste_name)

mean_SD_pelvis_all_subjects_acrobatics = []
wall_index_all_subjects_acrobatics = []

for id_mvt, mvt_name in enumerate(movement_to_analyse):

    # if "ArMa" in liste_name:
    #     liste_name.remove("ArMa")
    #     liste_name.remove("MaBo")

    temp_liste_name = []
    for name in liste_name:
        home_path_subject = f"{home_path}{name}/Pos_JC/{mvt_name}"
        if not os.path.exists(home_path_subject):
            print(f"Subject {name} didn't realize {mvt_name}")
        else:
            temp_liste_name.append(name)

    anova_rot_df = pd.DataFrame(columns=columns_names_anova_rotation)
    anova_pos_df = pd.DataFrame(columns=columns_names_anova_position)
    anova_time_to75_df = pd.DataFrame(index=range(nombre_lignes_minimum), columns=temp_liste_name)
    anova_time_to10_df = pd.DataFrame(index=range(nombre_lignes_minimum), columns=temp_liste_name)

    n_half_twist = half_twists_per_movement[mvt_name]

    mean_SD_pelvis_all_subjects = []
    wall_index_all_subject = []

    for id_name, name in enumerate(temp_liste_name):
        print(f"{name} {mvt_name} is running")
        home_path_subject = f"{home_path}{name}/Pos_JC/{mvt_name}"

        fichiers_mat_subject = []
        for root, dirs, files in os.walk(home_path_subject):
            for file in files:
                if file.endswith(".mat"):
                    full_path = os.path.join(root, file)
                    fichiers_mat_subject.append(full_path)

        data_subject = []
        length_subject = []
        subject_info_dict = {}
        wall_index_subject = []

        for file in fichiers_mat_subject:
            (data,
             subject_expertise,
             laterality,
             length_segment,
             wall_index) = load_and_interpolate_for_point(file, include_expertise_laterality_length=True)
            data_subject.append(data)
            length_subject.append(length_segment)
            wall_index_subject.append(wall_index)

        joint_center_name_all_axes = data_subject[0].columns
        n_columns_all_axes = len(joint_center_name_all_axes)

        columns_to_exclude = [18, 19, 20, 27, 28, 29]
        columns_to_excludev2 = [6, 9]


        ################ Plot all try ################
        plt.figure(figsize=(30, 30))

        colors_subject1 = plt.cm.Blues(np.linspace(0.5, 1, len(data_subject)))

        for i in range(n_columns_all_axes):
            if i in columns_to_exclude:
                continue
            ax = plt.subplot(6, 6, i + 1 - sum(j < i for j in columns_to_exclude))
            jc_name = joint_center_name_all_axes[i]

            # Plot subject data
            lines = []
            for idx, trial_data in enumerate(data_subject):
                trial_name = fichiers_mat_subject[idx]
                trial = trial_data.iloc[:, i]
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
        plt.savefig(f"{home_path_subject}/all_data.png")
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

            data_subject1_col = [trial.iloc[:, i].values for trial in data_subject]
            std_subject1 = np.std(data_subject1_col, axis=0)
            std_subject1_all_data.append(std_subject1)

            plt.plot(std_subject1, label=f"subject1 - {col_name}", alpha=0.7, linewidth=1, color="blue")

            plt.title(f"SD - {col_name}")
            plt.ylabel("SD")
            if i == 0:
                plt.legend(loc='upper right')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.5, wspace=0.5)
        plt.savefig(f"{home_path_subject}/all_axes_sd.png")
        plt.close()


        std_subject1_all_data = np.stack(std_subject1_all_data)

        ################ Mean STD for the 3 axes ################

        result_subject = np.zeros((len(members), n_points))

        fig, axs = plt.subplots(5, 2, figsize=(14, 16))

        for i in range(len(members)):
            start_index = i * 3
            end_index = start_index + 3
            result_subject[i] = np.sum(std_subject1_all_data[start_index:end_index], axis=0)  # sum or mean

            row = i // 2
            col = i % 2
            axs[row, col].plot(result_subject[i], color="blue")
            axs[row, col].set_title(f'{members[i]}')
            axs[row, col].set_ylabel('SD')
            if i == 0:
                plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"{home_path_subject}/mean_axes_sd.png")
        # plt.show()
        plt.close()

        ################ Get 75% of twist ################
        if n_half_twist != 0:

            all_data_subject = []
            for i in range(len(data_subject)):
                all_data_subject.append(data_subject[i])
            all_data_subject = np.array(all_data_subject)

            total_dofs = all_data_subject.shape[2]

            timestramp_treshold_subject_75 = []
            for trials in range(all_data_subject.shape[0]):
                initial_rot = all_data_subject[trials, 0, 2]
                for timestamp in range(n_points):
                    if laterality[0] == "D":
                        threshold = initial_rot - n_half_twist * 0.75 * math.pi
                        if threshold > all_data_subject[trials, timestamp, 2]:
                            timestramp_treshold_subject_75.append(timestamp)
                            break
                    else:
                        threshold = initial_rot + n_half_twist * 0.75 * math.pi
                        if threshold < all_data_subject[trials, timestamp, 2]:
                            timestramp_treshold_subject_75.append(timestamp)
                            break

            ##
            timestramp_treshold_subject_10 = []
            for trials in range(all_data_subject.shape[0]):
                initial_rot = all_data_subject[trials, 0, 2]
                for timestamp in range(n_points):
                    if laterality[0] == "D":
                        threshold = initial_rot - n_half_twist * 0.1 * math.pi
                        if threshold > all_data_subject[trials, timestamp, 2]:
                            timestramp_treshold_subject_10.append(timestamp)
                            break
                    else:
                        threshold = initial_rot + n_half_twist * 0.1 * math.pi
                        if threshold < all_data_subject[trials, timestamp, 2]:
                            timestramp_treshold_subject_10.append(timestamp)
                            break
            ##

            length_segment_mean = np.mean(length_subject, axis=0)

            treshold_3_4 = round(np.mean(timestramp_treshold_subject_75))
            print(f" Treshold value {timestramp_treshold_subject_75} and the mean {treshold_3_4}")

            std_takeoff = result_subject[0, 0]
            std_3_4 = result_subject[0, treshold_3_4]
            std_landing = result_subject[0, n_points-1]

            print(std_takeoff)
            print(std_3_4)
            print(std_landing)

            if len(timestramp_treshold_subject_75) > len(anova_time_to75_df):
                anova_time_to75_df = anova_time_to75_df.reindex(range(len(timestramp_treshold_subject_75)))

            serie_nan = pd.Series([np.nan] * len(anova_time_to75_df))
            serie_nan[:len(timestramp_treshold_subject_75)] = timestramp_treshold_subject_75

            anova_time_to75_df[name] = serie_nan

            if anova_time_to75_df[name].dtype != 'object':
                anova_time_to75_df[name] = anova_time_to75_df[name].astype('object')

            anova_time_to75_df.at[0, name] = str(subject_expertise[0])

            ##
            if len(timestramp_treshold_subject_10) > len(anova_time_to10_df):
                anova_time_to10_df = anova_time_to10_df.reindex(range(len(timestramp_treshold_subject_10)))

            serie_nan = pd.Series([np.nan] * len(anova_time_to10_df))
            serie_nan[:len(timestramp_treshold_subject_10)] = timestramp_treshold_subject_10

            anova_time_to10_df[name] = serie_nan

            if anova_time_to10_df[name].dtype != 'object':
                anova_time_to10_df[name] = anova_time_to10_df[name].astype('object')

            anova_time_to10_df.at[0, name] = str(subject_expertise[0])
            ##

            for id_member, member in enumerate(members[2:], start=2):
                anova_pos_df.at[next_index, 'ID'] = name
                anova_pos_df.at[next_index, 'Expertise'] = str(subject_expertise[0])
                anova_pos_df.at[next_index, 'Timing'] = "Takeoff"
                anova_pos_df.at[next_index, member] = result_subject[id_member, 0] / length_segment_mean[0][id_member-2]

            anova_rot_df.loc[next_index] = [name, str(subject_expertise[0]), "Takeoff", std_takeoff]
            next_index += 1

            for id_member, member in enumerate(members[2:], start=2):
                anova_pos_df.at[next_index, 'ID'] = name
                anova_pos_df.at[next_index, 'Expertise'] = str(subject_expertise[0])
                anova_pos_df.at[next_index, 'Timing'] = "75%"
                anova_pos_df.at[next_index, member] = result_subject[id_member, treshold_3_4] / length_segment_mean[0][id_member-2] * mean_length_member[id_member-2]

            anova_rot_df.loc[next_index] = [name, str(subject_expertise[0]), "75%", std_3_4]
            next_index += 1

            for id_member, member in enumerate(members[2:], start=2):
                anova_pos_df.at[next_index, 'ID'] = name
                anova_pos_df.at[next_index, 'Expertise'] = str(subject_expertise[0])
                anova_pos_df.at[next_index, 'Timing'] = "Landing"
                anova_pos_df.at[next_index, member] = result_subject[id_member, n_points-1] / length_segment_mean[0][id_member-2] # * mean_length_member[id_member-2]

            anova_rot_df.loc[next_index] = [name, str(subject_expertise[0]), "Landing", std_landing]
            next_index += 1

        mean_SD_pelvis_all_subjects.append(result_subject[0])
        wall_index_all_subject.append(wall_index_subject)

        area_under_curve = simpson(result_subject[0], x=time_values)
        print("Area under curves with simpson method :", area_under_curve)

        area_df.at[name, 'ID'] = name
        area_df.at[name, 'Expertise'] = str(subject_expertise[0])
        area_df.at[name, mvt_name] = area_under_curve

    mean_SD_pelvis_all_subjects_acrobatics.append(mean_SD_pelvis_all_subjects)
    wall_index_all_subjects_acrobatics.append(wall_index_all_subject)

    if n_half_twist != 0:

        print(anova_rot_df)
        anova_rot_df.to_csv(f'/home/lim/Documents/StageMathieu/Tab_result3/results_{mvt_name}_rotation.csv', index=False)
        anova_pos_df.to_csv(f'/home/lim/Documents/StageMathieu/Tab_result3/results_{mvt_name}_position.csv', index=False)
        anova_time_to75_df.to_csv(f'/home/lim/Documents/StageMathieu/Tab_result3/results_{mvt_name}_times_75.csv', index=False)
        anova_time_to10_df.to_csv(f'/home/lim/Documents/StageMathieu/Tab_result3/results_{mvt_name}_times_10.csv', index=False)

mat_data = {
                "mean_SD_pelvis_all_subjects_acrobatics": mean_SD_pelvis_all_subjects_acrobatics,
                "movement_to_analyse": movement_to_analyse,
                "wall_index_all_subjects_acrobatics": wall_index_all_subjects_acrobatics,
                "liste_name": liste_name
            }

print(area_df)
area_df.to_csv(f'/home/lim/Documents/StageMathieu/Tab_result3/results_area_under_curve2.csv', index=False)

scipy.io.savemat("/home/lim/Documents/StageMathieu/Tab_result3/sd_pelvis_and_gaze_orientation.mat", mat_data)

