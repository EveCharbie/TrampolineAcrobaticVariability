import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from TrampolineAcrobaticVariability.Function.Function_stat import safe_interpolate

members = ["AvBrasD", "MainD", "AvBrasG", "MainG", "JambeD", "PiedD", "JambeG", "PiedG"]
full_name_acrobatics = {
    '4-': '4-/',
    '4-o':  '4-o',
    '8--o': '8--o',
    '8-1<': '8-1<',
    '8-1o': '8-1o',
    '41': '41/',
    '811<': '811<',
    '41o': '41o',
    '8-3<': '8-3<',
    '42': '42/',
    '822': '822/',
    '831<': '831<',
    '43': '43/',

}

file_path = "/home/lim/Documents/StageMathieu/Tab_result3/sd_pelvis_and_gaze_orientation.mat"
data_loaded = scipy.io.loadmat(file_path)
mean_SD_pelvis_all_subjects_acrobatics = data_loaded["mean_SD_pelvis_all_subjects_acrobatics"]
members_data_all_subjects_acrobatics = data_loaded["members_data_all_subjects_acrobatics"]
movement_to_analyse = data_loaded["movement_to_analyse"]
gaze_position_temporal_evolution_projected_all_subject_acrobatics = data_loaded["gaze_position_temporal_evolution_projected_all_subject_acrobatics"]

movement_to_analyse = np.char.strip(movement_to_analyse)
X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])
tolerance = 1e-6
num_points = 100
time = np.arange(100)

liste_name = data_loaded["liste_name"]
list_name_for_movement = data_loaded["list_name_for_movement"]


for idx_mvt, mvt in enumerate(movement_to_analyse):
    name_acro = full_name_acrobatics[mvt]
    print(name_acro)

    data_pelvis = np.degrees(np.mean(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt], axis=0))
    data_members = np.mean(members_data_all_subjects_acrobatics[0][idx_mvt], axis=0)
    data_upper_body = np.mean(data_members[:4], axis=0)
    data_lower_body = np.mean(data_members[4:], axis=0)

    subject_gaze = []

    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):

        trials_gaze = []

        for idx_trials in range(
                len(gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0])):
            gaze_position = \
                gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials]

            data_mat = np.zeros(len(gaze_position), dtype=int)

            for idx_ligne, ligne in enumerate(gaze_position):
                if (X[0][0] <= ligne[0] <= X[0][1] and Y[:, 1][0] <= ligne[1] <= Y[:, 1][1] and abs(
                        ligne[2]) <= tolerance):
                    data_mat[idx_ligne] = 0
                else:
                    data_mat[idx_ligne] = 1

            data_mat = pd.DataFrame(data_mat)
            data_norm_mat = data_mat.apply(lambda x: safe_interpolate(x, num_points)).to_numpy().flatten()

            trials_gaze.append(data_norm_mat)

        trials_gaze_stack = pd.DataFrame(trials_gaze).T
        gaze_by_subject = np.mean(trials_gaze, axis=0)
        gaze_by_subject = (gaze_by_subject >= 0.5).astype(int)
        subject_gaze.append(gaze_by_subject)

    subject_gaze_stack = pd.DataFrame(subject_gaze)
    gaze_by_acrobatics = np.mean(subject_gaze_stack, axis=0)
    gaze_by_acrobatics = (gaze_by_acrobatics >= 0.5).astype(int)

    # Création des graphiques sur la même image
    # Création des graphiques sur la même image
    plt.figure(figsize=(14, 14))
    colors = np.linspace(0, 1, len(data_pelvis))

    # Premier graphique: pelvis_by_subject vs. data_upper_body avec gradient de couleur
    plt.subplot(2, 2, 1)
    scatter1 = plt.scatter(data_pelvis, data_upper_body, c=colors, cmap='viridis', label='Upper Body')
    plt.xlabel('Pelvis SDtotal')
    plt.ylabel('Upper Body SDtotal')
    plt.title('Pelvis vs Upper Body')
    plt.legend()

    # Deuxième graphique: pelvis_by_subject vs. data_lower_body avec gradient de couleur
    plt.subplot(2, 2, 2)
    scatter2 = plt.scatter(data_pelvis, data_lower_body, c=colors, cmap='viridis', label='Lower Body')
    plt.colorbar(scatter2, label='Time')
    plt.xlabel('Pelvis SDtotal')
    plt.ylabel('Lower Body SDtotal')
    plt.title('Pelvis vs Lower Body')
    plt.legend()

    # Troisième graphique: time vs. data_upper_body et data_lower_body avec data_pelvis sur un axe ordonné différent
    ax1 = plt.subplot(2, 2, 3)
    ax1.plot(time, data_upper_body, label='Upper Body', color='b')
    ax1.plot(time, data_lower_body, label='Lower Body', color='r')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Upper and Lower Body SDtotal')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(time, data_pelvis, label='Pelvis', color='g')
    ax2.set_ylabel('Pelvis SDtotal')
    ax2.legend(loc='upper right')

    # Tracé des zones grises en fonction de gaze_by_acrobatics
    in_gaze = False
    start = 0
    for i in range(len(gaze_by_acrobatics)):
        if gaze_by_acrobatics[i] == 0 and not in_gaze:
            start = i
            in_gaze = True
        elif gaze_by_acrobatics[i] == 1 and in_gaze:
            ax1.axvspan(start, i, color='gray', alpha=0.3)
            in_gaze = False
    if in_gaze:
        ax1.axvspan(start, len(gaze_by_acrobatics), color='gray', alpha=0.3)

    # Quatrième graphique: time vs. members avec data_pelvis sur un axe ordonné différent
    ax3 = plt.subplot(2, 2, 4)
    for i in range(4):
        ax3.plot(time, data_members[i], label=members[i], color='b')
    for i in range(4, 8):
        ax3.plot(time, data_members[i], label=members[i], color='r')

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Upper and Lower Body SDtotal')
    ax3.legend(loc='upper left')

    ax4 = ax3.twinx()
    ax4.plot(time, data_pelvis, label='Pelvis', color='g')
    ax4.set_ylabel('Pelvis SDtotal')
    ax4.legend(loc='upper right')

    # Tracé des zones grises en fonction de gaze_by_acrobatics
    in_gaze = False
    start = 0
    for i in range(len(gaze_by_acrobatics)):
        if gaze_by_acrobatics[i] == 0 and not in_gaze:
            start = i
            in_gaze = True
        elif gaze_by_acrobatics[i] == 1 and in_gaze:
            ax3.axvspan(start, i, color='gray', alpha=0.3)
            in_gaze = False
    if in_gaze:
        ax3.axvspan(start, len(gaze_by_acrobatics), color='gray', alpha=0.3)

    plt.tight_layout()
    plt.show()

    # # Boucle sur les sujets
    # for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):
    #     data_pelvis_by_subject = np.degrees(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt][idx_subject])
    #     data_members_by_subject = members_data_all_subjects_acrobatics[0][idx_mvt][idx_subject]
    #     data_upper_body_by_subject = np.mean(data_members_by_subject[:4], axis=0)
    #     data_lower_body_by_subject = np.mean(data_members_by_subject[4:], axis=0)
    #
    #     plt.figure(figsize=(14, 6))
    #     colors = np.linspace(0, 1, len(data_pelvis_by_subject))
    #     plt.subplot(1, 2, 1)
    #     scatter1 = plt.scatter(data_pelvis_by_subject, data_upper_body_by_subject, c=colors, cmap='viridis', label='Upper Body')
    #     plt.colorbar(scatter1, label='Position in List')
    #     plt.xlabel('Pelvis by Subject')
    #     plt.ylabel('Upper Body Mean')
    #     plt.title('Pelvis by Subject vs Upper Body Mean')
    #     plt.legend()
    #     plt.subplot(1, 2, 2)
    #     scatter2 = plt.scatter(data_pelvis_by_subject, data_lower_body_by_subject, c=colors, cmap='viridis', label='Lower Body')
    #     plt.colorbar(scatter2, label='Position in List')
    #     plt.xlabel('Pelvis by Subject')
    #     plt.ylabel('Lower Body Mean')
    #     plt.title('Pelvis by Subject vs Lower Body Mean')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()



