import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from TrampolineAcrobaticVariability.Function.Function_stat import safe_interpolate
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
members = ["AvBrasD", "MainD", "AvBrasG", "MainG", "JambeD", "PiedD", "JambeG", "PiedG"]
members = ["ElbowR", "HandR", "ElbowL", "HandL", "KneeR", "FootR", "KneeL", "FootL"]

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

    subject_gaze_mat = []
    subject_gaze_ground = []

    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):

        trials_gaze_mat = []
        trials_gaze_ground = []

        for idx_trials in range(
                len(gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0])):
            gaze_position = \
                gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials]

            data_mat = np.zeros(len(gaze_position), dtype=int)
            data_ground = np.zeros(len(gaze_position), dtype=int)

            for idx_ligne, ligne in enumerate(gaze_position):
                if (X[0][0] <= ligne[0] <= X[0][1] and Y[:, 1][0] <= ligne[1] <= Y[:, 1][1] and abs(
                        ligne[2]) <= tolerance):
                    data_mat[idx_ligne] = 0
                else:
                    data_mat[idx_ligne] = 1

            for idx_ligne, ligne in enumerate(gaze_position):
                if abs((ligne[2]) <= tolerance):  # Ground
                    data_ground[idx_ligne] = 0
                else:
                    data_ground[idx_ligne] = 1

            data_mat = pd.DataFrame(data_mat)
            data_ground = pd.DataFrame(data_ground)

            data_norm_mat = data_mat.apply(lambda x: safe_interpolate(x, num_points)).to_numpy().flatten()
            data_norm_ground = data_ground.apply(lambda x: safe_interpolate(x, num_points)).to_numpy().flatten()

            trials_gaze_mat.append(data_norm_mat)
            trials_gaze_ground.append(data_norm_ground)

        gaze_by_subject_mat = np.mean(trials_gaze_mat, axis=0)
        gaze_by_subject_mat = (gaze_by_subject_mat >= 0.5).astype(int)
        subject_gaze_mat.append(gaze_by_subject_mat)

        gaze_by_subject_ground = np.mean(trials_gaze_ground, axis=0)
        gaze_by_subject_ground = (gaze_by_subject_ground >= 0.5).astype(int)
        subject_gaze_ground.append(gaze_by_subject_ground)

    gaze_by_acrobatics_mat = np.mean(subject_gaze_mat, axis=0)
    gaze_by_acrobatics_mat = (gaze_by_acrobatics_mat >= 0.5).astype(int)

    gaze_by_acrobatics_ground = np.mean(subject_gaze_ground, axis=0)
    gaze_by_acrobatics_ground = (gaze_by_acrobatics_ground >= 0.5).astype(int)




    # Dimensions des axes
    left, width = 0.1, 0.38
    bottom, height = 0.6, 0.35
    bottom_h = bottom - height - 0.05
    small_height = 0.01

    # Définitions des rectangles pour les grands graphiques
    rect1 = [left, bottom, width, height]
    rect2 = [left + width + 0.05, bottom, width, height]
    rect3 = [left, bottom_h, width, height]
    rect4 = [left + width + 0.05, bottom_h, width, height]

    # Définitions des rectangles pour les petits graphiques en dessous
    rect5 = [left, bottom_h - small_height - 0.15, width, small_height+0.5]
    rect6 = [left + width + 0.05, bottom_h - small_height - 0.05, width, small_height]

    fig = plt.figure(figsize=(14, 14))
    fig.suptitle(f"Acrobatics {full_name_acrobatics[mvt]}")
    # Création des axes
    ax1 = plt.axes(rect1)
    ax2 = plt.axes(rect3)
    ax3 = plt.axes(rect2)
    ax4 = plt.axes(rect4)
    ax_legend = plt.axes(rect6)
    ax_colorbar = plt.axes(rect5)

    colors = np.linspace(0, 1, len(data_pelvis))

    # Premier graphique: pelvis_by_subject vs. data_upper_body avec gradient de couleur
    scatter1 = ax1.scatter(data_pelvis, data_upper_body, c=colors, cmap='viridis', label='Upper Body')
    # ax1.set_xlabel('Pelvis SDtotal')
    ax1.set_ylabel('Upper Body SDtotal')
    ax1.set_title('Pelvis vs Limbs')
    # ax1.legend()
    ax1.set_xticklabels([])  # Supprimer les xlabels


    # Deuxième graphique: pelvis_by_subject vs. data_lower_body avec gradient de couleur
    scatter2 = ax2.scatter(data_pelvis, data_lower_body, c=colors, cmap='viridis', label='Lower Body')
    # fig.colorbar(scatter2, ax=ax2, label='Time')
    ax2.set_xlabel('Pelvis SDtotal')
    ax2.set_ylabel('Lower Body SDtotal')
    # ax2.set_title('Pelvis vs Lower Body')
    # ax2.legend()

    # Troisième graphique: time vs. data_upper_body et data_lower_body avec data_pelvis sur un axe ordonné différent
    ax3.plot(time, data_upper_body, label='Upper Body', color='b')
    ax3.plot(time, data_lower_body, label='Lower Body', color='r')
    # ax3.set_xlabel('Time')
    ax3.set_ylabel('Limbs joint center positions SDtotal')
    ax3.set_title('Pelvis and limbs across time')
    # ax3.legend(loc='upper left')
    ax3.set_xticklabels([])  # Supprimer les xlabels

    ax3bis = ax3.twinx()
    ax3bis.plot(time, data_pelvis, label='Pelvis', color='g')
    ax3bis.set_ylabel('Pelvis SDtotal')
    # ax3bis.legend(loc='upper right')

    # Tracé des zones grises en fonction de gaze_by_acrobatics
    in_gaze = False
    start = 0
    for i in range(len(gaze_by_acrobatics_mat)):
        if gaze_by_acrobatics_mat[i] == 0 and not in_gaze:
            start = i
            in_gaze = True
        elif gaze_by_acrobatics_mat[i] == 1 and in_gaze:
            ax3.axvspan(start, i, color='gray', alpha=0.4)
            in_gaze = False
    if in_gaze:
        ax3.axvspan(start, len(gaze_by_acrobatics_mat), color='gray', alpha=0.4)

    # Tracé des zones grises en fonction de gaze_by_acrobatics
    in_gaze = False
    start = 0
    for i in range(len(gaze_by_acrobatics_ground)):
        if gaze_by_acrobatics_ground[i] == 0 and not in_gaze:
            start = i
            in_gaze = True
        elif gaze_by_acrobatics_ground[i] == 1 and in_gaze:
            ax3.axvspan(start, i, color='gray', alpha=0.3)
            in_gaze = False
    if in_gaze:
        ax3.axvspan(start, len(gaze_by_acrobatics_ground), color='gray', alpha=0.3)

    # Quatrième graphique: time vs. members avec data_pelvis sur un axe ordonné différent
    upper_body_colors = plt.cm.Blues(np.linspace(0.4, 1, 4))
    lower_body_colors = plt.cm.Reds(np.linspace(0.4, 1, 4))

    lines = []
    labels = []
    for i in range(4):
        line, = ax4.plot(time, data_members[i], label=members[i], color=upper_body_colors[i])
        lines.append(line)
        labels.append(members[i])
    for i in range(4, 8):
        line, = ax4.plot(time, data_members[i], label=members[i], color=lower_body_colors[i - 4])
        lines.append(line)
        labels.append(members[i])

    ax4.set_xlabel('Time')
    ax4.set_ylabel('Limbs joint center positions SDtotal')
    # ax4.set_xticklabels([])  # Supprimer les xlabels

    ax4bis = ax4.twinx()
    line, = ax4bis.plot(time, data_pelvis, label='Pelvis', color='g')
    lines.append(line)
    labels.append('Pelvis')
    ax4bis.set_ylabel('Pelvis SDtotal')

    # Tracé des zones grises en fonction de gaze_by_acrobatics
    in_gaze = False
    start = 0
    for i in range(len(gaze_by_acrobatics_mat)):
        if gaze_by_acrobatics_mat[i] == 0 and not in_gaze:
            start = i
            in_gaze = True
        elif gaze_by_acrobatics_mat[i] == 1 and in_gaze:
            ax4.axvspan(start, i, color='gray', alpha=0.4)
            in_gaze = False
    if in_gaze:
        ax4.axvspan(start, len(gaze_by_acrobatics_mat), color='gray', alpha=0.4)

    # Tracé des zones grises en fonction de gaze_by_acrobatics
    in_gaze = False
    start = 0
    for i in range(len(gaze_by_acrobatics_ground)):
        if gaze_by_acrobatics_ground[i] == 0 and not in_gaze:
            start = i
            in_gaze = True
        elif gaze_by_acrobatics_ground[i] == 1 and in_gaze:
            ax4.axvspan(start, i, color='gray', alpha=0.3)
            in_gaze = False
    if in_gaze:
        ax4.axvspan(start, len(gaze_by_acrobatics_ground), color='gray', alpha=0.3)

    ax_legend.axis('off')  # Masquer les axes
    # Lignes et labels supplémentaires
    additional_lines = [Line2D([0], [0], color='b', lw=2),
                        Line2D([0], [0], color='r', lw=2),
                        Line2D([0], [0], color='grey', lw=2, alpha=0.5),
                        Line2D([0], [0], color='grey', lw=2, alpha=1.0)]
    additional_labels = ['Upper Body', 'Lower Body', 'Gaze on ground', 'Gaze on trampoline']

    all_lines = lines + additional_lines
    all_labels = labels + additional_labels

    ax_legend.legend(all_lines, all_labels, loc='upper center', title='Body Part', ncol=3)

    # Créer un subplot pour la colorbar
    norm = plt.Normalize(0, 100)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax_colorbar, orientation='horizontal')
    cbar.set_label('0 to 100% of movement')
    cbar.set_ticks([0, 25, 50, 75, 100])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax_colorbar.axis('off')  # Masquer les axes
    plt.savefig(f"/home/lim/Documents/StageMathieu/All_graphique/{mvt}_all_graphique.png", dpi=350)

    # plt.show()
    plt.close()



