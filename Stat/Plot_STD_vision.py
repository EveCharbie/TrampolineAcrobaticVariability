import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def safe_interpolate(x, num_points):
    # Create an array to store interpolated values, initialized with NaN
    interpolated_values = np.full(num_points, np.nan)

    # Check if all elements are finite, ignore NaNs for interpolation
    finite_mask = np.isfinite(x)
    if finite_mask.any():  # Ensure there's at least one finite value to interpolate
        # Interpolate only finite values
        valid_x = x[finite_mask]
        valid_indices = np.linspace(0, 1, len(x))[finite_mask]

        # Perform interpolation over the range with finite values
        interpolated_valid_values = np.interp(np.linspace(0, 1, num_points), valid_indices, valid_x)

        # Round interpolated values to the nearest integer
        rounded_values = np.round(interpolated_valid_values).astype(int)

        # Place rounded interpolated values back in the full array
        interpolated_values = rounded_values

    return interpolated_values

file_path = "/home/lim/Documents/StageMathieu/Tab_result/sd_pelvis_and_gaze_orientation.mat"
data_loaded = scipy.io.loadmat(file_path)
mean_SD_pelvis_all_subjects_acrobatics = data_loaded["mean_SD_pelvis_all_subjects_acrobatics"]
movement_to_analyse = data_loaded["movement_to_analyse"]
movement_to_analyse = np.char.strip(movement_to_analyse)

wall_index_all_subjects_acrobatics = data_loaded["wall_index_all_subjects_acrobatics"]
liste_name = data_loaded["liste_name"]
gaze_position_temporal_evolution_projected_all_subject_acrobatics = data_loaded["gaze_position_temporal_evolution_projected_all_subject_acrobatics"]
list_name_for_movement = data_loaded["list_name_for_movement"]

X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])

name_to_color = {
    'GuSe': '#1f77b4', 'JaSh': '#ff7f0e', 'JeCa': '#2ca02c', 'AnBe': '#d62728',
    'AnSt': '#9467bd', 'SaBe': '#8c564b', 'JoBu': '#e377c2', 'JaNo': '#7f7f7f',
    'SaMi': '#bcbd22', 'AlLe': '#17becf', 'MaBo': '#aec7e8', 'SoMe': '#ffbb78',
    'JeCh': '#98df8a', 'LiDu': '#ff9896', 'LeJa': '#c5b0d5', 'ArMa': '#c49c94',
    'AlAd': '#dbdb8d'
}

anonyme_name = {
    'GuSe': '1',
    'JaSh': '2',
    'JeCa': '3',
    'AnBe': '4',
    'AnSt': '5',
    'SaBe': '6',
    'JoBu': '7',
    'JaNo': '8',
    'SaMi': '9',
    'AlLe': '10',
    'MaBo': '11',
    'SoMe': '12',
    'JeCh': '13',
    'LiDu': '14',
    'LeJa': '15',
    'ArMa': '16',
    'AlAd': '17'
}

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

num_points = 100

for idx_mvt, mvt in enumerate(movement_to_analyse):
    name_acro = full_name_acrobatics[mvt]

    up_subject = 0.1
    max_value = 1 #np.max(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt].flatten())
    # Configuration initiale des axes et des listes de ticks et labels
    # fig, ax = plt.subplots(figsize=(5, 11))
    fig, ax = plt.subplots(figsize=(280 / 100, 396 / 100))
    initial_ticks = np.arange(0, max_value, 0.2)
    current_ticks = list(initial_ticks)
    current_labels = [f"{tick:.1f}" for tick in initial_ticks]

    # Boucle sur les sujets
    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):
        color = name_to_color[name_subject]
        anonyme = anonyme_name[name_subject]
        up_line = max_value + up_subject
        # np.set_printoptions(threshold=np.inf)
        for idx_trials in range(
                len(gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0])):
            gaze_position = \
                gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials]

            data_ground = np.zeros(len(gaze_position), dtype=int)
            tolerance = 1e-6  # Tolérance pour vérifier si ligne[2] est très proche de 0

            for idx_ligne, ligne in enumerate(gaze_position):
            #     if (X[0][0] <= ligne[0] <= X[0][1] and Y[:, 1][0] <= ligne[1] <= Y[:, 1][1] and abs(ligne[2]) <= tolerance):  # Trampo
                if abs((ligne[2]) <= tolerance):  # Ground
                    data_ground[idx_ligne] = 0
                else:
                    data_ground[idx_ligne] = 1
            data_ground = pd.DataFrame(data_ground)

            ##
            data_mat = np.zeros(len(gaze_position), dtype=int)

            for idx_ligne, ligne in enumerate(gaze_position):
                if (X[0][0] <= ligne[0] <= X[0][1] and Y[:, 1][0] <= ligne[1] <= Y[:, 1][1] and abs(ligne[2]) <= tolerance):  # Trampo
                    data_mat[idx_ligne] = 0
                else:
                    data_mat[idx_ligne] = 1
            data_mat = pd.DataFrame(data_mat)
            ##

            pd.set_option('display.max_rows', None)

            data_norm_ground = data_ground.apply(lambda x: safe_interpolate(x, num_points))
            data_norm_mat = data_mat.apply(lambda x: safe_interpolate(x, num_points))

            y_line_position = up_line
            y_values_ground = np.full(len(data_norm_ground[0]), np.nan)
            y_values_ground[data_norm_ground[0] == 0] = y_line_position
            ax.plot(y_values_ground, '-', color=color, alpha=0.1, label='Presence of Zero' if idx_trials == 0 else "")

            ##
            y_values_mat = np.full(len(data_norm_mat[0]), np.nan)
            y_values_mat[data_norm_mat[0] == 0] = y_line_position
            ax.plot(y_values_mat, '-', color=color, alpha=1, label='Presence of Zero' if idx_trials == 0 else "")
            ##

            if max_value + up_subject not in current_ticks:
                current_ticks.append(max_value + up_subject)
                current_labels.append(str(anonyme))
            current_ticks.sort()
            special_index = current_ticks.index(max_value + up_subject)
            current_labels[special_index] = str(anonyme)

            up_line += 0.0005

        plt.plot(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt][idx_subject], label=f'Subject {idx_subject + 1}',
                 color=color, linestyle='--')
        # up_subject += 0.022
        up_subject += 0.04

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)

    for label in ax.get_yticklabels():
        value = float(label.get_text())
        if value < 1:
            label.set_fontsize(9)
        else:
            label.set_fontsize(7)

    ax.set_title(f"Horizontal Line Indicating Presence of Zeros {mvt}")
    # ax.set_xlabel("Index")
    # ax.set_ylabel("Line Presence (Custom Y Position)")
    ax.set_xlim(0, len(data_norm_ground[0]))
    ax.set_ylim(0, max_value + 0.04*19.5)
    ax.tick_params(axis='x', labelsize=9)

    # Ajout de la légende
    line1, = plt.plot([], [], color='black', label='Gaze on trampoline')
    line2, = plt.plot([], [], color='black', linestyle='--', label='SDtotal on pelvic rotation')
    # plt.legend(handles=[line1, line2], fontsize='small')

    plt.title(f'{name_acro}', fontsize=11)
    # plt.xlabel('Time (%)')
    # plt.ylabel('SD pelvic rotation')
    plt.subplots_adjust(left=0.102, right=0.960, top=0.945, bottom=0.052)
    plt.savefig(f"/home/lim/Documents/StageMathieu/Gaze_ground/{mvt}_gaze.png", dpi=1000)
    # plt.show()


# Dummy plot to create legend
fig, ax = plt.subplots()

# Ajout de la légende
line1, = ax.plot([], [], color='black', label='Gaze on the ground')
line2, = ax.plot([], [], color='black', linestyle='--', label='SDtotal on pelvic rotation')

# Créez une légende et enregistrez-la dans une image séparée
figlegend = plt.figure(figsize=(3, 1), facecolor='white')
plt.figlegend(handles=[line1, line2], loc='center', fontsize='small')
figlegend.savefig('/home/lim/Documents/StageMathieu/Gaze_ground/legend.png', bbox_inches='tight', pad_inches=0)
