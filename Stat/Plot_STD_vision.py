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
    'GuSe': 'Athlete 1',
    'JaSh': 'Athlete 2',
    'JeCa': 'Athlete 3',
    'AnBe': 'Athlete 4',
    'AnSt': 'Athlete 5',
    'SaBe': 'Athlete 6',
    'JoBu': 'Athlete 7',
    'JaNo': 'Athlete 8',
    'SaMi': 'Athlete 9',
    'AlLe': 'Athlete 10',
    'MaBo': 'Athlete 11',
    'SoMe': 'Athlete 12',
    'JeCh': 'Athlete 13',
    'LiDu': 'Athlete 14',
    'LeJa': 'Athlete 15',
    'ArMa': 'Athlete 16',
    'AlAd': 'Athlete 17'
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
    fig, ax = plt.subplots(figsize=(5, 8))
    initial_ticks = np.arange(0, max_value, 0.2)
    current_ticks = list(initial_ticks)
    current_labels = [f"{tick:.1f}" for tick in initial_ticks]

    # Boucle sur les sujets
    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):
        color = name_to_color[name_subject]
        anonyme = anonyme_name[name_subject]
        up_line = max_value + up_subject
        np.set_printoptions(threshold=np.inf)
        for idx_trials in range(
                len(gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0])):
            gaze_position = \
                gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials]

            data = np.zeros(len(gaze_position), dtype=int)

            tolerance = 1e-6  # Tolérance pour vérifier si ligne[2] est très proche de 0

            for idx_ligne, ligne in enumerate(gaze_position):
                if (X[0][0] <= ligne[0] <= X[0][1] and
                        Y[:, 1][0] <= ligne[1] <= Y[:, 1][1] and
                        abs(ligne[2]) <= tolerance):
                    data[idx_ligne] = 0
                else:
                    data[idx_ligne] = 1

            # for idx_ligne, ligne in enumerate(gaze_position):
            #     if X[0][0] <= ligne[0] <= X[0][1] and Y[:, 1][0] <= ligne[1] <= Y[:, 1][1]:
            #         data[idx_ligne] = 0
            #     else:
            #         data[idx_ligne] = 1

            data = pd.DataFrame(data) #Trampo
            # data = pd.DataFrame(wall_index_all_subjects_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials][:, 0]) #Ground

            data_norm = data.apply(lambda x: safe_interpolate(x, num_points))

            y_line_position = up_line
            y_values = np.full(len(data_norm[0]), np.nan)
            y_values[data_norm[0] == 0] = y_line_position
            ax.plot(y_values, '-', color=color, label='Presence of Zero' if idx_trials == 0 else "")

            if max_value + up_subject not in current_ticks:
                current_ticks.append(max_value + up_subject)
                current_labels.append(str(anonyme))
            current_ticks.sort()
            special_index = current_ticks.index(max_value + up_subject)
            current_labels[special_index] = str(anonyme)

            up_line += 0.0005

        plt.plot(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt][idx_subject], label=f'Subject {idx_subject + 1}',
                 color=color, linestyle='--')
        up_subject += 0.022

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels, fontsize=9)

    for label in ax.get_yticklabels():
        if label.get_text().replace('.', '', 1).isdigit():
            label.set_fontsize(10)
        else:
            label.set_fontsize(5)

    ax.set_title(f"Horizontal Line Indicating Presence of Zeros {mvt}")
    ax.set_xlabel("Index")
    # ax.set_ylabel("Line Presence (Custom Y Position)")
    ax.set_xlim(0, len(data_norm[0]))
    ax.set_ylim(0, max_value + 0.48)

    # Ajout de la légende
    line1, = plt.plot([], [], color='black', label='Gaze on trampoline')
    line2, = plt.plot([], [], color='black', linestyle='--', label='SDtotal on pelvic rotation')
    plt.legend(handles=[line1, line2], fontsize='small')

    plt.title(f'{name_acro}', fontsize=15)
    plt.xlabel('Time (%)')
    # plt.ylabel('SD pelvic rotation')
    plt.savefig(f"/home/lim/Documents/StageMathieu/Gaze_trampo/{mvt}_gaze.png")
    plt.subplots_adjust(left=0.11, right=0.957, top=0.937, bottom=0.082)
    # plt.show()


