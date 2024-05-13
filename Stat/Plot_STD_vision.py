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

file_path = "/home/lim/Documents/StageMathieu/Tab_result3/sd_pelvis_and_gaze_orientation.mat"
data_loaded = scipy.io.loadmat(file_path)
mean_SD_pelvis_all_subjects_acrobatics = data_loaded["mean_SD_pelvis_all_subjects_acrobatics"]
movement_to_analyse = data_loaded["movement_to_analyse"]
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

num_points = 100

for idx_mvt, mvt in enumerate(movement_to_analyse):

    up_subject = 0.1
    max_value = np.max(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt].flatten())
    # Configuration initiale des axes et des listes de ticks et labels
    fig, ax = plt.subplots(figsize=(3, 7))
    initial_ticks = np.arange(0, max_value, 0.2)
    current_ticks = list(initial_ticks)
    current_labels = [f"{tick:.1f}" for tick in initial_ticks]

    # Boucle sur les sujets
    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):
        color = name_to_color[name_subject]
        up_line = max_value + up_subject

        for idx_trials in range(
                len(gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0])):
            gaze_position = \
                gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials]
            data = np.zeros(len(gaze_position), dtype=int)

            for idx_ligne, ligne in enumerate(gaze_position):
                if X[0][0] <= ligne[0] <= X[0][1] and Y[:, 1][0] <= ligne[1] <= Y[:, 1][1]:
                    data[idx_ligne] = 0
                else:
                    data[idx_ligne] = 1

            data = pd.DataFrame(data)
            data_norm = data.apply(lambda x: safe_interpolate(x, num_points))

            y_line_position = up_line
            y_values = np.full(len(data_norm[0]), np.nan)
            y_values[data_norm[0] == 0] = y_line_position
            ax.plot(y_values, '-', color=color, label='Presence of Zero' if idx_trials == 0 else "")

            if max_value + up_subject not in current_ticks:
                current_ticks.append(max_value + up_subject)
                current_labels.append(str(name_subject))
            current_ticks.sort()
            special_index = current_ticks.index(max_value + up_subject)
            current_labels[special_index] = str(name_subject)

            up_line += 0.0005

        plt.plot(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt][idx_subject], label=f'Subject {idx_subject + 1}',
                 color=color, linestyle='--')
        up_subject += 0.02

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels, fontsize=9)

    for label in ax.get_yticklabels():
        if label.get_text().replace('.', '', 1).isdigit():
            label.set_fontsize(10)
        else:
            label.set_fontsize(5)

    ax.set_title(f"Horizontal Line Indicating Presence of Zeros {mvt}")
    ax.set_xlabel("Index")
    ax.set_ylabel("Line Presence (Custom Y Position)")
    ax.set_xlim(0, len(data_norm[0]))
    ax.set_ylim(0, max_value + 0.45)

    # Ajout de la légende
    line1, = plt.plot([], [], color='black', label='Gaze on trampoline')
    line2, = plt.plot([], [], color='black', linestyle='--', label='SDtotal on pelvic rotation')
    plt.legend(handles=[line1, line2], fontsize='small')

    plt.title(f'{mvt}', fontsize=15)
    plt.xlabel('Time (%)')
    plt.ylabel('SD pelvic rotation')
    plt.savefig(f"/home/lim/Documents/StageMathieu/meeting/{mvt}_gaze.png")
    plt.subplots_adjust(left=0.11, right=0.957, top=0.937, bottom=0.082)
    # plt.show()


