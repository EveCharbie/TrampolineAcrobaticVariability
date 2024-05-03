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

X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])

print(liste_name)
num_points = 100

for idx_mvt, mvt in enumerate(movement_to_analyse):

    up_subject = 0.1
    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
                     '#c49c94', '#dbdb8d']

    colors = custom_colors[:len(wall_index_all_subjects_acrobatics[0][idx_mvt][0])]
    max_value = np.max(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt].flatten())
    fig, ax = plt.subplots(figsize=(10, 10))

    for idx_subject in range(len(wall_index_all_subjects_acrobatics[0][idx_mvt][0])):
        color = colors[idx_subject]
        up_line = max_value + up_subject

        for idx_trials in range(len(gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0])):
            gaze_position = gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials]
            ##
            data = np.zeros(len(gaze_position), dtype=int)

            for idx_ligne, ligne in enumerate(gaze_position):
                if X[0][0] <= ligne[0] <= X[0][1] and \
                        Y[:, 1][0] <= ligne[1] <= Y[:, 1][1]:
                    data[idx_ligne] = 0
                else:
                    data[idx_ligne] = 1

            ##


            # data = wall_index_all_subjects_acrobatics[0][i][0][j][0][k]
            data = pd.DataFrame(data)
            data_norm = data.apply(lambda x: safe_interpolate(x, num_points))

            # Choose the y-coordinate where the line will be plotted
            y_line_position = up_line

            # Create an array to hold the y-values for the plot line
            y_values = np.full(len(data_norm[0]), np.nan)

            # Set y-values to the chosen y-coordinate where the data is zero
            y_values[data_norm[0] == 0] = y_line_position

            # Plot the line with the color obtained for this iteration of j
            ax.plot(y_values, '-', color=color, label='Presence of Zero')

            ax.set_title(f"Horizontal Line Indicating Presence of Zeros {mvt}")
            ax.set_xlabel("Index")
            ax.set_ylabel("Line Presence (Custom Y Position)")
            ax.set_xlim(0, len(data_norm[0]))
            ax.set_ylim(0, max_value+0.45)

            up_line += 0.0005
        plt.plot(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt][idx_subject], label=f'Subject {idx_subject + 1}', color=color)
        up_subject += 0.02

    plt.title(f'Data for {mvt} Acrobatic Movement')
    plt.xlabel('Data Points')
    plt.ylabel('Values')
    plt.show()




