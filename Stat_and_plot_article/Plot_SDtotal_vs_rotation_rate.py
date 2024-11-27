import pickle
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

file_path = "/home/lim/Documents/StageMathieu/Tab_result3/sd_pelvis_and_gaze_orientation.mat"
data_loaded = scipy.io.loadmat(file_path)
mean_SD_pelvis_all_subjects_acrobatics = data_loaded["mean_SD_pelvis_all_subjects_acrobatics"]
movement_to_analyse = data_loaded["movement_to_analyse"]
list_name_for_movement = data_loaded["list_name_for_movement"]
movement_to_analyse = np.char.strip(movement_to_analyse)

with open("/home/lim/Documents/StageMathieu/Tab_result3/pelvis_omega.pkl", 'rb') as f:
    pelvis_omega = pickle.load(f)

name_to_color = {
    '4-': '#1f77b4',
    '4-o': '#ff7f0e',
    '8--o': '#2ca02c',
    '8-1<': '#d62728',
    '8-1o': '#9467bd',
    '41': '#8c564b',
    '811<': '#e377c2',
    '41o': '#7f7f7f',
    '8-3<': '#bcbd22',
    '42': '#17becf',
    '822': '#aec7e8',
    '831<': '#ffbb78',
    '43': '#98df8a',
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

name_to_marker = {
    'GuSe': 's',
    'JaSh': 'o',
    'JeCa': 's',
    'AnBe': 'o',
    'AnSt': 'o',
    'SaBe': 's',
    'JoBu': 'o',
    'JaNo': 'o',
    'SaMi': 's',
    'AlLe': 's',
    'MaBo': 's',
    'SoMe': 's',
    'JeCh': 's',
    'LiDu': 'o',
    'LeJa': 'o',
    'ArMa': 'o',
    'AlAd': 's'
}

all_points_x = None
all_points_y = None
fig, ax = plt.subplots(figsize=(10, 10))
for idx_mvt, mvt in enumerate(movement_to_analyse):
    name_acro = full_name_acrobatics[mvt]

    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):
        color = name_to_color[mvt]
        marker = name_to_marker[name_subject]

        if pelvis_omega[mvt][name_subject] is not None:
            pelvis_omega_deg = pelvis_omega[mvt][name_subject][1:] * 180 / np.pi
            SD_pelvis_deg = mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt][idx_subject][1:] * 180/ np.pi
            plt.plot(pelvis_omega_deg, SD_pelvis_deg, color=color, marker=marker, markersize=0.5, linestyle='None')
            if all_points_x is None:
                all_points_x = pelvis_omega_deg
                all_points_y = SD_pelvis_deg
            else:
                all_points_x = np.hstack((all_points_x, pelvis_omega_deg))
                all_points_y = np.hstack((all_points_y, SD_pelvis_deg))

slope, intercept, r_value, p_value, std_err = linregress(all_points_x, all_points_y)
x_line = np.linspace(np.min(all_points_x), np.max(all_points_x))
y_line = slope * x_line + intercept
p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(10, 2, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))


plt.plot(x_line, y_line, '--', color='gray', linewidth=1.5, label='Linear regression')

plt.xlabel(r"Rotation rate ($^\circ$/s)")
plt.ylabel(r"Variability of pelvis orientation ($^\circ$)")
plt.legend()
plt.subplots_adjust(left=0.102, right=0.9, top=0.945, bottom=0.047)
plt.savefig(f"/home/lim/Documents/StageMathieu/pelvis_velocity_vs_SD/all.png", dpi=1000)
# plt.show()

print(f"*** Regression equation : {slope} x + {intercept} ***")
