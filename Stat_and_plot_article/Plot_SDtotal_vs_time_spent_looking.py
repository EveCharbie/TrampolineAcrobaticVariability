import pickle
import scipy.io
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import linregress

file_path = "/home/lim/Documents/StageMathieu/Tab_result3/sd_pelvis_and_gaze_orientation.mat"
data_loaded = scipy.io.loadmat(file_path)
mean_SD_pelvis_all_subjects_acrobatics = data_loaded["mean_SD_pelvis_all_subjects_acrobatics"]
movement_to_analyse = data_loaded["movement_to_analyse"]
list_name_for_movement = data_loaded["list_name_for_movement"]
movement_to_analyse = np.char.strip(movement_to_analyse)

with open("/home/lim/Documents/StageMathieu/Tab_result3/time_spent_looking.pkl", 'rb') as f:
    time_spent_looking, mean_time_spent_looking, std_time_spent_looking = pickle.load(f)

with open("/home/lim/Documents/StageMathieu/Tab_result3/pelvis_SD_T75.pkl", "rb") as f:
    sd_pelvis_T75 = pickle.load(f)

mvt_to_color = {
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

x_boxplot_top = []
all_points_x = None
all_points_y = None
fig, ax = plt.subplots(figsize=(10, 6))
min_lineregress = 1000
max_lineregress = -1
for idx_mvt, mvt in enumerate(full_name_acrobatics.keys()):
    name_acro = full_name_acrobatics[mvt]

    time_spent_looking_this_mvt = []
    for idx_subject, name_subject in enumerate(mean_time_spent_looking["gymnasium_floor"][mvt].keys()):
        this_data = mean_time_spent_looking["gymnasium_floor"][mvt][name_subject] * 100
        time_spent_looking_this_mvt += [this_data]

    mean_time_spent_looking_this_mvt = np.mean(np.array(time_spent_looking_this_mvt))
    std_time_spent_looking_this_mvt = np.std(np.array(time_spent_looking_this_mvt))
    ax.errorbar(mean_time_spent_looking_this_mvt, np.nanmedian(np.array(sd_pelvis_T75[mvt])),
                xerr=std_time_spent_looking_this_mvt,
                linestyle="", capsize=2, color="k", linewidth=0.5)
    sns.boxplot(data=sd_pelvis_T75[mvt],
                ax=ax,
                color=mvt_to_color[mvt],
                positions=[mean_time_spent_looking_this_mvt],
                width=1,
                linecolor='k',
                linewidth=0.5)
    x_boxplot_top += [mean_time_spent_looking_this_mvt]

    if mean_time_spent_looking_this_mvt < min_lineregress:
        min_lineregress = mean_time_spent_looking_this_mvt
    if mean_time_spent_looking_this_mvt > max_lineregress:
        max_lineregress = mean_time_spent_looking_this_mvt

    if all_points_x is None:
        all_points_x = np.array(time_spent_looking_this_mvt)
        all_points_y = np.array(sd_pelvis_T75[mvt])[np.where(np.isfinite(np.array(sd_pelvis_T75[mvt], dtype=np.float64)))[0]]
    else:
        all_points_x = np.hstack((all_points_x, np.array(time_spent_looking_this_mvt)))
        all_points_y = np.hstack((all_points_y, np.array(sd_pelvis_T75[mvt])[np.where(np.isfinite(np.array(sd_pelvis_T75[mvt], dtype=np.float64)))[0]]))

points_x_array = np.zeros(all_points_x.shape)
points_y_array = np.zeros(all_points_y.shape)
for i in range(all_points_x.shape[0]):
    points_x_array[i] = all_points_x[i]
    points_y_array[i] = all_points_y[i]

slope, intercept, r_value, p_value, std_err = linregress(points_x_array, points_y_array)
x_line = np.linspace(min_lineregress, max_lineregress)
y_line = slope * x_line + intercept
p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(10, 2, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.plot(x_line, y_line, '--', color='k', linewidth=2, label='Linear regression')

secax = ax.secondary_xaxis('top')
secax.set_xticks(x_boxplot_top)

secax.set_xticklabels([full_name_acrobatics[mvt] for mvt in full_name_acrobatics.keys()])
secax.set_xlabel('Acrobatics', labelpad=15)
secax.tick_params(axis='x', rotation=45)

ax.set_xlabel("Time spent looking at the gymnasium floor (%)")
ax.set_ylabel(r"Variability of pelvis orientation ($^\circ$)")
ax.legend()

plt.xticks([40, 50, 60, 70, 80, 90, 100], ['40', '50', '60', '70', '80', '90', '100'])

plt.subplots_adjust(left=0.102, right=0.9, top=0.9, bottom=0.1)
plt.savefig(f"/home/lim/Documents/StageMathieu/time_spent_looking_vs_pelvis_SD.svg", format="svg")
plt.show()

print(f"*** Regression equation : {slope} x + {intercept} ***")
