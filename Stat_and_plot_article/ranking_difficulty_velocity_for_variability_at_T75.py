import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import linregress


#######################################################################################################################
### !!!!!! Warning: the pandas DataFrames and the sorted order are redundant notation and are used interchangably. ####
### !!!!!! If the order is changed the resluts might be unexpected !!!!!! #############################################
#######################################################################################################################


home_path = "/home/lim/Documents/StageMathieu/Tab_result3"
with open("/home/lim/Documents/StageMathieu/Tab_result3/pelvis_omega.pkl", "rb") as f:
    pelvis_omega, result_df = pickle.load(f)

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

result_array = np.array(result_df)
# ## Velocity at T75
mean_velocity_at_T75 = {
    '8-1o': result_array[np.where(result_array[:, 2] == '8-1o')[0][0], 0],
    '8-1<': result_array[np.where(result_array[:, 2] == '8-1<')[0][0], 0],
    '41':   result_array[np.where(result_array[:, 2] == '41')[0][0], 0],
    '811<': result_array[np.where(result_array[:, 2] == '811<')[0][0], 0],
    '41o':  result_array[np.where(result_array[:, 2] == '41o')[0][0], 0],
    '8-3<': result_array[np.where(result_array[:, 2] == '8-3<')[0][0], 0],
    '42':   result_array[np.where(result_array[:, 2] == '42')[0][0], 0],
    '822':  result_array[np.where(result_array[:, 2] == '822')[0][0], 0],
    '831<': result_array[np.where(result_array[:, 2] == '831<')[0][0], 0],
    '43':   result_array[np.where(result_array[:, 2] == '43')[0][0], 0],
}
# mean_velocity_at_T75 = {
#     '8-1o': 809,
#     '8-1<': 822,
#     '41': 445,
#     '811<': 1011,
#     '41o': 459,
#     '8-3<': 1003,
#     '42': 568,
#     '822': 1239,
#     '831<': 691,
#     '43': 703,
# }
std_velocity_at_T75 = {
    '8-1o': result_array[np.where(result_array[:, 2] == '8-1o')[0][0], 1],
    '8-1<': result_array[np.where(result_array[:, 2] == '8-1<')[0][0], 1],
    '41':   result_array[np.where(result_array[:, 2] == '41')[0][0], 1],
    '811<': result_array[np.where(result_array[:, 2] == '811<')[0][0], 1],
    '41o':  result_array[np.where(result_array[:, 2] == '41o')[0][0], 1],
    '8-3<': result_array[np.where(result_array[:, 2] == '8-3<')[0][0], 1],
    '42':   result_array[np.where(result_array[:, 2] == '42')[0][0], 1],
    '822':  result_array[np.where(result_array[:, 2] == '822')[0][0], 1],
    '831<': result_array[np.where(result_array[:, 2] == '831<')[0][0], 1],
    '43':   result_array[np.where(result_array[:, 2] == '43')[0][0], 1],
}
# std_velocity_at_T75 = {
#     '8-1o': 25,
#     '8-1<': 21,
#     '41': 21,
#     '811<': 45,
#     '41o': 13,
#     '8-3<': 35,
#     '42': 27,
#     '822': 31,
#     '831<': 66,
#     '43': 19,
# }


sorted_velocity = dict(sorted(mean_velocity_at_T75.items(), key=lambda item: item[1]))
order_and_ratio = pd.DataFrame(list(sorted_velocity.items()), columns=['Movement_Name', 'Mean Velocity_at_T75'])

name = [
    'GuSe', 'JaSh', 'JeCa', 'AnBe', 'AnSt', 'SaBe', 'JoBu',
    'JaNo', 'SaMi', 'AlLe', 'MaBo', 'SoMe', 'JeCh', 'LiDu',
    'LeJa', 'ArMa', 'AlAd'
]

rotation_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)

complete_data = pd.DataFrame(columns=order_and_ratio["Movement_Name"], index=name)

for file in rotation_files:
    data = pd.read_csv(file)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')
    anova_rot_df = data.pivot_table(index=['ID'], columns='Timing', values='Std')
    for gymnast_name in data["ID"].unique():
        complete_data.loc[gymnast_name, mvt_name] = np.degrees(anova_rot_df.loc[gymnast_name, "75%"])

all_x_positions = []
all_values = []

for i, col in enumerate(order_and_ratio["Movement_Name"]):
    if col in complete_data.columns:
        all_x_positions.extend([mean_velocity_at_T75[col]] * complete_data[col].dropna().shape[0])
        all_values.extend(complete_data[col].dropna().values)

slope, intercept, r_value, p_value, std_err = linregress(all_x_positions, all_values)

x_reg_line = np.linspace(np.min(result_array[:, 0]), np.max(result_array[:, 0]), 100)
y_reg_line = slope * x_reg_line + intercept
print(f"*** All acrobatics regression equation : {slope} x + {intercept} ***")

plt.rc('font', size=14)          # Taille de la police du texte
plt.rc('axes', titlesize=16)     # Taille de la police du titre des axes
plt.rc('axes', labelsize=14)     # Taille de la police des labels des axes
plt.rc('xtick', labelsize=12)    # Taille de la police des labels des ticks en x
plt.rc('ytick', labelsize=12)    # Taille de la police des labels des ticks en y
plt.rc('legend', fontsize=12)    # Taille de la police de la légende

fig, ax = plt.subplots(figsize=(10, 6))
for i_mvt, mvt in enumerate(list(order_and_ratio["Movement_Name"])):
    sns.boxplot(data=[complete_data[mvt]],
                ax=ax,
                color=mvt_to_color[mvt],
                positions=[mean_velocity_at_T75[mvt]],
                width=8,
                linecolor='k')
    plt.errorbar(mean_velocity_at_T75[mvt], np.nanmedian(complete_data[mvt]), xerr=std_velocity_at_T75[mvt], linestyle="", capsize=1.5, color="k",
                 elinewidth=0.8)

ax.plot(x_reg_line, y_reg_line, color='gray', linestyle='--', label='Regression Line', linewidth=1.5)

p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xlabel('Rotation rate (°/s)', labelpad=15)
ax.set_ylabel('Variability of pelvis orientation at T$_{75}$ (°)')
ax.set_ylim(0, 60)
ax.legend(loc='lower right')

velocity_values = np.array([mean_velocity_at_T75[key] for key in mean_velocity_at_T75.keys()])
sorted_order = np.argsort(velocity_values)
x_boxplot_top = velocity_values[sorted_order]
secax = ax.secondary_xaxis('top')
secax.set_xticks(x_boxplot_top)

secax.set_xticklabels([full_name_acrobatics[order_and_ratio["Movement_Name"][i]]][0] for i in range(10))
secax.set_xlabel('Acrobatics', labelpad=15)
secax.tick_params(axis='x', rotation=45)

plt.xticks([i for i in np.arange(250, 900, 50)], [str(i) for i in np.arange(250, 900, 50)])
plt.subplots_adjust(top=0.897, bottom=0.108, left=0.066, right=0.995)

plt.savefig("/home/lim/Documents/StageMathieu/meeting/linear_reg_all_acrobatics_with_velocity_scale.svg", format="svg")
plt.show()

with open("/home/lim/Documents/StageMathieu/Tab_result3/pelvis_SD_T75.pkl", "wb") as f:
    pickle.dump(complete_data, f)
