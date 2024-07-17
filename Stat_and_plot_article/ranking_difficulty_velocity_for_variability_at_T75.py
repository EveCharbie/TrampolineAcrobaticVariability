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

# ## Velocity at T75
# velocity_at_T75 = {
#     '8-1o': 109,
#     '8-1<': 119,
#     '41': 88,
#     '811<': 127,
#     '41o': 82,
#     '8-3<': 204,
#     '42': 128,
#     '822': 181,
#     '831<': 320,
#     '43': 183,
# }

## Velocity at T75
mean_velocity_at_T75 = {
    '8-1o': 809,
    '8-1<': 822,
    '41': 445,
    '811<': 1011,
    '41o': 459,
    '8-3<': 1003,
    '42': 568,
    '822': 1239,
    '831<': 691,
    '43': 703,
}
std_velocity_at_T75 = {
    '8-1o': 25,
    '8-1<': 21,
    '41': 21,
    '811<': 45,
    '41o': 13,
    '8-3<': 35,
    '42': 27,
    '822': 31,
    '831<': 66,
    '43': 19,
}

velocity_values = np.array([mean_velocity_at_T75[key] for key in mean_velocity_at_T75.keys()])
sorted_order = np.argsort(velocity_values)

x_boxplot_top = velocity_values[sorted_order]
orderxlabeltop = list(list(mean_velocity_at_T75.keys())[idx] for idx in sorted_order)
boxplot_xerr = np.array([std_velocity_at_T75[key] for key in orderxlabeltop])

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

# x_reg_line = np.linspace(0, len(mean_velocity_at_T75)-1, 100)
x_reg_line = np.linspace(445, 1239, 100)
y_reg_line = slope * x_reg_line + intercept

plt.rc('font', size=14)          # Taille de la police du texte
plt.rc('axes', titlesize=16)     # Taille de la police du titre des axes
plt.rc('axes', labelsize=14)     # Taille de la police des labels des axes
plt.rc('xtick', labelsize=12)    # Taille de la police des labels des ticks en x
plt.rc('ytick', labelsize=12)    # Taille de la police des labels des ticks en y
plt.rc('legend', fontsize=12)    # Taille de la police de la légende

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=complete_data[order_and_ratio["Movement_Name"]], ax=ax, color="skyblue", positions=velocity_values, width=8)
plt.setp(ax.artists, edgecolor='k')
plt.setp(ax.lines, color='k')
variability_values = np.array([np.nanmedian(complete_data[key]) for key in order_and_ratio["Movement_Name"]])
plt.errorbar(velocity_values, variability_values, xerr=boxplot_xerr, linestyle="", capsize=1.5, color="k", elinewidth=0.8)
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', linestyle='--', label='Regression Line', linewidth=1.5)

p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xlabel('Rotation rate (°/s)', labelpad=15)
ax.set_ylabel('Variability of pelvis orientation at T$_{75}$ (°)')
ax.set_ylim(0, 60)
ax.legend(loc='lower right')

ax.set_xticks(np.arange(450, 1251, 100))
ax.set_xticklabels(np.arange(450, 1251, 100))

secax = ax.secondary_xaxis('top')
secax.set_xticks(x_boxplot_top)
orderxlabeltop = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " "]

secax.set_xticklabels(orderxlabeltop)
secax.set_xlabel('Acrobatics', labelpad=15)
secax.tick_params(axis='x', rotation=45)

plt.subplots_adjust(top=0.897, bottom=0.108, left=0.066, right=0.995)

plt.savefig("/home/lim/Documents/StageMathieu/meeting/linear_reg_all_acrobatics_with_velocity_scale.png", dpi=1000)
plt.show()

