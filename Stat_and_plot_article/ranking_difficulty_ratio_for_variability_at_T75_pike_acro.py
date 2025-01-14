import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import linregress

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

result_array = np.array(result_df)
mean_velocity_at_T75 = {
    '8-1<':   result_array[np.where(result_array[:, 2] == '8-1<')[0][0], 0],
    '811<':   result_array[np.where(result_array[:, 2] == '811<')[0][0], 0],
    '8-3<':   result_array[np.where(result_array[:, 2] == '8-3<')[0][0], 0],
    '831<':   result_array[np.where(result_array[:, 2] == '831<')[0][0], 0],
}

std_velocity_at_T75 = {
    '8-1<':   result_array[np.where(result_array[:, 2] == '8-1<')[0][0], 1],
    '811<':   result_array[np.where(result_array[:, 2] == '811<')[0][0], 1],
    '8-3<':   result_array[np.where(result_array[:, 2] == '8-3<')[0][0], 1],
    '831<':   result_array[np.where(result_array[:, 2] == '831<')[0][0], 1],
}

velocity_values = np.array([mean_velocity_at_T75[key] for key in mean_velocity_at_T75.keys()])
sorted_order = np.argsort(velocity_values)

x_boxplot_top = velocity_values[sorted_order]
orderxlabeltop = list(list(mean_velocity_at_T75.keys())[idx] for idx in sorted_order)
boxplot_xerr = np.array([std_velocity_at_T75[key] for key in orderxlabeltop])

sorted_velocity = dict(sorted(mean_velocity_at_T75.items(), key=lambda item: item[1]))
order_and_ratio = pd.DataFrame(list(sorted_velocity.items()), columns=['Movement_Name', 'Mean Velocity_at_T75'])


rotation_files = []
for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)

files = [
    '/home/lim/Documents/StageMathieu/Tab_result3/results_8-1<_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result3/results_811<_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result3/results_8-3<_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result3/results_831<_rotation.csv'
]

complete_data = pd.DataFrame(columns=['8-1<', '811<', '8-3<', '831<'])

for file in files:
    data = pd.read_csv(file)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID
    anova_rot_df = data.pivot_table(index=['ID', 'Expertise'], columns='Timing', values='Std')
    complete_data[mvt_name] = np.degrees(anova_rot_df["75%"])

complete_data = complete_data.dropna()

values = np.concatenate([complete_data[col] for col in ['8-1<', '811<', '8-3<', '831<']])


plt.rc('font', size=14)          # Taille de la police du texte
plt.rc('axes', titlesize=16)     # Taille de la police du titre des axes
plt.rc('axes', labelsize=14)     # Taille de la police des labels des axes
plt.rc('xtick', labelsize=12)    # Taille de la police des labels des ticks en x
plt.rc('ytick', labelsize=12)    # Taille de la police des labels des ticks en y
plt.rc('legend', fontsize=12)    # Taille de la police de la légende

fig, ax = plt.subplots(figsize=(10, 6))
for i, mvt in enumerate(['8-1<', '811<', '8-3<', '831<']):
    sns.boxplot(data=[complete_data[mvt]],
                ax=ax,
                color=mvt_to_color[mvt],
                positions=[velocity_values[i]],
                width=8,
                linecolor='k')

all_x_positions = []
all_values = []
for i, col in enumerate(['8-1<', '811<', '8-3<', '831<']):
    if col in complete_data.columns:
        all_x_positions.extend([mean_velocity_at_T75[col]] * complete_data[col].dropna().shape[0])
        all_values.extend(complete_data[col].dropna().values)

variability_values = np.array([np.nanmedian(complete_data['8-1<']),
                               np.nanmedian(complete_data['811<']),
                               np.nanmedian(complete_data['8-3<']),
                               np.nanmedian(complete_data['831<'])])
print("Median 8-1< : ", np.nanmedian(complete_data['8-1<']))
print("Median 811< : ", np.nanmedian(complete_data['811<']))
print("Median 8-3< : ", np.nanmedian(complete_data['8-3<']))
print("Median 831< : ", np.nanmedian(complete_data['831<']))


slope, intercept, r_value, p_value, std_err = linregress(all_x_positions, all_values)

x_reg_line = np.linspace(np.min(velocity_values), np.max(velocity_values))
y_reg_line = slope * x_reg_line + intercept
print(f"*** Straight acrobatics regression equation : {slope} x + {intercept} ***")

plt.errorbar(x_boxplot_top, variability_values, xerr=boxplot_xerr, linestyle="", capsize=5, color="k", elinewidth=0.8)
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', linestyle="--", label='Regression Line', linewidth=2)

p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xticks(np.arange(450, 900, 100))
ax.set_xticklabels(np.arange(450, 900, 100))

ax.set_xlabel('Rotation rate (°/s)', labelpad=15)
ax.set_ylabel('Variability of pelvis orientation at T$_{75}$ (°)')
ax.legend(loc='lower right')

secax = ax.secondary_xaxis('top')
secax.set_xticks(x_boxplot_top)
secax.set_xticklabels(['8-1<', '811<', '8-3<', '831<'])
secax.set_xlabel('Acrobatics', labelpad=15)
secax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.872, bottom=0.108, left=0.066, right=0.995)

plt.savefig("/home/lim/Documents/StageMathieu/meeting/T75_variability_pike_acrobatics.svg", format="svg")
plt.show()

