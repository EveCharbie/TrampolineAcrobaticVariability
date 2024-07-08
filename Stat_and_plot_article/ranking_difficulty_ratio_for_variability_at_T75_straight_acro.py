import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import linregress

home_path = "/Tab_result/"
x_boxplot_top = [445, 568, 703]
orderxlabeltop = ['41/', '42/', '43/']

rotation_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)

files = [
    '/home/lim/Documents/StageMathieu/Tab_result/results_41_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_42_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_43_rotation.csv'
]

complete_data = pd.DataFrame(columns=['41', '42', '43'])

for file in files:
    data = pd.read_csv(file)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID
    anova_rot_df = data.pivot_table(index=['ID', 'Expertise'], columns='Timing', values='Std')
    complete_data[mvt_name] = np.degrees(anova_rot_df["75%"])

complete_data = complete_data.dropna()

difficulty_values = [445, 568, 703]
difficulty_levels = np.concatenate([np.full(len(complete_data[col]), difficulty_values[i]) for i, col in enumerate(['41', '42', '43'])])


values = np.concatenate([complete_data[col] for col in ['41', '42', '43']])

slope, intercept, r_value, p_value, std_err = linregress(difficulty_levels, values)

x_reg_line = np.array([445, 568, 703])
y_reg_line = slope * x_reg_line + intercept

plt.rc('font', size=14)          # Taille de la police du texte
plt.rc('axes', titlesize=16)     # Taille de la police du titre des axes
plt.rc('axes', labelsize=14)     # Taille de la police des labels des axes
plt.rc('xtick', labelsize=12)    # Taille de la police des labels des ticks en x
plt.rc('ytick', labelsize=12)    # Taille de la police des labels des ticks en y
plt.rc('legend', fontsize=12)    # Taille de la police de la légende

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=[complete_data['41'], complete_data['42'], complete_data['43']], ax=ax, color="skyblue", positions=[445, 568, 703], width=50)
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', label='Regression Line', linewidth=1.5)

p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xticks(np.arange(400, 800, 100))
ax.set_xticklabels(np.arange(400, 800, 100))

ax.set_xlabel('Rotation rate (°/s)', labelpad=15)
ax.set_ylabel('Variability of pelvis orientation at T$_{75}$ (°)')
# ax.set_xticks([445, 568, 703])
# ax.set_xticklabels(['41/', '42/', '43/'])
ax.legend(loc='lower right')

secax = ax.secondary_xaxis('top')
secax.set_xticks(x_boxplot_top)
secax.set_xticklabels(orderxlabeltop)
secax.set_xlabel('Acrobatics', labelpad=15)
secax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.872, bottom=0.108, left=0.066, right=0.995)

plt.savefig("/home/lim/Documents/StageMathieu/meeting/75_with_difficulty.png", dpi=1000)
plt.show()

