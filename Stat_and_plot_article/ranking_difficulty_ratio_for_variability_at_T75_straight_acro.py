import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import linregress

home_path = "/Tab_result/"
x_boxplot_top = [0.5, 1, 1.5]
orderxlabeltop = ['0.5', '1', '1.5']

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

difficulty_values = [0.5, 1, 1.5]
difficulty_levels = np.concatenate([np.full(len(complete_data[col]), difficulty_values[i]) for i, col in enumerate(['41', '42', '43'])])


values = np.concatenate([complete_data[col] for col in ['41', '42', '43']])

slope, intercept, r_value, p_value, std_err = linregress(difficulty_levels, values)

x_reg_line = np.array([0.5, 1, 1.5])
y_reg_line = slope * x_reg_line + intercept

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=[complete_data['41'], complete_data['42'], complete_data['43']], ax=ax, color="skyblue", positions=[0.5, 1, 1.5], width=0.2)
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', label='Regression Line', linewidth=1.5)

p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xlabel('Acrobatics', labelpad=15)
ax.set_ylabel('Variability of pelvis rotations at T$_{75}$ (deg)')
ax.set_xticks([0.5, 1, 1.5])
ax.set_xticklabels(['41/', '42/', '43/'])
ax.legend(loc='lower right')

secax = ax.secondary_xaxis('top')
secax.set_xticks(x_boxplot_top)
secax.set_xticklabels(orderxlabeltop)
secax.set_xlabel('Ratio twists somersaults', labelpad=15)

plt.tight_layout()
plt.subplots_adjust(left=0.060, right=0.995, top=0.902, bottom=0.103)

plt.savefig("/home/lim/Documents/StageMathieu/meeting/75_with_difficulty.png", dpi=1000)
plt.show()

