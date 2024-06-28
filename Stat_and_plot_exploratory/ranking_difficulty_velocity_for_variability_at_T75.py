import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy.stats import linregress

home_path = "/Tab_result3/"

## Velocity at T75
velocity_at_T75 = {
    '8-1o': 109,
    '8-1<': 119,
    '41': 88,
    '811<': 127,
    '41o': 82,
    '8-3<': 204,
    '42': 128,
    '822': 181,
    '831<': 320,
    '43': 183,
}

sorted_velocity = dict(sorted(velocity_at_T75.items(), key=lambda item: item[1]))
order_and_ratio = pd.DataFrame(list(sorted_velocity.items()), columns=['Movement_Name', 'Velocity_at_T75'])

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
        all_x_positions.extend([velocity_at_T75[col]] * complete_data[col].dropna().shape[0])
        all_values.extend(complete_data[col].dropna().values)

slope, intercept, r_value, p_value, std_err = linregress(all_x_positions, all_values)

x_reg_line = np.linspace(min(order_and_ratio["Velocity_at_T75"]), max(order_and_ratio["Velocity_at_T75"]), 100)
y_reg_line = slope * x_reg_line + intercept

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(data=complete_data[order_and_ratio["Movement_Name"]], ax=ax, color="skyblue")

p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'r = {r_value:.2f}\n{p_text}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xlabel('Velocity at T75', labelpad=15)
ax.set_ylabel('Variability of pelvis rotations at T$_{75}$ (deg)')
ax.set_ylim(0, 58)
ax.legend(loc='lower right')

plt.subplots_adjust(top=0.907, bottom=0.098, left=0.056, right=0.995)

plt.savefig("/home/lim/Documents/StageMathieu/meeting/linear_reg_all_acrobatics_with_velocity.png", dpi=1000)
plt.show()

