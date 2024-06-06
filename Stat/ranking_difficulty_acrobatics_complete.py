import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene
import os
import numpy as np
from scipy.stats import linregress, mannwhitneyu

from scipy.stats import kruskal
import scikit_posthocs as sp
from TrampolineAcrobaticVariability.Function.Function_stat import (perform_anova_and_tukey,
                                                                   perform_kruskal_and_dunn,
                                                                   prepare_data)

home_path = "/home/lim/Documents/StageMathieu/Tab_result/"

order = ['8-1o', '8-1<', '811<', '41', '41o', '8-3<', '42', '831<', '822', '43']

orderxlabel = ['8-1o', '8-1<', '811<', '41/', '41o', '8-3<', '42/', '831<', '822/', '43/']


ratio_twist_somersault = {
    '8-1o': '#1f77b4',
    '8-1<': '#ff7f0e',
    '41': '#2ca02c',
    '811<': '#ff7f0e',
    '41o': '#1f77b4',
    '8-3<': '#ff7f0e',
    '42': '#2ca02c',
    '822': '#2ca02c',
    '831<': '#ff7f0e',
    '43': '#2ca02c',
}

ratio = [0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 1, 1, 1, 1.5]

name = [
    'GuSe',
    'JaSh',
    'JeCa',
    'AnBe',
    'AnSt',
    'SaBe',
    'JoBu',
    'JaNo',
    'SaMi',
    'AlLe',
    'MaBo',
    'SoMe',
    'JeCh',
    'LiDu',
    'LeJa',
    'ArMa',
    'AlAd'
]

rotation_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)

complete_data = pd.DataFrame(columns=order, index=name)

# complete_data = []
for file in rotation_files:
    data = pd.read_csv(file)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')
    anova_rot_df = data.pivot_table(index=['ID'], columns='Timing', values='Std')

    for name in (data["ID"].unique()):

        complete_data.loc[name, mvt_name] = anova_rot_df.loc[name, "75%"]



# 1. Boxplots for the different difficulty levels without expertise distinction
x_boxplot_centers = [0, 1, 4, 5, 6, 9, 12, 13, 14, 17]

ratio = [0.5, 0.5, 5, 5, 5, 9, 13, 13, 13, 17]


means = [complete_data[col].mean() for col in order]
slope, intercept, r_value, p_value, std_err = linregress(ratio, means)

x_reg_line = np.array(x_boxplot_centers)
y_reg_line = slope * x_reg_line + intercept

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=complete_data[order], ax=ax, color="skyblue", positions=x_boxplot_centers)
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', label='Regression Line', linewidth=1.5)

text_str = f'R-squared: {r_value**2:.2f}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# ax.set_title('Boxplot with Regression Line for Different Difficulty Level')
ax.set_xlabel('Acrobatics by Difficulty Level')
ax.set_ylabel('Variability of pelvis rotations at T$_{75}$ (rad)')
ax.set_xticks(x_boxplot_centers)
ax.set_xticklabels(orderxlabel)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("/home/lim/Documents/StageMathieu/meeting/reg_all.png", dpi=1000)

plt.show()
print(p_value)


correlation = complete_data[order].corr()
print(correlation)

order = ['8-1o', '8-1<', '811<', '41', '41o', '8-3<', '42', '831<', '822', '43']
orderxlabel = ['8-1o', '8-1<', '811<', '41/', '41o', '8-3<', '42/', '831<', '822/', '43/']

ratio = [0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 1, 1, 1, 1.5]
x_boxplot_centers = [0, 1, 4, 5, 6, 9, 12, 13, 14, 17]

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

complete_data = pd.DataFrame(columns=order, index=name)

for file in rotation_files:
    data = pd.read_csv(file)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')
    anova_rot_df = data.pivot_table(index=['ID'], columns='Timing', values='Std')
    for gymnast_name in data["ID"].unique():
        complete_data.loc[gymnast_name, mvt_name] = anova_rot_df.loc[gymnast_name, "75%"]

# Prepare data for regression using x_boxplot_centers
all_x_positions = []
all_values = []

for i, col in enumerate(order):
    all_x_positions.extend([x_boxplot_centers[i]] * complete_data[col].dropna().shape[0])
    all_values.extend(complete_data[col].dropna().values)

# Perform linear regression using x_boxplot_centers and all values
slope, intercept, r_value, p_value, std_err = linregress(all_x_positions, all_values)

# Create the regression line
x_reg_line = np.linspace(min(x_boxplot_centers), max(x_boxplot_centers), 100)
y_reg_line = slope * x_reg_line + intercept

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=complete_data[order], ax=ax, color="skyblue", positions=x_boxplot_centers)
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', label='Regression Line', linewidth=1.5)

# Add R-squared text
p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.2f}"
text_str = f'R-squared: {r_value**2:.2f}\n{p_text}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Set labels and title
ax.set_xlabel('Acrobatics by Difficulty Level')
ax.set_ylabel('Variability of pelvis rotations at T$_{75}$ (rad)')
ax.set_xticks(x_boxplot_centers)
ax.set_xticklabels(orderxlabel)
ax.set_ylim(0.1, 0.98)
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("/home/lim/Documents/StageMathieu/meeting/reg_all.png", dpi=1000)

plt.show()

print(p_value)
