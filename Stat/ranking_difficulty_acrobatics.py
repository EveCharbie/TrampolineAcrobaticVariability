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

# complete_data = []
for file in files:
    data = pd.read_csv(file)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID
    anova_rot_df = data.pivot_table(index=['ID', 'Expertise'], columns='Timing', values='Std')
    complete_data[mvt_name] = anova_rot_df["75%"]


# 1. Boxplots for the different difficulty levels without expertise distinction
x_boxplot_centers = [0, 1, 2]

means = [complete_data[col].mean() for col in ['41', '42', '43']]
slope, intercept, r_value, p_value, std_err = linregress(x_boxplot_centers, means)

x_reg_line = np.array(x_boxplot_centers)
y_reg_line = slope * x_reg_line + intercept

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=complete_data[['41', '42', '43']], ax=ax, color="skyblue")
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', label='Regression Line', linewidth=1.5)

text_str = f'R-squared: {r_value**2:.2f}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_title('Boxplot with Regression Line for Different Difficulty Level')
ax.set_xlabel('Acrobatics by Difficulty Level')
ax.set_ylabel('75%')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['41', '42', '43'])
ax.legend(loc='lower right')

plt.show()

correlation = complete_data[['41', '42', '43']].corr()
print(correlation)





