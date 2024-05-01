import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu
import numpy as np
import matplotlib.patches as mpatches


movement_to_analyse = ['4-', '4-o', '8--o', '8-1o', '8-1<', '811<', '41', '41o', '8-3<', '42', '831<', '822', '43']

# movement_to_analyse = ['41', '41o', '811<', '42', '822', '831<', '43']
# movement_to_analyse = ['41', '42', '43']

data = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result/results_area_under_curve2.csv')
combined_data = pd.melt(data, id_vars=['ID', 'Expertise'], value_vars=movement_to_analyse, var_name='Difficulty', value_name='Score')


# 1. Boxplots for the different difficulty levels without expertise distinction
x_boxplot_centers = list(range(len(movement_to_analyse)))

means = [data[col].mean() for col in movement_to_analyse]
slope, intercept, r_value, p_value, std_err = linregress(x_boxplot_centers, means)

x_reg_line = np.array(x_boxplot_centers)
y_reg_line = slope * x_reg_line + intercept

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=data[movement_to_analyse], ax=ax, color="skyblue")
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', label='Regression Line', linewidth=1.5)

text_str = f'Intercept: {intercept:.2f}\nR-squared: {r_value**2:.2f}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_title('Boxplot with Regression Line for Each Difficulty Level')
ax.set_xlabel('Difficulty Level')
ax.set_ylabel('Area Under Curve')
ax.set_xticks(x_boxplot_centers)
ax.set_xticklabels(movement_to_analyse)
ax.legend(loc='lower right')

plt.show()

correlation = data[movement_to_analyse].corr()
print(correlation)


