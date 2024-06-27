import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu
import numpy as np
import matplotlib.patches as mpatches

data = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_area_under_curve2.csv')
combined_data = pd.melt(data, id_vars=['ID', 'Expertise'], value_vars=['41', '42', '43'], var_name='Difficulty', value_name='Score')

x_boxplot_centers = [0, 1, 2]

means = [data[col].mean() for col in ['41', '42', '43']]
slope, intercept, r_value, p_value, std_err = linregress(x_boxplot_centers, means)

x_reg_line = np.array(x_boxplot_centers)
y_reg_line = slope * x_reg_line + intercept

fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=data[['41', '42', '43']], ax=ax, color="skyblue")
sns.lineplot(x=x_reg_line, y=y_reg_line, ax=ax, color='gray', label='Regression Line', linewidth=1.5)

text_str = f'Intercept: {intercept:.2f}\nR-squared: {r_value**2:.2f}'
ax.text(0.02, 0.95, text_str, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_title('Boxplot with Regression Line for Each Difficulty Level')
ax.set_xlabel('Difficulty Level')
ax.set_ylabel('Area Under Curve')
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(['41', '42', '43'])
ax.legend(loc='lower right')

plt.show()

correlation_elite = data[data['Expertise'] == 'Elite'][['41', '42', '43']].corr(method='pearson')
correlation_subelite = data[data['Expertise'] == 'SubElite'][['41', '42', '43']].corr(method='pearson')

elite_data = combined_data[combined_data['Expertise'] == 'Elite']
subelite_data = combined_data[combined_data['Expertise'] == 'SubElite']

levels = ['41', '42', '43']

positions_elite = [1, 5, 9]
positions_subelite = [2, 6, 10]

fig, ax = plt.subplots(figsize=(12, 8))

colors = {'Elite': 'lightblue', 'SubElite': 'lightgreen'}
median_color = 'black'

for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    ax.boxplot(data_elite, positions=[positions_elite[i]], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors['Elite']),
               medianprops=dict(color=median_color))

for i, level in enumerate(levels):
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    ax.boxplot(data_subelite, positions=[positions_subelite[i]], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors['SubElite']),
               medianprops=dict(color=median_color))

for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    stat, pvalue = mannwhitneyu(data_elite, data_subelite)
    y_max = max(data_elite.max(), data_subelite.max()) + 5

    p_text = f"p < 0.001" if pvalue < 0.001 else f"p = {pvalue:.3f}"

    mid_point = (positions_elite[i] + positions_subelite[i]) / 2
    line_y = y_max + 1
    ax.hlines(y=line_y, xmin=positions_elite[i], xmax=positions_subelite[i], colors="black", linestyles='solid', lw=1)
    ax.vlines(x=positions_elite[i], ymin=line_y - 0.5, ymax=line_y + 0.5, colors="black", linestyles='solid', lw=1)
    ax.vlines(x=positions_subelite[i], ymin=line_y - 0.5, ymax=line_y + 0.5, colors="black", linestyles='solid', lw=1)

    ax.annotate(p_text, xy=(mid_point, line_y + 0.5), textcoords="offset points", xytext=(0, 5), ha='center')

legend_patches = [mpatches.Patch(color=colors['Elite'], label='Elite'),
                  mpatches.Patch(color=colors['SubElite'], label='SubElite')]
ax.legend(handles=legend_patches, title='Expertise', loc='lower right')
ax.set_ylim(15, 70)
ax.set_xticks([1.5, 5.5, 9.5])
ax.set_xticklabels(['41', '42', '43'])
ax.set_title('Comparison of AUC by Expertise')
ax.set_xlabel('Difficulty Level')
ax.set_ylabel('AUC')

plt.show()

