import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, shapiro

from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm


def prepare_data(data):
    upper_body_columns = data[['AvBrasD', 'MainD', 'AvBrasG', 'MainG']]
    data["upper_body"] = upper_body_columns.mean(axis=1)
    lower_body_columns = data[['JambeD', 'PiedD', 'JambeG', 'PiedG']]
    data["lower_body"] = lower_body_columns.mean(axis=1)
    return data[['Expertise', 'Timing', 'upper_body', 'lower_body']]


files = [
    '/home/lim/Documents/StageMathieu/Tab_result/results_41_position.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_41o_position.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_42_position.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_43_position.csv'
]

all_axes = []

num_axes = 0

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for file in files:
    if num_axes >= 4:
        break

    data = pd.read_csv(file)
    data_prepared = prepare_data(data)
    data_prepared = data[data['Timing'] != 'Other'].copy()
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_position.csv', '')  # Clean file ID
    data_prepared['Source'] = mvt_name

    print(f"===== Movement {mvt_name} is running =====")

    measurements = ['upper_body', 'lower_body']
    for measure in measurements:
        for expertise in data_prepared['Expertise'].unique():
            for timing in data_prepared['Timing'].unique():
                group_data = data_prepared[(data_prepared['Expertise'] == expertise)
                                           & (data_prepared['Timing'] == timing)][measure]
                stat, p_value = shapiro(group_data)
                if p_value < 0.05:
                    print(f"Normality issue in {timing}, {expertise}, {measure} of {file} (P-value: {p_value:.4f})")

                groups_data = [
                    data_prepared[(data_prepared['Expertise'] == exp) & (data_prepared['Timing'] == tim)][measure] for
                    exp in data_prepared['Expertise'].unique() for tim in data_prepared['Timing'].unique()]
                levene_stat, levene_p = levene(*groups_data)
                if levene_p < 0.05:
                    print(f"Variance homogeneity issue across groups in {file} (P-value: {levene_p:.4f})")

    # Parametric test ANOVA for upper_body
    modele_upper = ols("upper_body ~ C(Expertise) * C(Timing)", data=data_prepared).fit()
    result_anova_upper = anova_lm(modele_upper, typ=2)
    print(result_anova_upper)

    # Parametric test ANOVA for lower_body
    modele_lower = ols("lower_body ~ C(Expertise) * C(Timing)", data=data_prepared).fit()
    result_anova_lower = anova_lm(modele_lower, typ=2)
    print(result_anova_lower)

    # Parametric test MANOVA for upper and lower body
    maov = MANOVA.from_formula('upper_body + lower_body ~ C(Expertise) * C(Timing)', data=data_prepared)
    print(maov.mv_test())

    palette_upper = sns.color_palette("Reds", 2)
    palette_lower = sns.color_palette("Blues", 2)

    # Plot upper_body
    upper_plot = sns.pointplot(x='Timing', y='upper_body', hue='Expertise', data=data_prepared,
                               dodge=0.1, markers=['o', 's'], capsize=0.1, err_kws={'linewidth': 0.7},
                               palette=palette_upper, errorbar='sd', ax=axes[num_axes // 2, num_axes % 2])

    # Plot lower_body
    lower_plot = sns.pointplot(x='Timing', y='lower_body', hue='Expertise', data=data_prepared,
                               dodge=0.1, markers=['o', 's'], capsize=0.1, err_kws={'linewidth': 0.7},
                               palette=palette_lower, errorbar='sd', ax=axes[num_axes // 2, num_axes % 2])

    axes[num_axes // 2, num_axes % 2].set_title(
        f"{mvt_name}")
    axes[num_axes // 2, num_axes % 2].set_xlabel('Timing')
    axes[num_axes // 2, num_axes % 2].set_ylabel('Standard Deviation')

    handles, labels = axes[num_axes // 2, num_axes % 2].get_legend_handles_labels()
    new_labels = []
    exp_levels = list(data_prepared['Expertise'].unique())
    for i in range(len(labels) // 2):
        new_labels.append(f'Upper Body - {labels[i]}')
    for i in range(len(labels) // 2, len(labels)):
        new_labels.append(f'Lower Body - {exp_levels[i % len(exp_levels)]}')

    if num_axes == 0:
        axes[num_axes // 2, num_axes % 2].legend(handles, new_labels,
                                                 bbox_to_anchor=(1, 1))

    if num_axes > 0:
        axes[num_axes // 2, num_axes % 2].get_legend().remove()

    num_axes += 1

for i in range(num_axes, 4):
    fig.delaxes(axes.flatten()[i])

plt.suptitle("Interaction Between Timing and Expertise on Standard Deviation of upper and lower body")
plt.tight_layout()
plt.show()
