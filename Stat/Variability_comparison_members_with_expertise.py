import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, mannwhitneyu, shapiro, kruskal
import matplotlib.patches as mpatches
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.anova import anova_lm
from scikit_posthocs import posthoc_dunn


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

for file in files:
    data = pd.read_csv(file)
    data_prepared = prepare_data(data)
    data_prepared = data[data['Timing'] != 'Other'].copy()
    data_prepared['Source'] = file.split('/')[-1].replace('results_', '').replace('_position.csv', '')

    print(f"===== Movement {data_prepared['Source'][0]} is running =====")

    # Check normality and homogeneity of variance for each group
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


    plt.figure(figsize=(12, 8))
    upper_plot = sns.pointplot(x='Timing', y='upper_body', hue='Expertise', data=data_prepared,
                               dodge=0.1, markers=['o', 's'], capsize=0.1, err_kws={'linewidth': 0.7},
                               palette='Set2', errorbar='sd')
    handles, labels = upper_plot.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title="Expertise")

    lower_plot = sns.pointplot(x='Timing', y='lower_body', hue='Expertise', data=data_prepared,
                               dodge=0.1, markers=['o', 's'], capsize=0.1, err_kws={'linewidth': 0.7},
                               palette='Set1', errorbar='sd')
    handles, labels = lower_plot.get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title="Expertise")

    plt.title('Interaction Between Timing, Expertise, and Body Part on Standard Deviation')
    plt.xlabel('Timing')
    plt.ylabel('Standard Deviation')

    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    exp_levels = list(data_prepared['Expertise'].unique())
    for i in range(len(labels) // 2):
        new_labels.append(f'Upper Body - {labels[i]}')
    for i in range(len(labels) // 2, len(labels)):
        new_labels.append(f'Lower Body - {exp_levels[i % len(exp_levels)]}')

    plt.legend(handles, new_labels, title='Expertise and Body Part', bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()

