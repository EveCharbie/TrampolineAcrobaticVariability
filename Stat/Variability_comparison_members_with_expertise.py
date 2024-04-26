import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, mannwhitneyu
import matplotlib.patches as mpatches
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
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

for file in files:
    data = pd.read_csv(file)
    data_prepared = prepare_data(data)
    data_prepared = data[data['Timing'] != 'Other'].copy()
    data_prepared['Source'] = file.split('/')[-1].replace('results_', '').replace('_position.csv', '')

    # Upper body ANOVA
    modele_upper = ols("upper_body ~ C(Expertise) * C(Timing)", data=data_prepared).fit()
    result_anova_upper = anova_lm(modele_upper, typ=2)
    print(result_anova_upper)

    # Lower body ANOVA
    modele_lower = ols("lower_body ~ C(Expertise) * C(Timing)", data=data_prepared).fit()
    result_anova_lower = anova_lm(modele_lower, typ=2)
    print(result_anova_lower)

    # MANOVA
    maov = MANOVA.from_formula('upper_body + lower_body ~ C(Expertise) * C(Timing)', data=data_prepared)
    print(maov.mv_test())

    plt.figure(figsize=(12, 8))

    # Plot for upper body
    upper_plot = sns.pointplot(x='Timing', y='upper_body', hue='Expertise', data=data_prepared,
                               dodge=0.1, markers=['o', 's'], capsize=0.1, err_kws={'linewidth': 0.3},
                               palette='Set2', errorbar='sd', label='Upper Body')

    # Plot for lower body using the same axes
    lower_plot = sns.pointplot(x='Timing', y='lower_body', hue='Expertise', data=data_prepared,
                               dodge=0.1, markers=['o', 's'], capsize=0.1, err_kws={'linewidth': 0.3},
                               palette='Set1', errorbar='sd', label='Lower Body')

    # Adding details to the plot
    plt.title('Interaction Between Timing, Expertise, and Body Part on Standard Deviation')
    plt.xlabel('Timing')
    plt.ylabel('Standard Deviation')

    # Adjust legend to show body part and expertise
    handles, labels = plt.gca().get_legend_handles_labels()
    new_labels = []
    exp_levels = list(data_prepared['Expertise'].unique())
    for i in range(len(labels) // 2):
        new_labels.append(f'Upper Body - {labels[i]}')
    for i in range(len(labels) // 2, len(labels)):
        new_labels.append(f'Lower Body - {exp_levels[i % len(exp_levels)]}')

    plt.legend(handles, new_labels, title='Expertise and Body Part', bbox_to_anchor=(1, 1), loc='upper left')
    plt.show()
