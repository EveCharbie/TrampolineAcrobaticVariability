import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, mannwhitneyu, shapiro
import matplotlib.patches as mpatches
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA
import os


def prepare_data(data):
    # Calcul des moyennes pour le haut et le bas du corps
    upper_body_columns = data[['AvBrasD', 'MainD', 'AvBrasG', 'MainG']]
    data["upper_body"] = upper_body_columns.mean(axis=1)
    lower_body_columns = data[['JambeD', 'PiedD', 'JambeG', 'PiedG']]
    data["lower_body"] = lower_body_columns.mean(axis=1)

    # Création des groupes basés uniquement sur le Timing
    conditions = [
        (data['Timing'] == 'Takeoff'),
        (data['Timing'] == '75%'),
        (data['Timing'] == 'Landing')
    ]
    labels = ['Takeoff', '75%', 'Landing']

    data['Timing'] = np.select(conditions, labels, default='Other')

    data_subset = data[data['Timing'] != 'Other']
    return data_subset[['upper_body', 'lower_body', 'Timing']]


all_data = pd.DataFrame()

home_path = "/home/lim/Documents/StageMathieu/Tab_result/"

rotation_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'position' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)


for file in rotation_files:
    data = pd.read_csv(file)
    data_prepared = prepare_data(data)
    data_prepared['Source'] = file.split('/')[-1].replace('results_', '').replace('_position.csv', '')

    print(f"===== Movement {data_prepared['Source'][0]} is running =====")

    # Check normality and homogeneity of variances
    issues = []
    body_parts = ['upper_body', 'lower_body']

    for body_part in body_parts:
        for timing in data_prepared['Timing'].unique():
            group_data = data_prepared[data_prepared['Timing'] == timing][body_part]
            stat, p = shapiro(group_data)
            if p < 0.05:
                issues.append(f"Normality issue in {timing} for {body_part} (P-value: {p:.4f})")

        levene_stat, levene_p = levene(
            *[data_prepared[data_prepared['Timing'] == timing][body_part] for timing in
              data_prepared['Timing'].unique()]
        )
        if levene_p < 0.05:
            issues.append(f"Variance homogeneity issue for {body_part} (P-value: {levene_p:.4f})")

    if issues:
        print("\n".join(issues))


    # Parametric test ANOVA for upper_body
    modele_upper = ols("upper_body ~ C(Timing)", data=data_prepared).fit()
    result_anova_upper = sm.stats.anova_lm(modele_upper, typ=2)
    print(result_anova_upper)

    # Parametric test ANOVA for lower_body
    modele_lower = ols("lower_body ~ C(Timing)", data=data_prepared).fit()
    result_anova_lower = sm.stats.anova_lm(modele_lower, typ=2)
    print(result_anova_lower)

    # Parametric test MANOVA for upper and lower body
    maov = MANOVA.from_formula('upper_body + lower_body ~ C(Timing)', data=data_prepared)
    print(maov.mv_test())

    all_data = pd.concat([all_data, data_prepared], ignore_index=True)


all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(24, 8))  # 1 row, 2 columns of plots

sns.pointplot(x='Timing', y='upper_body', hue='Source', data=all_data, dodge=True,
              capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd', ax=axes[0])
axes[0].set_title('Upper Body Standard Deviation Across Different Timings')
axes[0].set_xlabel('Timing')
axes[0].set_ylabel('Standard Deviation')
axes[0].legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.pointplot(x='Timing', y='lower_body', hue='Source', data=all_data, dodge=True,
              capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd', ax=axes[1])
axes[1].set_title('Lower Body Standard Deviation Across Different Timings')
axes[1].set_xlabel('Timing')
axes[1].set_ylabel('Standard Deviation')
axes[1].legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.show()
