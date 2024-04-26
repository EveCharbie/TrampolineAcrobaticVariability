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

rotation_files = []  # Renamed variable to avoid conflict

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'position' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)  # Use the new list here


for file in rotation_files:
    data = pd.read_csv(file)
    data_prepared = prepare_data(data)
    data_prepared['Source'] = file.split('/')[-1].replace('results_', '').replace('_position.csv', '')  # Clean file ID

    # Modèle ANOVA pour le haut du corps sans tenir compte de l'expertise
    modele_upper = ols("upper_body ~ C(Timing)", data=data_prepared).fit()
    result_anova_upper = sm.stats.anova_lm(modele_upper, typ=2)
    print(result_anova_upper)

    # Modèle ANOVA pour le bas du corps sans tenir compte de l'expertise
    modele_lower = ols("lower_body ~ C(Timing)", data=data_prepared).fit()
    result_anova_lower = sm.stats.anova_lm(modele_lower, typ=2)
    print(result_anova_lower)

    # MANOVA pour les deux mesures sans l'interaction avec l'expertise
    maov = MANOVA.from_formula('upper_body + lower_body ~ C(Timing)', data=data_prepared)
    print(maov.mv_test())

    all_data = pd.concat([all_data, data_prepared], ignore_index=True)


# Prepare data for plotting
all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)

# Create the plot
plt.figure(figsize=(12, 8))
plot = sns.pointplot(x='Timing', y='lower_body', hue='Source', data=all_data, dodge=True,
                     capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd')


# Adding plot details
plt.title('Standard Deviation Across Different Timings from Multiple Files')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')
plt.legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Display the plot
plt.show()
