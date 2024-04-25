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

def prepare_data(data, dataset_label, body_part):
    upper_body_columns = data[['AvBrasD', 'MainD', 'AvBrasG', 'MainG']]
    data["upper_body"] = upper_body_columns.mean(axis=1)
    lower_body_columns = data[['JambeD', 'PiedD', 'JambeG', 'PiedG']]
    data["lower_body"] = lower_body_columns.mean(axis=1)

    conditions = [
        (data['Timing'] == '75%') & (data['Expertise'] == 'Elite'),
        (data['Timing'] == 'Landing') & (data['Expertise'] == 'Elite'),
        (data['Timing'] == '75%') & (data['Expertise'] == 'SubElite'),
        (data['Timing'] == 'Landing') & (data['Expertise'] == 'SubElite')
    ]
    labels = ['Elite 75%', 'Elite Landing', 'SubElite 75%', 'SubElite Landing']

    data['Group'] = np.select(conditions, labels, default='Other')

    data_subset = data[data['Group'] != 'Other']
    data_subset['Dataset'] = dataset_label
    return data_subset[['upper_body', 'lower_body', 'Group', 'Dataset']]


data_41 = pd.read_csv('/results_41_position.csv')
data_42 = pd.read_csv('/results_42_position.csv')
data_43 = pd.read_csv('/results_43_position.csv')


# Préparer les données
data_41_prepared = prepare_data(data_41, 'Dataset 41', 'upper_body')
data_42_prepared = prepare_data(data_42, 'Dataset 42', 'upper_body')
data_43_prepared = prepare_data(data_43, 'Dataset 43', 'upper_body')

data_anova = pd.read_csv('/results_43_position.csv')

upper_body_colums = data_anova[['AvBrasD', 'MainD', 'AvBrasG', 'MainG']]
data_anova["upper_body"] = upper_body_colums.mean(axis=1)

lower_body_colums = data_anova[['JambeD', 'PiedD', 'JambeG', 'PiedG']]
data_anova["lower_body"] = lower_body_colums.mean(axis=1)


modele_upper = ols("upper_body ~ C(Expertise) * C(Timing)", data=data_anova).fit()
result_anova_upper = sm.stats.anova_lm(modele_upper, typ=2)
print(result_anova_upper)

modele_lower = ols("lower_body ~ C(Expertise) * C(Timing)", data=data_anova).fit()
result_anova_lower = sm.stats.anova_lm(modele_lower, typ=2)
print(result_anova_lower)

maov = MANOVA.from_formula('upper_body + lower_body ~ C(Expertise) * C(Timing)', data=data_anova)
print(maov.mv_test())


all_data = pd.concat([data_41_prepared, data_42_prepared, data_43_prepared])

# Créer le boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Group', y='upper_body', hue='Dataset', data=all_data, palette='Set2')
plt.title('Distribution of Upper Body Measurements Across Expertise, Timing, and Dataset')
plt.ylabel('Upper Body SD')
plt.xlabel('Group')
plt.legend(title='Dataset')


# Créer le boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Group', y='lower_body', hue='Dataset', data=all_data, palette='Set2')
plt.title('Distribution of Lower Body Measurements Across Expertise, Timing, and Dataset')
plt.ylabel('Lower Body SD')
plt.xlabel('Group')
plt.legend(title='Dataset')

# Afficher le graphique
plt.show()