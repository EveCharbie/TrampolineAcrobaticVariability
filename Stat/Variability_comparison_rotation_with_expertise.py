import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns

# data = pd.read_csv('/home/lim/Documents/StageMathieu/results_41_rotation.csv')
# data = pd.read_csv('/home/lim/Documents/StageMathieu/results_42_rotation.csv')
data = pd.read_csv('/home/lim/Documents/StageMathieu/results_43_rotation.csv')

anova_rot_df = data.pivot_table(index=['ID', 'Expertise'], columns='Timing', values='Std')
anova_rot_df.reset_index(inplace=True)

# Renommer les colonnes pour refléter les nouveaux noms
anova_rot_df.columns = ['SubjectID', 'Expertise', 'Std_75%', 'Std_Takeoff', 'Std_Landing']


# Filtrez les données pour chaque comparaison spécifique
data_75_landing = data[data['Timing'].isin(['75%', 'landing'])]
data_75_takeoff = data[data['Timing'].isin(['75%', 'Takeoff'])]
data_landing_takeoff = data[data['Timing'].isin(['landing', 'Takeoff'])]

# Modèle ANOVA pour 75% vs Landing
model_75_landing = ols('Std ~ C(Timing) * C(Expertise)', data=data_75_landing).fit()
anova_results_75_landing = anova_lm(model_75_landing, typ=2)  # Type II ANOVA

# Modèle ANOVA pour 75% vs Takeoff
model_75_takeoff = ols('Std ~ C(Timing) * C(Expertise)', data=data_75_takeoff).fit()
anova_results_75_takeoff = anova_lm(model_75_takeoff, typ=2)  # Type II ANOVA

# Modèle ANOVA pour landing vs Takeoff
model_landing_takeoff = ols('Std ~ C(Timing) * C(Expertise)', data=data_landing_takeoff).fit()
anova_results_landing_takeoff = anova_lm(model_landing_takeoff, typ=2)  # Type II ANOVA

# Affichage des résultats
print("ANOVA Results for 75% vs Landing:\n", anova_results_75_landing)
print("\nANOVA Results for 75% vs Takeoff:\n", anova_results_75_takeoff)
print("\nANOVA Results for landing vs Takeoff:\n", anova_results_landing_takeoff)

data['Timing'] = pd.Categorical(data['Timing'], categories=["Takeoff", "75%", "landing"], ordered=True)
data['Expertise'] = pd.Categorical(data['Expertise'], categories=["Elite", "SubElite"], ordered=True)

# Création du graphique
plt.figure(figsize=(10, 6))
sns.pointplot(x='Timing', y='Std', hue='Expertise', data=data, dodge=True, markers=['o', 's'],
              capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd')

# Ajout de détails au graphique
plt.title('Interaction Between Timing and Expertise on Standard Deviation')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')
plt.legend(title='Expertise')

data['Timing'] = pd.Categorical(data['Timing'], categories=["Takeoff", "75%", "landing"], ordered=True)

# Création du graphique sans différencier par 'Expertise'
plt.figure(figsize=(10, 6))
sns.pointplot(x='Timing', y='Std', data=data, dodge=True, markers='o',
              capsize=0.1, err_kws={'linewidth': 0.5}, errorbar='sd')

# Ajout de détails au graphique
plt.title('Standard Deviation Across Different Timings')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')

# Afficher le graphique
plt.show()
