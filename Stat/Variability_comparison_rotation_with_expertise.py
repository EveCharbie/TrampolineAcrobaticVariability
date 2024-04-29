import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, levene
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd


files = [
    '/home/lim/Documents/StageMathieu/Tab_result/results_41_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_41o_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_42_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result/results_43_rotation.csv'
]

for file in files:
    data = pd.read_csv(file)

    anova_rot_df = data.pivot_table(index=['ID', 'Expertise'], columns='Timing', values='Std')
    anova_rot_df.reset_index(inplace=True)

    anova_rot_df.columns = ['SubjectID', 'Expertise', 'Std_75%', 'Std_Takeoff', 'Std_Landing']

    data_75_Landing = data[data['Timing'].isin(['75%', 'Landing'])]
    data_75_takeoff = data[data['Timing'].isin(['75%', 'Takeoff'])]
    data_Landing_takeoff = data[data['Timing'].isin(['Landing', 'Takeoff'])]


    # Check normality and homogeneity for each group
    groups = ['75%', 'Landing', 'Takeoff']
    expertise_levels = data['Expertise'].unique()

    for group in groups:
        for expertise in expertise_levels:
            subgroup_data = data[(data['Timing'] == group) & (data['Expertise'] == expertise)]['Std']
            stat, p_value = shapiro(subgroup_data)
            if p_value < 0.05:
                print(f"Normality issue in {group}-{expertise} of {file} (P-value: {p_value:.4f})")

    data_groups = [data[(data['Timing'] == group) & (data['Expertise'] == expertise)]['Std'] for group in groups for
                   expertise in expertise_levels]
    levene_stat, levene_p = levene(*data_groups)
    if levene_p < 0.05:
        print(f"Variance homogeneity issue across groups in {file} (P-value: {levene_p:.4f})")

    ##
    # Modèle ANOVA pour 75% vs Landing
    model_75_Landing = ols('Std ~ C(Timing) * C(Expertise)', data=data_75_Landing).fit()
    anova_results_75_Landing = anova_lm(model_75_Landing, typ=2)

    # Modèle ANOVA pour 75% vs Takeoff
    model_75_takeoff = ols('Std ~ C(Timing) * C(Expertise)', data=data_75_takeoff).fit()
    anova_results_75_takeoff = anova_lm(model_75_takeoff, typ=2)

    # Modèle ANOVA pour Landing vs Takeoff
    model_Landing_takeoff = ols('Std ~ C(Timing) * C(Expertise)', data=data_Landing_takeoff).fit()
    anova_results_Landing_takeoff = anova_lm(model_Landing_takeoff, typ=2)

    # Affichage des résultats
    print("ANOVA Results for 75% vs Landing:\n", anova_results_75_Landing)
    print("\nANOVA Results for 75% vs Takeoff:\n", anova_results_75_takeoff)
    print("\nANOVA Results for Landing vs Takeoff:\n", anova_results_Landing_takeoff)


    data_specific = data[data['Timing'].isin(['75%', 'Landing', 'Takeoff'])]

    # Modèle ANOVA en tenant compte de 'Timing' et 'Expertise'
    model = ols('Std ~ C(Timing) * C(Expertise)', data=data_specific).fit()
    anova_results = anova_lm(model, typ=2)  # Type II ANOVA
    print("ANOVA Results with Expertise considered:\n", anova_results)

    # Création des groupes pour le test de Tukey en convertissant d'abord en chaîne
    data_specific['groups'] = data_specific['Timing'].astype(str) + "_" + data_specific['Expertise'].astype(str)

    # Test post-hoc de Tukey HSD
    tukey = pairwise_tukeyhsd(endog=data_specific['Std'], groups=data_specific['groups'], alpha=0.05)
    print(tukey)

    data['Timing'] = pd.Categorical(data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)
    data['Expertise'] = pd.Categorical(data['Expertise'], categories=["Elite", "SubElite"], ordered=True)

    ####

    # Création du graphique en différenciant par 'Expertise'
    plt.figure(figsize=(12, 8))
    sns.pointplot(x='Timing', y='Std', hue='Expertise', data=data, dodge=True, markers=['o', 's'],
                  capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd')

    # Ajout de détails au graphique
    plt.title('Interaction Between Timing and Expertise on Standard Deviation')
    plt.xlabel('Timing')
    plt.ylabel('Standard Deviation')
    plt.legend(title='Expertise')

    # Afficher le graphique
    plt.show()


