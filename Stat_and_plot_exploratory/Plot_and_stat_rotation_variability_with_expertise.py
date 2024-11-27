import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro, levene
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd


files = [
    '/home/lim/Documents/StageMathieu/Tab_result3/results_41_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result3/results_41o_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result3/results_42_rotation.csv',
    '/home/lim/Documents/StageMathieu/Tab_result3/results_43_rotation.csv'
]

num_axes = 0

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

legend_added = False

for file in files:
    if num_axes >= 4:
        break

    data = pd.read_csv(file)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID

    anova_rot_df = data.pivot_table(index=['ID', 'Expertise'], columns='Timing', values='Std')
    anova_rot_df.reset_index(inplace=True)

    anova_rot_df.columns = ['SubjectID', 'Expertise', 'Std_75%', 'Std_Takeoff', 'Std_Landing']

    data_75_Landing = data[data['Timing'].isin(['75%', 'Landing'])]
    data_75_takeoff = data[data['Timing'].isin(['75%', 'Takeoff'])]
    data_Landing_takeoff = data[data['Timing'].isin(['Landing', 'Takeoff'])]

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

    # Modèle ANOVA for 75% vs Landing
    model_75_Landing = ols('Std ~ C(Timing) * C(Expertise)', data=data_75_Landing).fit()
    anova_results_75_Landing = anova_lm(model_75_Landing, typ=2)

    # Modèle ANOVA for 75% vs Takeoff
    model_75_takeoff = ols('Std ~ C(Timing) * C(Expertise)', data=data_75_takeoff).fit()
    anova_results_75_takeoff = anova_lm(model_75_takeoff, typ=2)

    # Modèle ANOVA for Landing vs Takeoff
    model_Landing_takeoff = ols('Std ~ C(Timing) * C(Expertise)', data=data_Landing_takeoff).fit()
    anova_results_Landing_takeoff = anova_lm(model_Landing_takeoff, typ=2)

    data_specific = data[data['Timing'].isin(['75%', 'Landing', 'Takeoff'])]

    model = ols('Std ~ C(Timing) * C(Expertise)', data=data_specific).fit()
    anova_results = anova_lm(model, typ=2)

    data_specific['groups'] = data_specific['Timing'].astype(str) + "_" + data_specific['Expertise'].astype(str)

    tukey = pairwise_tukeyhsd(endog=data_specific['Std'], groups=data_specific['groups'], alpha=0.05)

    data['Timing'] = pd.Categorical(data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)
    data['Expertise'] = pd.Categorical(data['Expertise'], categories=["Elite", "SubElite"], ordered=True)

    sns.pointplot(x='Timing', y='Std', hue='Expertise', data=data, dodge=True, markers=['o', 's'],
                  capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd',
                  ax=axes[num_axes // 2, num_axes % 2])

    axes[num_axes // 2, num_axes % 2].set_title(
        f"{mvt_name}")
    axes[num_axes // 2, num_axes % 2].set_xlabel('Timing')
    axes[num_axes // 2, num_axes % 2].set_ylabel('Standard Deviation')

    if num_axes > 0:
        axes[num_axes // 2, num_axes % 2].get_legend().remove()

    num_axes += 1

for i in range(num_axes, 4):
    fig.delaxes(axes.flatten()[i])

plt.suptitle("Interaction Between Timing and Expertise on Standard Deviation")

plt.show()