import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene
import os
from scipy.stats import kruskal
import scikit_posthocs as sp


home_path = "/home/lim/Documents/StageMathieu/Tab_result/"

rotation_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)


all_data = pd.DataFrame()

for file in rotation_files:
    data = pd.read_csv(file)
    data_specific = data[data['Timing'].isin(['75%', 'Landing', 'Takeoff'])]
    data_specific['Source'] = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID

    print(f"===== Movement {data_specific['Source'][0]} is running =====")


    # Check normality and homogeneity of variances
    issues = []
    for timing in data_specific['Timing'].unique():
        group_data = data_specific[data_specific['Timing'] == timing]['Std']
        stat, p = shapiro(group_data)
        if p < 0.05:
            issues.append(f"Normality issue in {timing} of {file} (P-value: {p:.4f})")

    levene_stat, levene_p = levene(
        *[data_specific[data_specific['Timing'] == timing]['Std'] for timing in data_specific['Timing'].unique()])
    if levene_p < 0.05:
        issues.append(f"Variance homogeneity issue in {file} (P-value: {levene_p:.4f})")

    if issues:
        print("\n".join(issues))


    # Parametric-test ANOVA
    model = ols('Std ~ C(Timing)', data=data_specific).fit()
    anova_results = anova_lm(model, typ=2)
    print(f"ANOVA Results for {data_specific['Source'].iloc[0]}:\n", anova_results)

    # Tukey HSD Test
    tukey = pairwise_tukeyhsd(endog=data_specific['Std'], groups=data_specific['Timing'], alpha=0.05)
    print(f"Tukey HSD Results for {data_specific['Source'].iloc[0]}:\n", tukey)


    # Non_parametric-test Kruskal_Wallis
    kruskal_data = [data_specific[data_specific['Timing'] == timing]['Std'].values for timing in
                    data_specific['Timing'].unique()]
    kruskal_stat, kruskal_p = kruskal(*kruskal_data)
    print(f"Kruskal-Wallis Test Results for {data_specific['Source'].iloc[0]} (P-value: {kruskal_p:.4f})")

    if kruskal_p < 0.05:
        posthoc_results = sp.posthoc_dunn(data_specific, val_col='Std', group_col='Timing', p_adjust='bonferroni')
        print(f"Post-hoc Dunn's Test Results for {data_specific['Source'].iloc[0]}:\n", posthoc_results)


    # Append data to the plotting DataFrame
    all_data = pd.concat([all_data, data_specific], ignore_index=True)

all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)

plt.figure(figsize=(12, 8))
plot = sns.pointplot(x='Timing', y='Std', hue='Source', data=all_data, dodge=True,
                     capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd')

plt.title('Standard Deviation Across Different Timings from Multiple Files')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')
plt.legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.show()