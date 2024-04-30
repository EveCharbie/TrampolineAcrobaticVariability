import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, shapiro
import numpy as np
from statsmodels.formula.api import ols
import os
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import extract_identifier
from TrampolineAcrobaticVariability.Function.Function_stat import (perform_anova_and_tukey,
                                                                   perform_kruskal_and_dunn,
                                                                   prepare_data)


all_data = pd.DataFrame()

home_path = "/home/lim/Documents/StageMathieu/Tab_result/"

position_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'position' in file:
            full_path = os.path.join(root, file)
            position_files.append(full_path)

order = ['8-1<', '811<', '41', '41o', '42', '831<', '822', '43']
order_index = {key: index for index, key in enumerate(order)}
position_files = sorted(position_files, key=lambda x: order_index.get(extract_identifier(x), float('inf')))

for file in position_files:
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

    perform_anova_and_tukey(data_prepared, 'upper_body', 'Timing')
    perform_anova_and_tukey(data_prepared, 'lower_body', 'Timing')

    perform_kruskal_and_dunn(data_prepared, 'upper_body', 'Timing')
    perform_kruskal_and_dunn(data_prepared, 'lower_body', 'Timing')


    all_data = pd.concat([all_data, data_prepared], ignore_index=True)


all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)

fig, axes = plt.subplots(1, 2, figsize=(24, 8))  # 1 row, 2 columns of plots

sns.pointplot(x='Timing', y='upper_body', hue='Source', data=all_data, dodge=0.5,
              capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd', ax=axes[0])
axes[0].set_title('Upper Body Standard Deviation Across Different Timings')
axes[0].set_xlabel('Timing')
axes[0].set_ylabel('Standard Deviation')
axes[0].legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

sns.pointplot(x='Timing', y='lower_body', hue='Source', data=all_data, dodge=0.5,
              capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd', ax=axes[1])
axes[1].set_title('Lower Body Standard Deviation Across Different Timings')
axes[1].set_xlabel('Timing')
axes[1].set_ylabel('Standard Deviation')
axes[1].legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.tight_layout()
plt.show()
