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
order = ['8-1o', '8-1<', '811<', '41', '41o', '42', '8-3<', '831<', '822', '43']
index = ['takeoff_75', '75_landing']
body_parts = ['upper_body', 'lower_body']

position_files = []
significant_value_upper_body = pd.DataFrame(columns=order, index=index)
significant_value_lower_body = pd.DataFrame(columns=order, index=index)

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'position' in file:
            full_path = os.path.join(root, file)
            position_files.append(full_path)

order_index = {key: index for index, key in enumerate(order)}
position_files = sorted(position_files, key=lambda x: order_index.get(extract_identifier(x), float('inf')))

for file in position_files:
    data = pd.read_csv(file)
    data_prepared = prepare_data(data)
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_position.csv', '')
    data_prepared['Source'] = mvt_name

    print(f"===== Movement {mvt_name} is running =====")

    # Check normality and homogeneity of variances
    issues = []

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

    # perform_anova_and_tukey(data_prepared, 'upper_body', 'Timing')
    # perform_anova_and_tukey(data_prepared, 'lower_body', 'Timing')

    posthoc_results_upper_body = perform_kruskal_and_dunn(data_prepared, 'upper_body', 'Timing')
    posthoc_results_lower_body = perform_kruskal_and_dunn(data_prepared, 'lower_body', 'Timing')

    significant_value_upper_body.loc["takeoff_75", mvt_name] = posthoc_results_upper_body.loc["Takeoff", "75%"]
    significant_value_upper_body.loc["75_landing", mvt_name] = posthoc_results_upper_body.loc["75%", "Landing"]
    significant_value_lower_body.loc["takeoff_75", mvt_name] = posthoc_results_lower_body.loc["Takeoff", "75%"]
    significant_value_lower_body.loc["75_landing", mvt_name] = posthoc_results_lower_body.loc["75%", "Landing"]

    all_data = pd.concat([all_data, data_prepared], ignore_index=True)


all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)

# fig, axes = plt.subplots(1, 2, figsize=(24, 8))  # 1 row, 2 columns of plots
#
# sns.pointplot(x='Timing', y='upper_body', hue='Source', data=all_data, dodge=0.5,
#               capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd', ax=axes[0])
# axes[0].set_title('Upper Body Standard Deviation Across Different Timings')
# axes[0].set_xlabel('Timing')
# axes[0].set_ylabel('Standard Deviation')
# axes[0].legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
# sns.pointplot(x='Timing', y='lower_body', hue='Source', data=all_data, dodge=0.5,
#               capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd', ax=axes[1])
# axes[1].set_title('Lower Body Standard Deviation Across Different Timings')
# axes[1].set_xlabel('Timing')
# axes[1].set_ylabel('Standard Deviation')
# axes[1].legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
# plt.tight_layout()
# plt.show()

categories = all_data['Timing'].cat.categories
pos_plot = np.array([1, 5, 9])
colors = plt.colormaps['tab20b_r'](np.linspace(0, 1, len(all_data['Source'].unique())))

## Plot upper body

plt.figure(figsize=(10, 8))
ax = plt.gca()
i_plot = 0

for i, mvt_name in enumerate(order):
    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)["upper_body"].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)["upper_body"].std()

    # #
    # print(mvt_name)
    # pourcentages = {}
    # keys = list(means.keys())
    # for i in range(len(keys) - 1):
    #     key1, key2 = keys[i], keys[i + 1]
    #     valeur1, valeur2 = means[key1], means[key2]
    #     pourcentage = ((valeur2 - valeur1) / valeur1) * 100
    #     pourcentages[f"{key1} to {key2}"] = pourcentage
    #
    # # Affichage des pourcentages
    # for key, value in pourcentages.items():
    #     print(f"{key}: {value:.2f}%")
    # #

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=mvt_name,
                 color=colors[i], capsize=5, elinewidth=0.5, capthick=0.5)

    plt.plot(pos_plot + i * 0.1, means, '-', color=colors[i])

    y_max = all_data["upper_body"].max()

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value_upper_body[mvt_name][sig_key]

        if p_value < 0.05:
            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1
            line_y = y_max + 0.05 * i_plot

            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1, xmax=pos_plot[j + 1] + i * 0.1, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1, ymin=line_y - 0.01, ymax=line_y, colors=colors[i], linestyles='solid',
                      lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1, ymin=line_y - 0.01, ymax=line_y, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 0.01, p_text, ha='center', va='bottom', color=colors[i])

            i_plot += 1

plt.xticks([1.5, 5.5, 9.5], categories)
plt.title('Upper Body Standard Deviation')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')
plt.legend(title='File ID', bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)


## Plot lower body

plt.figure(figsize=(10, 8))
ax = plt.gca()
i_plot = 0

for i, mvt_name in enumerate(order):
    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)["lower_body"].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)["lower_body"].std()

    # #
    # print(mvt_name)
    # pourcentages = {}
    # keys = list(means.keys())
    # for i in range(len(keys) - 1):
    #     key1, key2 = keys[i], keys[i + 1]
    #     valeur1, valeur2 = means[key1], means[key2]
    #     pourcentage = ((valeur2 - valeur1) / valeur1) * 100
    #     pourcentages[f"{key1} to {key2}"] = pourcentage
    #
    # # Affichage des pourcentages
    # for key, value in pourcentages.items():
    #     print(f"{key}: {value:.2f}%")
    # #

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=mvt_name,
                 color=colors[i], capsize=5, elinewidth=0.5, capthick=0.5)

    plt.plot(pos_plot + i * 0.1, means, '-', color=colors[i])

    y_max = all_data["lower_body"].max()

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value_lower_body[mvt_name][sig_key]

        if p_value < 0.05:
            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1
            line_y = y_max + 0.02 * i_plot

            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1, xmax=pos_plot[j + 1] + i * 0.1, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1, ymin=line_y - 0.006, ymax=line_y, colors=colors[i], linestyles='solid',
                      lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1, ymin=line_y - 0.006, ymax=line_y, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 0.005, p_text, ha='center', va='bottom', color=colors[i])
            i_plot += 1

plt.xticks([1.5, 5.5, 9.5], categories)
plt.title('Lower Body Standard Deviation')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')
plt.legend(title='File ID', bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)


print("Statistical test for all acrobatics")
posthoc_results_total = perform_kruskal_and_dunn(all_data, 'upper_body', 'Timing')
significant_value_takeoff_75 = posthoc_results_total.loc["Takeoff", "75%"]
significant_value_75_landing = posthoc_results_total.loc["75%", "Landing"]

plt.figure(figsize=(12, 8))
ax = plt.gca()
pos_plot = np.array([1, 5, 9])

means = all_data.groupby('Timing', observed=True)['upper_body'].mean()
std_devs = all_data.groupby('Timing', observed=True)['upper_body'].std()

pourcentages = {}
keys = list(means.keys())
for i in range(len(keys) - 1):
    key1, key2 = keys[i], keys[i + 1]
    valeur1, valeur2 = means[key1], means[key2]
    pourcentage = ((valeur2 - valeur1) / valeur1) * 100
    pourcentages[f"{key1} to {key2}"] = pourcentage

# Affichage des pourcentages
for key, value in pourcentages.items():
    print(f"{key}: {value:.2f}%")

plt.errorbar(x=pos_plot, y=means, yerr=std_devs, fmt='o', capsize=5, elinewidth=0.5, capthick=0.5, color="black")
plt.plot(pos_plot, means, '-', color="black")

y_max = all_data["upper_body"].max() - 0.4

for j in range(len(pos_plot) - 1):
    p_value = significant_value_takeoff_75 if j == 0 else significant_value_75_landing

    if p_value < 0.05:
        p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2
        line_y = y_max + 0.03 * i_plot

        ax.hlines(y=line_y, xmin=pos_plot[j] + 0.1, xmax=pos_plot[j + 1] - 0.1,
                  linestyles='solid', lw=1, color="black")
        ax.vlines(x=pos_plot[j] + 0.1, ymin=line_y - 0.01, ymax=line_y, linestyles='solid',
                  lw=1, color="black")
        ax.vlines(x=pos_plot[j + 1] - 0.1, ymin=line_y - 0.01, ymax=line_y,
                  linestyles='solid', lw=1, color="black")
        ax.text(mid_point, line_y, p_text, ha='center', va='bottom')

plt.xticks([1, 5, 9], categories)
plt.title('Mean Upper Body Standard Deviation')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')


print("Statistical test for all acrobatics")
posthoc_results_total = perform_kruskal_and_dunn(all_data, 'lower_body', 'Timing')
significant_value_takeoff_75 = posthoc_results_total.loc["Takeoff", "75%"]
significant_value_75_landing = posthoc_results_total.loc["75%", "Landing"]

plt.figure(figsize=(12, 8))
ax = plt.gca()
pos_plot = np.array([1, 5, 9])

means = all_data.groupby('Timing', observed=True)['lower_body'].mean()
std_devs = all_data.groupby('Timing', observed=True)['lower_body'].std()

pourcentages = {}
keys = list(means.keys())
for i in range(len(keys) - 1):
    key1, key2 = keys[i], keys[i + 1]
    valeur1, valeur2 = means[key1], means[key2]
    pourcentage = ((valeur2 - valeur1) / valeur1) * 100
    pourcentages[f"{key1} to {key2}"] = pourcentage

# Affichage des pourcentages
for key, value in pourcentages.items():
    print(f"{key}: {value:.2f}%")

plt.errorbar(x=pos_plot, y=means, yerr=std_devs, fmt='o', capsize=5, elinewidth=0.5, capthick=0.5, color="black")
plt.plot(pos_plot, means, '-', color="black")

y_max = all_data["lower_body"].max() - 0.4

for j in range(len(pos_plot) - 1):
    p_value = significant_value_takeoff_75 if j == 0 else significant_value_75_landing

    if p_value < 0.05:
        p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2
        line_y = y_max + 0.03 * i_plot

        ax.hlines(y=line_y, xmin=pos_plot[j] + 0.1, xmax=pos_plot[j + 1] - 0.1,
                  linestyles='solid', lw=1, color="black")
        ax.vlines(x=pos_plot[j] + 0.1, ymin=line_y - 0.01, ymax=line_y, linestyles='solid',
                  lw=1, color="black")
        ax.vlines(x=pos_plot[j + 1] - 0.1, ymin=line_y - 0.01, ymax=line_y,
                  linestyles='solid', lw=1, color="black")
        ax.text(mid_point, line_y, p_text, ha='center', va='bottom')

plt.xticks([1, 5, 9], categories)
plt.title('Mean Lower Body Standard Deviation')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')
plt.show()