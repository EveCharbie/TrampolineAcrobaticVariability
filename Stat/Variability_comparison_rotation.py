import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene
import os
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp
from TrampolineAcrobaticVariability.Function.Function_stat import perform_anova_and_tukey, perform_kruskal_and_dunn
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import extract_identifier
home_path = "/home/lim/Documents/StageMathieu/Tab_result/"

rotation_files = []
index = ['takeoff_75', '75_landing']
order = ['8-1o', '8-1<', '811<', '41', '41o', '8-3<', '42', '831<', '822', '43']

full_name_acrobatics = {
    '4-': '4-/',
    '4-o':  '4-o',
    '8--o': '8--o',
    '8-1<': '8-1<',
    '8-1o': '8-1o',
    '41': '41/',
    '811<': '811<',
    '41o': '41o',
    '8-3<': '8-3<',
    '42': '42/',
    '822': '822/',
    '831<': '831<',
    '43': '43/',

}

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)

order_index = {key: index for index, key in enumerate(order)}
rotation_files = sorted(rotation_files, key=lambda x: order_index.get(extract_identifier(x), float('inf')))

all_data = pd.DataFrame()
significant_value = pd.DataFrame(columns=order, index=index)

for file in rotation_files:
    data = pd.read_csv(file)
    data_specific = data[data['Timing'].isin(['75%', 'Landing', 'Takeoff'])]
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID
    data_specific['Source'] = mvt_name

    print(f"===== Movement {mvt_name} is running =====")


    # # Check normality and homogeneity of variances
    # issues = []
    # for timing in data_specific['Timing'].unique():
    #     group_data = data_specific[data_specific['Timing'] == timing]['Std']
    #     stat, p = shapiro(group_data)
    #     if p < 0.05:
    #         issues.append(f"Normality issue in {timing} of {file} (P-value: {p:.4f})")
    #
    # levene_stat, levene_p = levene(
    #     *[data_specific[data_specific['Timing'] == timing]['Std'] for timing in data_specific['Timing'].unique()])
    # if levene_p < 0.05:
    #     issues.append(f"Variance homogeneity issue in {file} (P-value: {levene_p:.4f})")
    #
    # if issues:
    #     print("\n".join(issues))

    # perform_anova_and_tukey(data_specific, 'Std', 'Timing')
    posthoc_results = perform_kruskal_and_dunn(data_specific, 'Std', 'Timing')


    significant_value.loc["takeoff_75", mvt_name] = posthoc_results.loc["Takeoff", "75%"]
    significant_value.loc["75_landing", mvt_name] = posthoc_results.loc["75%", "Landing"]

    # Append data to the plotting DataFrame
    all_data = pd.concat([all_data, data_specific], ignore_index=True)

all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)

plt.figure(figsize=(15, 10))
ax = plt.gca()

categories = all_data['Timing'].cat.categories
pos_plot = np.array([1, 5, 9])

colors = plt.colormaps['tab20b_r'](np.linspace(0, 1, len(all_data['Source'].unique())))

i_plot = 0

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)['Std'].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)['Std'].std()

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

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=colors[i], capsize=5, elinewidth=0.5, capthick=0.5)

    plt.plot(pos_plot + i * 0.1, means, '-', color=colors[i])

    y_max = all_data["Std"].max()

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value[mvt_name][sig_key]

        if p_value < 0.05:
            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1
            line_y = y_max + 0.03 * i_plot

            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1, xmax=pos_plot[j + 1] + i * 0.1, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1, ymin=line_y - 0.01, ymax=line_y, colors=colors[i], linestyles='solid',
                      lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1, ymin=line_y - 0.01, ymax=line_y, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 0.005, p_text, ha='center', va='bottom', color=colors[i])

            i_plot += 1

plt.xticks([1.5, 5.5, 9.5], categories)
plt.title('Pelvic Rotation')
plt.xlabel('Timing')
plt.ylabel('SD')
plt.subplots_adjust(left=0.035, right=0.91, top=0.937, bottom=0.082)
plt.legend(title='Acrobatics Code', bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/rotation.png")


print("Statistical test for all acrobatics")
posthoc_results_total = perform_kruskal_and_dunn(all_data, 'Std', 'Timing')
significant_value_takeoff_75 = posthoc_results_total.loc["Takeoff", "75%"]
significant_value_75_landing = posthoc_results_total.loc["75%", "Landing"]

plt.figure(figsize=(12, 8))
ax = plt.gca()
pos_plot = np.array([1, 5, 9])

means = all_data.groupby('Timing', observed=True)['Std'].mean()
std_devs = all_data.groupby('Timing', observed=True)['Std'].std()

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

y_max = all_data["Std"].max() - 0.4

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
plt.title('Pelvic Rotation')
plt.xlabel('Timing')
plt.ylabel('SD')
plt.tight_layout()
plt.savefig("/home/lim/Documents/StageMathieu/meeting/mean_rotation.png")

plt.show()
