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
index = ['takeoff_75', '75_landing', 'takeoff_landing']
order = ['8-1o', '8-1<', '811<', '41', '41o', '8-3<', '42', '831<', '822', '43']
labels_x_empty = [" ", " ", " "]

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
    significant_value.loc["takeoff_landing", mvt_name] = posthoc_results.loc["Takeoff", "Landing"]

    # Append data to the plotting DataFrame
    all_data = pd.concat([all_data, data_specific], ignore_index=True)

all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)
all_data['Std'] = np.degrees(all_data['Std'])

# all acrobatics
##
plt.figure(figsize=(363 / 96, 242 / 96))

initial_ticks = np.arange(0, 50, 10)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.0f}" for tick in initial_ticks]
ax = plt.gca()

categories = all_data['Timing'].cat.categories
pos_plot = np.array([1, 5, 9])

colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(all_data['Source'].unique())))
y_increment = 4  # Increment for y-position for each level
line_y = all_data["Std"].max() - 5  # Initialize line_y outside the loop

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)['Std'].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)['Std'].std()

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=colors[i], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=colors[i], linewidth=0.75)

    significant_added = False  # Flag to track if a significance bar is added

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value[mvt_name][sig_key]

        if p_value < 0.05:
            significant_added = True
            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1
            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1 + 0.1, xmax=pos_plot[j + 1] + i * 0.1 - 0.1, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1 + 0.1, ymin=line_y - 0.9, ymax=line_y, colors=colors[i], linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1 - 0.1, ymin=line_y - 0.9, ymax=line_y, colors=colors[i], linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 0.8, p_text, ha='center', va='bottom', color=colors[i], fontsize=7)

    if significant_added:
        line_y += y_increment  # Increment line_y only if a significance bar was added
        significant_added = False

    # Ajouter la barre de significativité entre Takeoff et Landing
    p_value_tl = significant_value[mvt_name]['takeoff_landing']
    if p_value_tl < 0.05:
        significant_added = True
        p_text_tl = "***" if p_value_tl < 0.001 else "**" if p_value_tl < 0.01 else "*"
        mid_point_tl = (pos_plot[0] + pos_plot[2]) / 2 + i * 0.1

        ax.hlines(y=line_y, xmin=pos_plot[0] + i * 0.1 * 1.1, xmax=pos_plot[2] + i * 0.1 * 0.9,
                  colors=colors[i], linestyles='solid', lw=1)
        ax.vlines(x=pos_plot[0] + i * 0.1 * 1.1, ymin=line_y - 0.9, ymax=line_y, colors=colors[i],
                  linestyles='solid', lw=1)
        ax.vlines(x=pos_plot[2] + i * 0.1 * 0.9, ymin=line_y - 0.9, ymax=line_y, colors=colors[i],
                  linestyles='solid', lw=1)
        ax.text(mid_point_tl, line_y - 0.8, p_text_tl, ha='center', va='bottom', color=colors[i], fontsize=7)

    if significant_added:
        line_y += y_increment  # Increment line_y only if a significance bar was added

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)

# Réduire l'épaisseur du cadre du graphique
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x_empty)
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/rotation_all_analysis.png", dpi=1000)
plt.show()
##

plt.figure(figsize=(363 / 96, 242 / 96))

initial_ticks = np.arange(0, 50, 10)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.0f}" for tick in initial_ticks]
ax = plt.gca()

categories = all_data['Timing'].cat.categories
pos_plot = np.array([1, 5, 9])

i_plot = 0

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)['Std'].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)['Std'].std()


    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=colors[i], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=colors[i], linewidth=0.75)

    y_max = all_data["Std"].max()

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value[mvt_name][sig_key]
        line_y = (y_max - 5) + i_plot

        if p_value < 0.05:
            i_plot += 4

            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1
            # line_y = (y_max-5) + 0.03 * i_plot
            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1, xmax=pos_plot[j + 1] + i * 0.1, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1, ymin=line_y - 0.9, ymax=line_y, colors=colors[i], linestyles='solid',
                      lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1, ymin=line_y - 0.9, ymax=line_y, colors=colors[i],
                      linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 1.4, p_text, ha='center', va='bottom', color=colors[i], fontsize=7)


    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)
# Réduire l'épaisseur du cadre du graphique
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x_empty)
# plt.xticks([1.5, 5.5, 9.5], categories)
# plt.title('Pelvis Rotation')
# plt.xlabel('Timing')
# plt.ylabel('SD')
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
# plt.legend(title='Acrobatics Code', bbox_to_anchor=(1.005, 1), loc=2, borderaxespad=0.)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/rotation.png", dpi=1000)


# Créer une nouvelle figure pour la légende
fig_legend = plt.figure(figsize=(10, 2))  # Taille de la figure pour la légende
handles, labels = ax.get_legend_handles_labels()
fig_legend.legend(handles, labels, loc='center', title='Acrobatics Code', ncol=5)

# Enregistrer la légende dans une image séparée
fig_legend.savefig("/home/lim/Documents/StageMathieu/meeting/legend.png", dpi=1000, bbox_inches='tight')


print("Statistical test for all acrobatics")
posthoc_results_total = perform_kruskal_and_dunn(all_data, 'Std', 'Timing')
significant_value_takeoff_75 = posthoc_results_total.loc["Takeoff", "75%"]
significant_value_75_landing = posthoc_results_total.loc["75%", "Landing"]
significant_value_takeoff_landing = posthoc_results_total.loc["Takeoff", "Landing"]

plt.figure(figsize=(363 / 96, 242 / 96))

initial_ticks = np.arange(0, 50, 10)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.0f}" for tick in initial_ticks]

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

plt.errorbar(x=pos_plot, y=means, yerr=std_devs, fmt='o', capsize=5, elinewidth=0.5, capthick=0.5, color="black", markersize=5)
plt.plot(pos_plot, means, '-', color="black", linewidth=1)

y_max = all_data["Std"].max() - 0.4

for j in range(len(pos_plot) - 1):
    p_value = significant_value_takeoff_75 if j == 0 else significant_value_75_landing

    if p_value < 0.05:
        p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2
        line_y = 45

        ax.hlines(y=line_y, xmin=pos_plot[j] + 0.1, xmax=pos_plot[j + 1] - 0.1,
                  linestyles='solid', lw=1, color="black")
        ax.vlines(x=pos_plot[j] + 0.1, ymin=line_y - 0.9, ymax=line_y, linestyles='solid',
                  lw=1, color="black")
        ax.vlines(x=pos_plot[j + 1] - 0.1, ymin=line_y - 0.9, ymax=line_y,
                  linestyles='solid', lw=1, color="black")
        ax.text(mid_point, line_y, p_text, ha='center', va='bottom', fontsize=11)
    ax.set_ylim(0, 49)

    # Ajouter la barre de significativité entre Takeoff et Landing
p_value_tl = significant_value_takeoff_landing
if p_value_tl < 0.05:
    significant_added = True
    p_text_tl = "***" if p_value_tl < 0.001 else "**" if p_value_tl < 0.01 else "*"
    mid_point_tl = (pos_plot[0] + pos_plot[2]) / 2 + i * 0.1

    ax.hlines(y=line_y, xmin=pos_plot[0] + i * 0.1 * 1.1, xmax=pos_plot[2] + i * 0.1 * 0.9,
              colors="black", linestyles='solid', lw=1)
    ax.vlines(x=pos_plot[0] + i * 0.1 * 1.1, ymin=line_y - 0.9, ymax=line_y, colors="black",
              linestyles='solid', lw=1)
    ax.vlines(x=pos_plot[2] + i * 0.1 * 0.9, ymin=line_y - 0.9, ymax=line_y, colors="black",
              linestyles='solid', lw=1)
    ax.text(mid_point_tl, line_y - 0.8, p_text_tl, ha='center', va='bottom', color="black", fontsize=7)

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)


# Réduire l'épaisseur du cadre du graphique
for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1, 5, 9], labels_x_empty)
# plt.title('Pelvis Rotation')
# plt.xlabel('Timing')
# plt.ylabel('SD')
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/mean_rotation.png", dpi=1000)

plt.show()
