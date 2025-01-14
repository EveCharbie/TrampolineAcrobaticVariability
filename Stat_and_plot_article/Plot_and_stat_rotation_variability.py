import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from TrampolineAcrobaticVariability.Function.Function_stat import perform_anova_and_tukey, perform_kruskal_and_dunn
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import extract_identifier
home_path = "/home/lim/Documents/StageMathieu/Tab_result3/"

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

name_to_color = {
    '4-': '#1f77b4',
    '4-o': '#ff7f0e',
    '8--o': '#2ca02c',
    '8-1<': '#d62728',
    '8-1o': '#9467bd',
    '41': '#8c564b',
    '811<': '#e377c2',
    '41o': '#7f7f7f',
    '8-3<': '#bcbd22',
    '42': '#17becf',
    '822': '#aec7e8',
    '831<': '#ffbb78',
    '43': '#98df8a',
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

    print(f"\n===== Movement {mvt_name} is running =====")

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

    significant_value.loc["takeoff_75", mvt_name] = posthoc_results["P-Value"][0]
    significant_value.loc["75_landing", mvt_name] = posthoc_results["P-Value"][2]
    significant_value.loc["takeoff_landing", mvt_name] = posthoc_results["P-Value"][1]

    all_data = pd.concat([all_data, data_specific], ignore_index=True)

all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)
all_data['Std'] = np.degrees(all_data['Std'])

plt.figure(figsize=(363 / 96, 242 / 96))

initial_ticks = np.arange(0, 50, 10)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.0f}" for tick in initial_ticks]
ax = plt.gca()

categories = all_data['Timing'].cat.categories
pos_plot = np.array([1, 5, 9])

y_increment = 4
line_y = all_data["Std"].max() - 5

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)['Std'].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)['Std'].std()

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=name_to_color[mvt_name], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=name_to_color[mvt_name], linewidth=0.75)

    significant_added = False

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value[mvt_name][sig_key]

        if p_value < 0.05:
            significant_added = True
            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1
            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1 + 0.1, xmax=pos_plot[j + 1] + i * 0.1 - 0.1, colors=name_to_color[mvt_name],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1 + 0.1, ymin=line_y - 0.9, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1 - 0.1, ymin=line_y - 0.9, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 0.8, p_text, ha='center', va='bottom', color=name_to_color[mvt_name], fontsize=7)

    if significant_added:
        line_y += y_increment
        significant_added = False

    p_value_tl = significant_value[mvt_name]['takeoff_landing']
    if p_value_tl < 0.05:
        significant_added = True
        p_text_tl = "***" if p_value_tl < 0.001 else "**" if p_value_tl < 0.01 else "*"
        mid_point_tl = (pos_plot[0] + pos_plot[2]) / 2 + i * 0.1

        ax.hlines(y=line_y, xmin=pos_plot[0] + i * 0.1 * 1.1, xmax=pos_plot[2] + i * 0.1 * 0.9,
                  colors=name_to_color[mvt_name], linestyles='solid', lw=1)
        ax.vlines(x=pos_plot[0] + i * 0.1 * 1.1, ymin=line_y - 0.9, ymax=line_y, colors=name_to_color[mvt_name],
                  linestyles='solid', lw=1)
        ax.vlines(x=pos_plot[2] + i * 0.1 * 0.9, ymin=line_y - 0.9, ymax=line_y, colors=name_to_color[mvt_name],
                  linestyles='solid', lw=1)
        ax.text(mid_point_tl, line_y - 0.8, p_text_tl, ha='center', va='bottom', color=name_to_color[mvt_name], fontsize=7)

    if significant_added:
        line_y += y_increment

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x_empty)
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/rotation_all_analysis.svg", format='svg')


fig_legend = plt.figure(figsize=(10, 2))
handles, labels = ax.get_legend_handles_labels()
fig_legend.legend(handles, labels, loc='center', title='Acrobatics Code', ncol=5)

fig_legend.savefig("/home/lim/Documents/StageMathieu/meeting/legend.svg", format='svg', bbox_inches='tight')


print("\n=====  Statistical test for all acrobatics ===== ")
posthoc_results_total = perform_kruskal_and_dunn(all_data, 'Std', 'Timing')
significant_value_takeoff_75 = posthoc_results_total["P-Value"][0]
significant_value_75_landing = posthoc_results_total["P-Value"][2]
significant_value_takeoff_landing = posthoc_results_total["P-Value"][1]

plt.figure(figsize=(363 / 96, 242 / 96))

initial_ticks = np.arange(0, 50, 10)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.0f}" for tick in initial_ticks]

ax = plt.gca()
pos_plot = np.array([1, 5, 9])

means = all_data.groupby('Timing', observed=True)['Std'].mean()
std_devs = all_data.groupby('Timing', observed=True)['Std'].std()

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

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1, 5, 9], labels_x_empty)
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/mean_rotation.svg", format='svg')

plt.show()
