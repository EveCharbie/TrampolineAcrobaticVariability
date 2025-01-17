import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levene, shapiro
import numpy as np
import os
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import extract_identifier
from TrampolineAcrobaticVariability.Function.Function_stat import (perform_anova_and_tukey,
                                                                   perform_kruskal_and_dunn,
                                                                   prepare_data)


all_data = pd.DataFrame()

labels_x = ["T$_{TO}$", "T$_{75}$", "T$_{LA}$"]
labels_x_empty = [" ", " ", " "]

home_path = "/home/lim/Documents/StageMathieu/Tab_result3/"
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


index = ['takeoff_75', '75_landing', 'takeoff_landing']
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

    print(f"\n===== Movement {mvt_name} is running =====")

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

    significant_value_upper_body.loc["takeoff_75", mvt_name] = posthoc_results_upper_body["P-Value"][0]
    significant_value_upper_body.loc["75_landing", mvt_name] = posthoc_results_upper_body["P-Value"][2]
    significant_value_upper_body.loc["takeoff_landing", mvt_name] = posthoc_results_upper_body["P-Value"][1]

    significant_value_lower_body.loc["takeoff_75", mvt_name] = posthoc_results_lower_body["P-Value"][0]
    significant_value_lower_body.loc["75_landing", mvt_name] = posthoc_results_lower_body["P-Value"][2]
    significant_value_lower_body.loc["takeoff_landing", mvt_name] = posthoc_results_lower_body["P-Value"][1]

    all_data = pd.concat([all_data, data_prepared], ignore_index=True)


all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)
all_data['Timing'] = all_data['Timing'].cat.rename_categories({"75%": "T75"})

categories = all_data['Timing'].cat.categories
pos_plot = np.array([1, 5, 9])


## Plot upper body each acrobatics

plt.figure(figsize=(363 / 96, 242 / 96))
initial_ticks = np.arange(0, 1.4, 0.2)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.1f}" for tick in initial_ticks]
ax = plt.gca()
i_plot = 0

y_max = all_data["upper_body"].max()
line_y = y_max + i_plot
y_increment = 0.1

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)["upper_body"].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)["upper_body"].std()

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=name_to_color[mvt_name], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=name_to_color[mvt_name], linewidth=0.75)

    significant_added = False

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value_upper_body[mvt_name][sig_key]

        if p_value < 0.05:
            significant_added = True
            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1

            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1 + 0.1, xmax=pos_plot[j + 1] + i * 0.1 -0.1, colors=name_to_color[mvt_name],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1+0.1, ymin=line_y - 0.04, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1-0.1, ymin=line_y - 0.04, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 0.035, p_text, ha='center', va='bottom', color=name_to_color[mvt_name], fontsize=7)

    if significant_added:
        line_y += y_increment
        significant_added = False

    p_value_tl = significant_value_upper_body[mvt_name]['takeoff_landing']
    if p_value_tl < 0.05:
        significant_added = True
        p_text_tl = "***" if p_value_tl < 0.001 else "**" if p_value_tl < 0.01 else "*"
        mid_point_tl = (pos_plot[0] + pos_plot[2]) / 2 + i * 0.1

        ax.hlines(y=line_y, xmin=pos_plot[0] + i * 0.1 + 0.1, xmax=pos_plot[2] + i * 0.1 - 0.1, colors=name_to_color[mvt_name],
                  linestyles='solid', lw=1)
        ax.vlines(x=pos_plot[0] + i * 0.1 + 0.1, ymin=line_y - 0.04, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid', lw=1)
        ax.vlines(x=pos_plot[2] + i * 0.1 - 0.1, ymin=line_y - 0.04, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid', lw=1)
        ax.text(mid_point_tl, line_y - 0.035, p_text_tl, ha='center', va='bottom', color=name_to_color[mvt_name], fontsize=7)

    if significant_added:
        line_y += y_increment

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)
    ax.set_ylim(0, 2.5)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x_empty)
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/upper_body_all_analysis.svg", format='svg')


## Plot lower body each acrobatics

plt.figure(figsize=(363 / 96, 242 / 96))
initial_ticks = np.arange(0, 0.6, 0.2)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.1f}" for tick in initial_ticks]
ax = plt.gca()
i_plot = 0
y_max = all_data["lower_body"].max()
line_y = y_max + i_plot
y_increment = 0.05

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)["lower_body"].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)["lower_body"].std()

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=name_to_color[mvt_name], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=name_to_color[mvt_name], linewidth=0.75)

    significant_added = False

    for j in range(len(pos_plot) - 1):
        sig_key = f"takeoff_75" if j == 0 else f"75_landing"
        p_value = significant_value_lower_body[mvt_name][sig_key]

        if p_value < 0.05:
            significant_added = True
            p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
            mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2 + i * 0.1

            ax.hlines(y=line_y, xmin=pos_plot[j] + i * 0.1 + 0.1, xmax=pos_plot[j + 1] + i * 0.1 - 0.1,
                      colors=name_to_color[mvt_name],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j] + i * 0.1 + 0.1, ymin=line_y - 0.02, ymax=line_y, colors=name_to_color[mvt_name],
                      linestyles='solid', lw=1)
            ax.vlines(x=pos_plot[j + 1] + i * 0.1 - 0.1, ymin=line_y - 0.02, ymax=line_y, colors=name_to_color[mvt_name],
                      linestyles='solid', lw=1)
            ax.text(mid_point, line_y - 0.015, p_text, ha='center', va='bottom', color=name_to_color[mvt_name], fontsize=7)

    if significant_added:
        line_y += y_increment
        significant_added = False

    p_value_tl = significant_value_lower_body[mvt_name]['takeoff_landing']
    if p_value_tl < 0.05:
        significant_added = True
        p_text_tl = "***" if p_value_tl < 0.001 else "**" if p_value_tl < 0.01 else "*"
        mid_point_tl = (pos_plot[0] + pos_plot[2]) / 2 + i * 0.1

        ax.hlines(y=line_y, xmin=pos_plot[0] + i * 0.1 + 0.1, xmax=pos_plot[2] + i * 0.1 - 0.1, colors=name_to_color[mvt_name],
                  linestyles='solid', lw=1)
        ax.vlines(x=pos_plot[0] + i * 0.1 + 0.1, ymin=line_y - 0.02, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid',
                  lw=1)
        ax.vlines(x=pos_plot[2] + i * 0.1 - 0.1, ymin=line_y - 0.02, ymax=line_y, colors=name_to_color[mvt_name], linestyles='solid',
                  lw=1)
        ax.text(mid_point_tl, line_y - 0.015, p_text_tl, ha='center', va='bottom', color=name_to_color[mvt_name], fontsize=7)

    if significant_added:
        line_y += y_increment

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x)
plt.xlabel('Timing')
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/lower_body_all_analysis.svg", format='svg')


print("\n===== Statistical test for all acrobatics (upper body) =====")
posthoc_results_total = perform_kruskal_and_dunn(all_data, 'upper_body', 'Timing')
significant_value_takeoff_75 = posthoc_results_total["P-Value"][0]
significant_value_75_landing = posthoc_results_total["P-Value"][2]
significant_value_takeoff_landing = posthoc_results_total["P-Value"][1]



## Upper body mean

plt.figure(figsize=(363 / 96, 242 / 96))
initial_ticks = np.arange(0, 1.2, 0.2)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.1f}" for tick in initial_ticks]
ax = plt.gca()
pos_plot = np.array([1, 5, 9])

y_max = all_data["upper_body"].max()
line_y = y_max - 0.15
y_increment = 0.02

means = all_data.groupby('Timing', observed=True)['upper_body'].mean()
std_devs = all_data.groupby('Timing', observed=True)['upper_body'].std()

plt.errorbar(x=pos_plot, y=means, yerr=std_devs, fmt='o', capsize=5, elinewidth=0.5, capthick=0.5, color="black", markersize=5)
plt.plot(pos_plot, means, '-', color="black", linewidth=1)


for j in range(len(pos_plot) - 1):
    p_value = significant_value_takeoff_75 if j == 0 else significant_value_75_landing

    if p_value < 0.05:
        p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2

        ax.hlines(y=line_y, xmin=pos_plot[j] + 0.1, xmax=pos_plot[j + 1] - 0.1,
                  linestyles='solid', lw=1, color="black")
        ax.vlines(x=pos_plot[j] + 0.1, ymin=line_y - 0.04, ymax=line_y, linestyles='solid',
                  lw=1, color="black")
        ax.vlines(x=pos_plot[j + 1] - 0.1, ymin=line_y - 0.04, ymax=line_y,
                  linestyles='solid', lw=1, color="black")
        ax.text(mid_point, line_y, p_text, ha='center', va='bottom', fontsize=11)

line_y += y_increment + 0.2
p_value_tl = significant_value_takeoff_landing
if p_value_tl < 0.05:
    significant_added = True
    p_text_tl = "***" if p_value_tl < 0.001 else "**" if p_value_tl < 0.01 else "*"
    mid_point_tl = (pos_plot[0] + pos_plot[2]) / 2

    ax.hlines(y=line_y, xmin=pos_plot[0] + 0.1, xmax=pos_plot[2] - 0.1,
              colors="black", linestyles='solid', lw=1)
    ax.vlines(x=pos_plot[0] + 0.1, ymin=line_y - 0.04, ymax=line_y, colors="black",
              linestyles='solid', lw=1)
    ax.vlines(x=pos_plot[2] - 0.1, ymin=line_y - 0.04, ymax=line_y, colors="black",
              linestyles='solid', lw=1)
    ax.text(mid_point_tl, line_y , p_text_tl, ha='center', va='bottom', color="black", fontsize=11)

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)
    ax.set_ylim(0, 1.55)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1, 5, 9], labels_x_empty)
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/mean_upper_body.svg", format='svg')


print("\n===== Statistical test for all acrobatics (lower body) =====")
posthoc_results_total = perform_kruskal_and_dunn(all_data, 'lower_body', 'Timing')
significant_value_takeoff_75 = posthoc_results_total["P-Value"][0]
significant_value_75_landing = posthoc_results_total["P-Value"][2]
significant_value_takeoff_landing = posthoc_results_total["P-Value"][1]


## Lower body mean
plt.figure(figsize=(363 / 96, 242 / 96))
initial_ticks = np.arange(0, 0.6, 0.2)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.1f}" for tick in initial_ticks]
ax = plt.gca()
pos_plot = np.array([1, 5, 9])

y_max = all_data["lower_body"].max()
line_y = y_max - 0.1
y_increment = 0.01

means = all_data.groupby('Timing', observed=True)['lower_body'].mean()
std_devs = all_data.groupby('Timing', observed=True)['lower_body'].std()

plt.errorbar(x=pos_plot, y=means, yerr=std_devs, fmt='o', capsize=5, elinewidth=0.5, capthick=0.5, color="black", markersize=5)
plt.plot(pos_plot, means, '-', color="black", linewidth=1)

for j in range(len(pos_plot) - 1):
    p_value = significant_value_takeoff_75 if j == 0 else significant_value_75_landing

    if p_value < 0.05:
        p_text = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
        mid_point = (pos_plot[j] + pos_plot[j + 1]) / 2

        ax.hlines(y=line_y, xmin=pos_plot[j] + 0.1, xmax=pos_plot[j + 1] - 0.1,
                  linestyles='solid', lw=1, color="black")
        ax.vlines(x=pos_plot[j] + 0.1, ymin=line_y - 0.015, ymax=line_y, linestyles='solid',
                  lw=1, color="black")
        ax.vlines(x=pos_plot[j + 1] - 0.1, ymin=line_y - 0.015, ymax=line_y,
                  linestyles='solid', lw=1, color="black")
        ax.text(mid_point, line_y, p_text, ha='center', va='bottom', fontsize=11)

line_y += y_increment + 0.1
p_value_tl = significant_value_takeoff_landing
if p_value_tl < 0.05:
    significant_added = True
    p_text_tl = "***" if p_value_tl < 0.001 else "**" if p_value_tl < 0.01 else "*"
    mid_point_tl = (pos_plot[0] + pos_plot[2]) / 2

    ax.hlines(y=line_y, xmin=pos_plot[0] + 0.1, xmax=pos_plot[2] - 0.1,
              colors="black", linestyles='solid', lw=1)
    ax.vlines(x=pos_plot[0] + 0.1, ymin=line_y - 0.015, ymax=line_y, colors="black",
              linestyles='solid', lw=1)
    ax.vlines(x=pos_plot[2] - 0.1, ymin=line_y - 0.015, ymax=line_y, colors="black",
              linestyles='solid', lw=1)
    ax.text(mid_point_tl, line_y , p_text_tl, ha='center', va='bottom', color="black", fontsize=11)
    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)
    ax.set_ylim(0, 0.6)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1, 5, 9], labels_x)
plt.xlabel('Timing')
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/mean_lower_body.svg", format='svg')

plt.show()