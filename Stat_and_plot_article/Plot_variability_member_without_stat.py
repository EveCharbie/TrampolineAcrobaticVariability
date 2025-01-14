import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import extract_identifier
from TrampolineAcrobaticVariability.Function.Function_stat import (prepare_data)


all_data = pd.DataFrame()

labels_x = ["T$_{TO}$", "T$_{75}$", "T$_{LA}$"]
labels_x_empty = [" ", " ", " "]

home_path = "/home/lim/Documents/StageMathieu/Tab_result3/"
order = ['8-1o', '8-1<', '811<', '41', '41o', '8-3<', '42', '831<', '822', '43', '4-', '4-o', '8--o']
mvt_to_color = {
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
order = order[:-3]

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

index = ['takeoff_75', '75_landing', 'takeoff_landing']
body_parts = ['upper_body', 'lower_body']

position_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        mvt_name_this_time = file.split('_')[1]
        if 'position' in file and mvt_name_this_time in order:
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

    all_data = pd.concat([all_data, data_prepared], ignore_index=True)


all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)
all_data['Timing'] = all_data['Timing'].cat.rename_categories({"75%": "T75"})

categories = all_data['Timing'].cat.categories
pos_plot = np.array([1, 5, 9])

## Plot upper body

plt.figure(figsize=(363 / 96, 242 / 96))
initial_ticks = np.arange(0, 1.4, 0.2)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.1f}" for tick in initial_ticks]
ax = plt.gca()

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)["upper_body"].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)["upper_body"].std()

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=mvt_to_color[mvt_name], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=mvt_to_color[mvt_name], linewidth=0.75)

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)
    ax.set_ylim(0, 1.7)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x_empty)
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/upper_body_without_significance_bar.png", dpi=1000)


## Plot lower body

plt.figure(figsize=(363 / 96, 242 / 96))
initial_ticks = np.arange(0, 0.6, 0.2)
current_ticks = list(initial_ticks)
current_labels = [f"{tick:.1f}" for tick in initial_ticks]
ax = plt.gca()

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)["lower_body"].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)["lower_body"].std()

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=mvt_to_color[mvt_name], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=mvt_to_color[mvt_name], linewidth=0.75)
    ax.set_ylim(0, 0.6)
    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels)
    ax.tick_params(axis='y', labelsize=8, width=0.4)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x)
plt.xlabel('Timing')
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/lower_body_without_significance_bar.png", dpi=300)
plt.show()








