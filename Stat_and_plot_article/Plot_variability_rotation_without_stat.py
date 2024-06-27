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

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)

order_index = {key: index for index, key in enumerate(order)}
rotation_files = sorted(rotation_files, key=lambda x: order_index.get(extract_identifier(x), float('inf')))

all_data = pd.DataFrame()

for file in rotation_files:
    data = pd.read_csv(file)
    data_specific = data[data['Timing'].isin(['75%', 'Landing', 'Takeoff'])]
    mvt_name = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID
    data_specific['Source'] = mvt_name

    print(f"===== Movement {mvt_name} is running =====")

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

colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(all_data['Source'].unique())))

significance_points = []

for i, mvt_name in enumerate(order):
    name_acro = full_name_acrobatics[mvt_name]

    filtered_data = all_data[all_data['Source'].str.contains(mvt_name)]

    means = filtered_data.groupby('Timing', observed=True)['Std'].mean()
    std_devs = filtered_data.groupby('Timing', observed=True)['Std'].std()

    plt.errorbar(x=pos_plot + i * 0.1, y=means, yerr=std_devs, fmt='o', label=name_acro,
                 color=colors[i], capsize=5, elinewidth=0.5, capthick=0.5, markersize=3)

    plt.plot(pos_plot + i * 0.1, means, '-', color=colors[i], linewidth=0.75)

ax.set_yticks(current_ticks)
ax.set_yticklabels(current_labels)
ax.tick_params(axis='y', labelsize=8, width=0.4)
ax.set_ylim(0, 50)

for spine in ax.spines.values():
    spine.set_linewidth(0.5)

plt.xticks([1.5, 5.5, 9.5], labels_x_empty)
plt.subplots_adjust(left=0.090, right=0.965, top=0.982, bottom=0.102)
plt.savefig("/home/lim/Documents/StageMathieu/meeting/rotation_without_significance_bar.png", dpi=1000)
plt.show()
