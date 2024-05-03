import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, mannwhitneyu
import matplotlib.patches as mpatches
import numpy as np

data_41 = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_41_times_75.csv')
data_41o = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_41o_times_75.csv')
data_42 = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_42_times_75.csv')
data_43 = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_43_times_75.csv')

levels = ['41', '41o', '42', '43']

expertise_labels_41 = data_41.iloc[0]
expertise_labels_41o = data_41o.iloc[0]
expertise_labels_42 = data_42.iloc[0]
expertise_labels_43 = data_43.iloc[0]

data_41 = data_41.iloc[1:]
data_41o = data_41o.iloc[1:]
data_42 = data_42.iloc[1:]
data_43 = data_43.iloc[1:]

data_41 = data_41.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_41['Expertise'] = data_41['Participant'].map(expertise_labels_41)

data_41o = data_41o.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_41o['Expertise'] = data_41o['Participant'].map(expertise_labels_41o)

data_42 = data_42.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_42['Expertise'] = data_42['Participant'].map(expertise_labels_42)

data_43 = data_43.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_43['Expertise'] = data_43['Participant'].map(expertise_labels_43)

data_41['Difficulty'] = '41'
data_41o['Difficulty'] = '41o'
data_42['Difficulty'] = '42'
data_43['Difficulty'] = '43'

combined_data = pd.concat([data_41, data_41o, data_42, data_43], ignore_index=True)
combined_data['Score'] = pd.to_numeric(combined_data['Score'], errors='coerce')


elite_data = combined_data[combined_data['Expertise'] == 'Elite']
subelite_data = combined_data[combined_data['Expertise'] == 'SubElite']

std_inter_elite = []
std_inter_subelite = []
std_mean_intra_elite = []
std_mean_intra_subelite = []

for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    std_level_elite = np.std(data_elite)
    std_level_subelite = np.std(data_subelite)

    std_inter_elite.append(std_level_elite) # Variability inter individu
    std_inter_subelite.append(std_level_subelite) # Variability inter individu

    std_intra_participant_elite = []
    std_intra_participant_subelite = []

    for participant in np.unique(elite_data['Participant']):
        data_participant_elite = elite_data[(elite_data['Difficulty'] == level) & (elite_data['Participant'] == participant)][
            'Score'].dropna()
        std_participant_elite = np.std(data_participant_elite)
        std_intra_participant_elite.append(std_participant_elite)

    for participant in np.unique(subelite_data['Participant']):

        data_participant_subelite = subelite_data[(subelite_data['Difficulty'] == level) & (subelite_data['Participant'] == participant)][
            'Score'].dropna()
        std_participant_subelite = np.std(data_participant_subelite)
        std_intra_participant_subelite.append(std_participant_subelite)

    std_mean_intra_elite.append(np.mean(std_intra_participant_elite)) # Variability intra individu
    std_mean_intra_subelite.append(np.mean(std_intra_participant_subelite)) # Variability intra individu


positions_elite = [1, 5, 9, 13]
positions_subelite = [2, 6, 10, 14]

fig, ax = plt.subplots(figsize=(12, 8))

# Définir les couleurs des boxplots
colors = {'Elite': 'lightblue', 'SubElite': 'lightgreen'}
median_color = 'black'

# Boxplots pour 'Elite'
for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    ax.boxplot(data_elite, positions=[positions_elite[i]], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors['Elite']),
               medianprops=dict(color=median_color))

# Boxplots pour 'SubElite'
for i, level in enumerate(levels):
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    ax.boxplot(data_subelite, positions=[positions_subelite[i]], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors['SubElite']),
               medianprops=dict(color=median_color))

for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    stat, pvalue = mannwhitneyu(data_elite, data_subelite)
    y_max = max(data_elite.max(), data_subelite.max()) + 5

    # Formatter le texte de l'annotation en fonction de la valeur p
    p_text = f"p < 0.001" if pvalue < 0.001 else f"p = {pvalue:.3f}"

    # Dessiner les lignes horizontales et verticales
    mid_point = (positions_elite[i] + positions_subelite[i]) / 2
    line_y = y_max + 1
    ax.hlines(y=line_y, xmin=positions_elite[i], xmax=positions_subelite[i], colors="black", linestyles='solid', lw=1)
    ax.vlines(x=positions_elite[i], ymin=line_y - 0.5, ymax=line_y + 0.5, colors="black", linestyles='solid', lw=1)
    ax.vlines(x=positions_subelite[i], ymin=line_y - 0.5, ymax=line_y + 0.5, colors="black", linestyles='solid', lw=1)

    ax.annotate(p_text, xy=(mid_point, line_y + 0.5), textcoords="offset points", xytext=(0, 5), ha='center')

# Légende pour les couleurs
legend_patches = [mpatches.Patch(color=colors['Elite'], label='Elite'),
                  mpatches.Patch(color=colors['SubElite'], label='SubElite')]
ax.legend(handles=legend_patches, title='Expertise')
ax.set_ylim(40, 100)
ax.set_xticks([1.5, 5.5, 9.5, 13.5])
ax.set_xticklabels(['41', '41o', '42', '43'])
ax.set_title('Comparison of % Time to reach 3/4 twist by Expertise')
ax.set_xlabel('Difficulty Level')
ax.set_ylabel('Time %')

# plt.show()


###


data_41 = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_41_times_10.csv')
data_41o = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_41o_times_10.csv')
data_42 = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_42_times_10.csv')
data_43 = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_43_times_10.csv')



expertise_labels_41 = data_41.iloc[0]
expertise_labels_41o = data_41o.iloc[0]
expertise_labels_42 = data_42.iloc[0]
expertise_labels_43 = data_43.iloc[0]

data_41 = data_41.iloc[1:]
data_41o = data_41o.iloc[1:]
data_42 = data_42.iloc[1:]
data_43 = data_43.iloc[1:]

data_41 = data_41.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_41['Expertise'] = data_41['Participant'].map(expertise_labels_41)

data_41o = data_41o.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_41o['Expertise'] = data_41o['Participant'].map(expertise_labels_41o)

data_42 = data_42.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_42['Expertise'] = data_42['Participant'].map(expertise_labels_42)

data_43 = data_43.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_43['Expertise'] = data_43['Participant'].map(expertise_labels_43)

data_41['Difficulty'] = '41'
data_41o['Difficulty'] = '41o'
data_42['Difficulty'] = '42'
data_43['Difficulty'] = '43'

combined_data = pd.concat([data_41, data_41o, data_42, data_43], ignore_index=True)
combined_data['Score'] = pd.to_numeric(combined_data['Score'], errors='coerce')


elite_data = combined_data[combined_data['Expertise'] == 'Elite']
subelite_data = combined_data[combined_data['Expertise'] == 'SubElite']

std_inter_elite = []
std_inter_subelite = []
std_mean_intra_elite = []
std_mean_intra_subelite = []

for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    std_level_elite = np.std(data_elite)
    std_level_subelite = np.std(data_subelite)

    std_inter_elite.append(std_level_elite) # Variability inter individu
    std_inter_subelite.append(std_level_subelite) # Variability inter individu

    std_intra_participant_elite = []
    std_intra_participant_subelite = []

    for participant in np.unique(elite_data['Participant']):
        data_participant_elite = elite_data[(elite_data['Difficulty'] == level) & (elite_data['Participant'] == participant)][
            'Score'].dropna()
        std_participant_elite = np.std(data_participant_elite)
        std_intra_participant_elite.append(std_participant_elite)

    for participant in np.unique(subelite_data['Participant']):

        data_participant_subelite = subelite_data[(subelite_data['Difficulty'] == level) & (subelite_data['Participant'] == participant)][
            'Score'].dropna()
        std_participant_subelite = np.std(data_participant_subelite)
        std_intra_participant_subelite.append(std_participant_subelite)

    std_mean_intra_elite.append(np.mean(std_intra_participant_elite)) # Variability intra individu
    std_mean_intra_subelite.append(np.mean(std_intra_participant_subelite)) # Variability intra individu


fig, ax = plt.subplots(figsize=(12, 8))

# Définir les couleurs des boxplots
colors = {'Elite': 'lightblue', 'SubElite': 'lightgreen'}
median_color = 'black'

# Boxplots pour 'Elite'
for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    ax.boxplot(data_elite, positions=[positions_elite[i]], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors['Elite']),
               medianprops=dict(color=median_color))

# Boxplots pour 'SubElite'
for i, level in enumerate(levels):
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    ax.boxplot(data_subelite, positions=[positions_subelite[i]], widths=0.6, patch_artist=True,
               boxprops=dict(facecolor=colors['SubElite']),
               medianprops=dict(color=median_color))

for i, level in enumerate(levels):
    data_elite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'Elite')][
        'Score'].dropna()
    data_subelite = combined_data[(combined_data['Difficulty'] == level) & (combined_data['Expertise'] == 'SubElite')][
        'Score'].dropna()
    stat, pvalue = mannwhitneyu(data_elite, data_subelite)
    y_max = max(data_elite.max(), data_subelite.max()) + 5

    # Formatter le texte de l'annotation en fonction de la valeur p
    p_text = f"p < 0.001" if pvalue < 0.001 else f"p = {pvalue:.3f}"

    # Dessiner les lignes horizontales et verticales
    mid_point = (positions_elite[i] + positions_subelite[i]) / 2
    line_y = y_max + 1
    ax.hlines(y=line_y, xmin=positions_elite[i], xmax=positions_subelite[i], colors="black", linestyles='solid', lw=1)
    ax.vlines(x=positions_elite[i], ymin=line_y - 0.5, ymax=line_y + 0.5, colors="black", linestyles='solid', lw=1)
    ax.vlines(x=positions_subelite[i], ymin=line_y - 0.5, ymax=line_y + 0.5, colors="black", linestyles='solid', lw=1)

    ax.annotate(p_text, xy=(mid_point, line_y + 0.5), textcoords="offset points", xytext=(0, 5), ha='center')

# Légende pour les couleurs
legend_patches = [mpatches.Patch(color=colors['Elite'], label='Elite'),
                  mpatches.Patch(color=colors['SubElite'], label='SubElite')]
ax.legend(handles=legend_patches, title='Expertise')
ax.set_ylim(5, 65)
ax.set_xticks([1.5, 5.5, 9.5, 13.5])
ax.set_xticklabels(['41', '41o', '42', '43'])
ax.set_title('Comparison of % Time to reach 1/10 twist by Expertise')
ax.set_xlabel('Difficulty Level')
ax.set_ylabel('Time %')

plt.show()
