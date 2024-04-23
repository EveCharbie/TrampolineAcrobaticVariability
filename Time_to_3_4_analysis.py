import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, mannwhitneyu
import matplotlib.patches as mpatches

data_41 = pd.read_csv('/home/lim/Documents/StageMathieu/results_41_times.csv')
data_42 = pd.read_csv('/home/lim/Documents/StageMathieu/results_42_times.csv')
data_43 = pd.read_csv('/home/lim/Documents/StageMathieu/results_43_times.csv')

expertise_labels_41 = data_41.iloc[0]
expertise_labels_42 = data_42.iloc[0]
expertise_labels_43 = data_43.iloc[0]

data_41 = data_41.iloc[1:]
data_42 = data_42.iloc[1:]
data_43 = data_43.iloc[1:]

# Transformer les données pour avoir 'Participant' et 'Score'
data_41 = data_41.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_41['Expertise'] = data_41['Participant'].map(expertise_labels_41)

data_42 = data_42.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_42['Expertise'] = data_42['Participant'].map(expertise_labels_42)

data_43 = data_43.melt(var_name='Participant', value_name='Score', ignore_index=False)
data_43['Expertise'] = data_43['Participant'].map(expertise_labels_43)

data_41['Difficulty'] = '41'
data_42['Difficulty'] = '42'
data_43['Difficulty'] = '43'

combined_data = pd.concat([data_41, data_42, data_43], ignore_index=True)
combined_data['Score'] = pd.to_numeric(combined_data['Score'], errors='coerce')


elite_data = combined_data[combined_data['Expertise'] == 'Elite']
subelite_data = combined_data[combined_data['Expertise'] == 'SubElite']

levels = ['41', '42', '43']

positions_elite = [1, 5, 9]
positions_subelite = [2, 6, 10]

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

# Ajouter des annotations et lignes pour le test de Mann-Whitney U
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
ax.set_ylim(40, 95)
ax.set_xticks([1.5, 5.5, 9.5])
ax.set_xticklabels(['41', '42', '43'])
ax.set_title('Comparison of % Time to reach 3/4 twist by Expertise')
ax.set_xlabel('Difficulty Level')
ax.set_ylabel('Time %')

plt.show()
