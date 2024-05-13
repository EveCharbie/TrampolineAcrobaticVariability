import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene
import os
from scipy.stats import kruskal
import scikit_posthocs as sp
from TrampolineAcrobaticVariability.Function.Function_stat import (perform_anova_and_tukey,
                                                                   perform_kruskal_and_dunn,
                                                                   prepare_data)

home_path = "/home/lim/Documents/StageMathieu/Tab_result/"

rotation_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'rotation' in file:
            full_path = os.path.join(root, file)
            rotation_files.append(full_path)

position_files = []

for root, dirs, files in os.walk(home_path):
    for file in files:
        if 'position' in file:
            full_path = os.path.join(root, file)
            position_files.append(full_path)


for file_rotation in rotation_files:

    file_position = file_rotation.replace('_rotation.csv', '_position.csv')

    data_rotation = pd.read_csv(file_rotation)
    data_rotation = data_rotation[data_rotation['Timing'].isin(['75%', 'Landing'])]
    data_rotation = data_rotation.pivot(index='ID', columns='Timing', values='Std')
    data_rotation['Delta'] = data_rotation['75%'] - data_rotation['Landing']
    data_rotation.reset_index(inplace=True)

    data_position = pd.read_csv(file_position)
    data_position = prepare_data(data_position)
    data_position = data_position[data_position['Timing'].isin(['75%', 'Landing'])]
    data_position = data_position.pivot(index='ID', columns='Timing', values=['upper_body', 'lower_body'])
    data_position['Delta_upper_body'] = data_position['upper_body']['75%'] - data_position['upper_body']['Landing']
    data_position['Delta_lower_body'] = data_position['lower_body']['75%'] - data_position['lower_body']['Landing']
    data_position = data_position.drop(columns=['upper_body', 'lower_body'], level=0)

    data_position.reset_index(inplace=True)
    data_position.reset_index(drop=True, inplace=True)

    combined_data = pd.merge(data_rotation[['ID', 'Delta']],
                             data_position[['ID', 'Delta_upper_body', 'Delta_lower_body']], on='ID', how='inner')

    # Calculer la corrélation de Pearson
    correlation_results = combined_data.corr()

    print(correlation_results)

