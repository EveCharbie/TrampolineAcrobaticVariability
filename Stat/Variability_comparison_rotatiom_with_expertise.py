import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import levene, mannwhitneyu
import matplotlib.patches as mpatches
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.multivariate.manova import MANOVA


data_41 = pd.read_csv('/results_41_rotation.csv')
data_42 = pd.read_csv('/results_42_rotation.csv')
data_43 = pd.read_csv('/results_43_rotation.csv')

anova_rot_df = data_41.pivot_table(index=['ID', 'Expertise'], columns='Timing', values='Std')
anova_rot_df.reset_index(inplace=True)

# Renommer les colonnes pour refléter les nouveaux noms
anova_rot_df.columns = ['SubjectID', 'Expertise', 'Std_75%', 'Std_Takeoff', 'Std_Landing']

