import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
import numpy as np


ratio_twist_somersault = {
    '8-1<': (0.25, 'pike'),
    '8-1o': (0.25, 'grouped'),
    '41': (0.5, 'stretched'),
    '811<': (0.5, 'pike'),
    '41o': (0.5, 'stretched'),
    '8-3<': (0.75, 'pike'),
    '42': (1, 'stretched'),
    '822': (1, 'stretched'),
    '831<': (1, 'pike'),
    '43': (1.5, 'stretched'),
}

## velocity
# ratio_twist_somersault = {
#     '8-1<': (119, 'pike'),
#     '8-1o': (109, 'grouped'),
#     '41': (88, 'stretched'),
#     '811<': (127, 'pike'),
#     '41o': (82, 'stretched'),
#     '8-3<': (204, 'pike'),
#     '42': (128, 'stretched'),
#     '822': (181, 'stretched'),
#     '831<': (320, 'pike'),
#     '43': (183, 'stretched'),
# }

movement_to_analyse = list(ratio_twist_somersault.keys())

data = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_area_under_curve2.csv')

combined_data = pd.melt(data, id_vars=['ID', 'Expertise'], value_vars=movement_to_analyse, var_name='Difficulty', value_name='Score')
combined_data = combined_data.dropna(subset=['Score'])

combined_data['TwistRatio'] = combined_data['Difficulty'].map(lambda x: ratio_twist_somersault[x][0])
combined_data['Type'] = combined_data['Difficulty'].map(lambda x: ratio_twist_somersault[x][1])

slope, intercept, r_value, p_value, std_err = linregress(combined_data['TwistRatio'], combined_data['Score'])

x_reg_line = np.linspace(min(combined_data['TwistRatio']), max(combined_data['TwistRatio']), 100)
y_reg_line = slope * x_reg_line + intercept

fig, ax = plt.subplots(figsize=(10, 6))
markers = {'grouped': 'o', 'stretched': '*', 'pike': '<'}

for (difficulty, type), group_data in combined_data.groupby(['Difficulty', 'Type']):
    ax.scatter(group_data['TwistRatio'], group_data['Score'], label=f"{difficulty}", marker=markers[type])

p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'R-squared: {r_value**2:.2f}\n{p_text}'
ax.plot(x_reg_line, y_reg_line, 'r-', label=text_str)

ax.set_title('Scatter Plot Grouped by Twist to Somersault Ratio with Regression Line')
ax.set_xlabel('Twist to Somersault Ratio')
ax.set_ylabel('Score')

ax.legend(title='Difficulty', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()

plt.show()

