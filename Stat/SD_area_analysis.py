import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score

# ratio_twist_somersault = {
#     '4-': 0,
#     '4-o': 0,
#     '8--o': 0,
#     '8-1<': 0.25,
#     '8-1o': 0.25,
#     '41': 0.5,
#     '811<': 0.5,
#     '41o': 0.5,
#     '8-3<': 0.75,
#     '42': 1,
#     '822': 1,
#     '831<': 1,
#     '43': 1.5,
#
# }

velocity_at_T75 = {
    '8-1o': 109,
    '8-1<': 119,
    '41': 88,
    '811<': 127,
    '41o': 82,
    '8-3<': 204,
    '42': 128,
    '822': 181,
    '831<': 320,
    '43': 183,
}


ratio_twist_somersault = {
    '8-1<': (119, 'pike'),
    '8-1o': (109, 'grouped'),
    '41': (88, 'stretched'),
    '811<': (127, 'pike'),
    '41o': (82, 'stretched'),
    '8-3<': (204, 'pike'),
    '42': (128, 'stretched'),
    '822': (181, 'stretched'),
    '831<': (320, 'pike'),
    '43': (183, 'stretched'),
}

movement_to_analyse = list(ratio_twist_somersault.keys())

# Load your data
data = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result3/results_area_under_curve2.csv')

# Melt the data to long format
combined_data = pd.melt(data, id_vars=['ID', 'Expertise'], value_vars=movement_to_analyse, var_name='Difficulty', value_name='Score')
combined_data = combined_data.dropna(subset=['Score'])
# Map the ratios to the 'Difficulty' column in combined_data
combined_data['TwistRatio'] = combined_data['Difficulty'].map(lambda x: ratio_twist_somersault[x][0])
combined_data['Type'] = combined_data['Difficulty'].map(lambda x: ratio_twist_somersault[x][1])

# Compute linear regression
slope, intercept, r_value, p_value, std_err = linregress(combined_data['TwistRatio'], combined_data['Score'])

# Create the regression line data
x_reg_line = np.linspace(min(combined_data['TwistRatio']), max(combined_data['TwistRatio']), 100)
y_reg_line = slope * x_reg_line + intercept

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))
markers = {'grouped': 'o', 'stretched': '*', 'pike': '<'}


# Plot each point individually to mimic 'hue' in seaborn
for (difficulty, type), group_data in combined_data.groupby(['Difficulty', 'Type']):
    ax.scatter(group_data['TwistRatio'], group_data['Score'], label=f"{difficulty}", marker=markers[type])

# Plot regression line
p_text = "p < 0.001" if p_value < 0.001 else f"p = {p_value:.3f}"
text_str = f'R-squared: {r_value**2:.2f}\n{p_text}'
ax.plot(x_reg_line, y_reg_line, 'r-', label=text_str)

# Set titles and labels
ax.set_title('Scatter Plot Grouped by Twist to Somersault Ratio with Regression Line')
ax.set_xlabel('Twist to Somersault Ratio')
ax.set_ylabel('Score')

# Add a legend
ax.legend(title='Difficulty', bbox_to_anchor=(1.05, 1), loc='upper left')

# Tight layout to adjust for the plot
plt.tight_layout()

plt.show()


correlation = data[movement_to_analyse].corr()
print(correlation)
