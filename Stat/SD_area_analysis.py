import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, mannwhitneyu
import numpy as np
import matplotlib.patches as mpatches
from sklearn.metrics import r2_score

ratio_twist_somersault = {
    '4-': 0,
    '4-o': 0,
    '8--o': 0,
    '8-1<': 0.25,
    '8-1o': 0.25,
    '41': 0.5,
    '811<': 0.5,
    '41o': 0.5,
    '8-3<': 0.75,
    '42': 1,
    '822': 1,
    '831<': 1,
    '43': 1.5,

}

movement_to_analyse = list(ratio_twist_somersault.keys())

# Load your data
data = pd.read_csv('/home/lim/Documents/StageMathieu/Tab_result/results_area_under_curve2.csv')

# Melt the data to long format
combined_data = pd.melt(data, id_vars=['ID', 'Expertise'], value_vars=movement_to_analyse, var_name='Difficulty', value_name='Score')
combined_data = combined_data.dropna(subset=['Score'])
# Map the ratios to the 'Difficulty' column in combined_data
combined_data['TwistRatio'] = combined_data['Difficulty'].map(ratio_twist_somersault)

# Compute linear regression
slope, intercept, r_value, p_value, std_err = linregress(combined_data['TwistRatio'], combined_data['Score'])

# Compute the predicted values using the regression line equation
predicted_values = slope * combined_data['TwistRatio'] + intercept

# Calculate R-squared value
r_squared = r2_score(combined_data['Score'], predicted_values)

# Create the regression line data
x_reg_line = np.linspace(min(combined_data['TwistRatio']), max(combined_data['TwistRatio']), 100)
y_reg_line = slope * x_reg_line + intercept

# Create the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot each point individually to mimic 'hue' in seaborn
for difficulty, group_data in combined_data.groupby('Difficulty'):
    ax.scatter(group_data['TwistRatio'], group_data['Score'], label=difficulty)

# Plot regression line
ax.plot(x_reg_line, y_reg_line, 'r-', label=f'R2 {r_squared:.2f}')

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
