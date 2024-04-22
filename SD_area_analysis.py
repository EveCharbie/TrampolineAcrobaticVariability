import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/home/lim/Documents/StageMathieu/results_area_under_curve.csv')

# Setting up Seaborn
sns.set(style="whitegrid")

# 1. Boxplots for the different difficulty levels without expertise distinction
plt.figure(figsize=(10, 6))
sns.boxplot(data=data.loc[:, '41':'43'])
plt.title('Boxplot of difficulty levels without expertise distinction')
plt.show()

# 2. Boxplots for each difficulty level by expertise group
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle('Boxplot of difficulty levels by expertise group')

sns.boxplot(ax=axes[0], x='Expertise', y='41', data=data)
axes[0].set_title('Level 41')

sns.boxplot(ax=axes[1], x='Expertise', y='42', data=data)
axes[1].set_title('Level 42')

sns.boxplot(ax=axes[2], x='Expertise', y='43', data=data)
axes[2].set_title('Level 43')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 3. Analysis of the correlation between obtained values and increasing difficulty
diff_41_42 = data['42'] - data['41']
diff_42_43 = data['43'] - data['42']

mean_diff_41_42 = diff_41_42.mean()
mean_diff_42_43 = diff_42_43.mean()

print(f'Mean score difference between levels 41 and 42: {mean_diff_41_42:.2f}')
print(f'Mean score difference between levels 42 and 43: {mean_diff_42_43:.2f}')
