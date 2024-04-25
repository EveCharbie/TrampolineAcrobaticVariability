import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# List of data files
files = [
    '/home/lim/Documents/StageMathieu/results_41_rotation.csv',
    '/home/lim/Documents/StageMathieu/results_42_rotation.csv',
    '/home/lim/Documents/StageMathieu/results_43_rotation.csv',
    '/home/lim/Documents/StageMathieu/results_41o_rotation.csv',
    '/home/lim/Documents/StageMathieu/results_811<_rotation.csv',
    '/home/lim/Documents/StageMathieu/results_822_rotation.csv',
    '/home/lim/Documents/StageMathieu/results_831<_rotation.csv'
]

# DataFrame to store aggregated data for plotting
all_data = pd.DataFrame()

# Process each file
for file in files:
    data = pd.read_csv(file)
    data_specific = data[data['Timing'].isin(['75%', 'Landing', 'Takeoff'])]
    data_specific['Source'] = file.split('/')[-1].replace('results_', '').replace('_rotation.csv', '')  # Clean file ID

    # ANOVA without considering 'Expertise'
    model = ols('Std ~ C(Timing)', data=data_specific).fit()
    anova_results = anova_lm(model, typ=2)
    print(f"ANOVA Results for {data_specific['Source'].iloc[0]}:\n", anova_results)

    # Tukey HSD Test
    tukey = pairwise_tukeyhsd(endog=data_specific['Std'], groups=data_specific['Timing'], alpha=0.05)
    print(f"Tukey HSD Results for {data_specific['Source'].iloc[0]}:\n", tukey)

    # Append data to the plotting DataFrame
    all_data = pd.concat([all_data, data_specific], ignore_index=True)

# Prepare data for plotting
all_data['Timing'] = pd.Categorical(all_data['Timing'], categories=["Takeoff", "75%", "Landing"], ordered=True)

# Create the plot
plt.figure(figsize=(12, 8))
plot = sns.pointplot(x='Timing', y='Std', hue='Source', data=all_data, dodge=True,
                     capsize=0.1, err_kws={'linewidth': 0.5}, palette='deep', errorbar='sd')


# Adding plot details
plt.title('Standard Deviation Across Different Timings from Multiple Files')
plt.xlabel('Timing')
plt.ylabel('Standard Deviation')
plt.legend(title='File ID', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

# Display the plot
plt.show()