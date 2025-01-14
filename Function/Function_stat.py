import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import rankdata
from scipy.stats import norm
import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np


def prepare_data(data):
    # Calcul des moyennes pour le haut et le bas du corps
    upper_body_columns = data[['AvBrasD', 'MainD', 'AvBrasG', 'MainG']]
    data["upper_body"] = upper_body_columns.mean(axis=1)
    lower_body_columns = data[['JambeD', 'PiedD', 'JambeG', 'PiedG']]
    data["lower_body"] = lower_body_columns.mean(axis=1)

    # Création des groupes basés uniquement sur le Timing
    conditions = [
        (data['Timing'] == 'Takeoff'),
        (data['Timing'] == '75%'),
        (data['Timing'] == 'Landing')
    ]
    labels = ['Takeoff', '75%', 'Landing']

    data['Timing'] = np.select(conditions, labels, default='Other')

    data_subset = data[data['Timing'] != 'Other']
    return data_subset[['ID', 'upper_body', 'lower_body', 'Timing']]


def perform_anova_and_tukey(data, dependent_var, group_var):
    # Fit the ANOVA model
    model = ols(f'{dependent_var} ~ C({group_var})', data=data).fit()
    anova_results = anova_lm(model, typ=2)
    print(f"ANOVA Results:\n{anova_results}\n")

    # Perform Tukey HSD post-hoc test if ANOVA is significant
    if anova_results.loc['C(Timing)', 'PR(>F)'] < 0.05:
        tukey_results = pairwise_tukeyhsd(endog=data[dependent_var], groups=data[group_var], alpha=0.05)
        print(f"Tukey HSD Results:\n{tukey_results}")
    else:
        print("No significant differences found by ANOVA; no post hoc test performed.")


def dunn_test(data, group_col, value_col, p_adjust=None):
    """
    Perform Dunn's test for multiple comparisons.

    Parameters:
    - data: pandas DataFrame with the data
    - group_col: column name for group labels
    - value_col: column name for values

    Returns:
    - A DataFrame with pairwise comparisons, z-scores, and p-values
    """
    # Extract groups
    groups = data[group_col].unique()
    group_data = [data[data[group_col] == group][value_col].values for group in groups]

    # Flatten data and assign ranks
    all_data = np.concatenate(group_data)
    ranks = rankdata(all_data)

    # Calculate rank sums and group sizes
    rank_sums = []
    idx_init = 0
    for group in group_data:
        rank_sums += [ranks[idx_init : idx_init+group.shape[0]].sum()]
        idx_init += group.shape[0]
    n = [len(group) for group in group_data]
    N = sum(n)

    # Calculate variance correction factor for ties
    _, ties = np.unique(ranks, return_counts=True)
    tie_correction = 1 - sum((ties ** 3 - ties) / (N ** 3 - N))

    if p_adjust == 'bonferroni':
        p_value_mult = len(rank_sums)
    elif p_adjust == None:
        p_value_mult = 1
    else:
        raise NotImplementedError(f"The p_adjust method {p_adjust} is not implemented yet, please try bonferroni")

    # Calculate z-scores for pairwise comparisons
    results = []
    for i, (r1, n1) in enumerate(zip(rank_sums, n)):
        for j, (r2, n2) in enumerate(zip(rank_sums, n)):
            if i < j:
                mean_diff = r1 / n1 - r2 / n2
                std_dev = np.sqrt(tie_correction * N * (N + 1) / 12 * (1 / n1 + 1 / n2))
                z_score = mean_diff / std_dev
                p_value = 2 * (1 - norm.cdf(abs(z_score))) * p_value_mult  # Two-tailed p-value
                results.append((groups[i], groups[j], z_score, p_value))

    # Create results DataFrame
    results_df = pd.DataFrame(results, columns=["Group1", "Group2", "Z-Score", "P-Value"])
    return results_df



def perform_kruskal_and_dunn(data, dependent_var, group_var):
    # Group data for Kruskal-Wallis test
    groups = [group[dependent_var].values for name, group in data.groupby(group_var)]
    kruskal_stat, kruskal_p = stats.kruskal(*groups)
    print(f"Kruskal-Wallis Test Results (P-value: {kruskal_p:.4f})")

    # Perform Dunn's post-hoc test if Kruskal-Wallis test is significant
    if kruskal_p < 0.05:
        # I verified that the p-values are the same for my implementation and sp's one :)
        # posthoc_results = sp.posthoc_dunn(data, val_col=dependent_var, group_col=group_var, p_adjust='bonferroni')
        dunn_results = dunn_test(data, value_col=dependent_var, group_col=group_var, p_adjust='bonferroni')
        print("Post-hoc Dunn's Test Results:")
        print(dunn_results)
        groups = data[group_var].unique()
        group_data = [data[data[group_var] == group][dependent_var].values for group in groups]
        print("Takeoff : ", np.mean(group_data[0]))
        print("T75 : ", np.mean(group_data[1]))
        print("Landing : ", np.mean(group_data[2]))
        return dunn_results
    else:
        print("No significant differences found by Kruskal-Wallis; no post hoc test performed.")
        fake_data = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        fake_data_df = pd.DataFrame(fake_data, columns=["Group1", "Group2", "Z-Score", "P-Value"])
        return fake_data_df


def safe_interpolate(x, num_points):
    # Create an array to store interpolated values, initialized with NaN
    interpolated_values = np.full(num_points, np.nan)

    # Check if all elements are finite, ignore NaNs for interpolation
    finite_mask = np.isfinite(x)
    if finite_mask.any():  # Ensure there's at least one finite value to interpolate
        # Interpolate only finite values
        valid_x = x[finite_mask]
        valid_indices = np.linspace(0, 1, len(x))[finite_mask]

        # Perform interpolation over the range with finite values
        interpolated_valid_values = np.interp(np.linspace(0, 1, num_points), valid_indices, valid_x)

        # Round interpolated values to the nearest integer
        rounded_values = np.round(interpolated_valid_values).astype(int)

        # Place rounded interpolated values back in the full array
        interpolated_values = rounded_values

    return interpolated_values