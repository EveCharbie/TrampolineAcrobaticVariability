import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Class to order the .mat data in order and easy access
class OrderMatData:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Mapping des indices aux suffixes attendus
        self.index_suffix_map = {
            0: 'X', 1: 'Y', 2: 'Z'
        }

    def __getitem__(self, key):
        matching_columns = [col for col in self.dataframe.columns if col.startswith(key)]
        if not matching_columns:
            raise KeyError(f"Variable {key} not found.")
        return self.dataframe[matching_columns]

    def get_column_by_index(self, key, index):
        # Vérifie si l'index est valide
        if index not in self.index_suffix_map:
            raise KeyError(f"Invalid index {index}.")

        expected_suffix = self.index_suffix_map[index]
        column_name = f"{key}_{expected_suffix}"

        if column_name not in self.dataframe.columns:
            raise KeyError(f"Column {column_name} does not exist.")

        return self.dataframe[column_name]


# Path of the .mat file
file_path_mat = '/home/lim/Documents/StageMathieu/Data_propre/SaMi/Q/'

# Output path for the graph
folder_path = "/home/lim/Documents/StageMathieu/Graph_from_mot/SaMi/MeanSD"

# Create folder path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Interval and name of different path
file_intervals = [
    (file_path_mat + 'Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat', (3299, 3591)),
    (file_path_mat + 'Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat', (3139, 3440)),
    # Other file
]


# Define function to load and interpolate data in 0-100% from a .mat and apply the class MyData
def load_and_interpolate(file, interval, num_points=100):
    # Load data with the DoF
    data = scipy.io.loadmat(file)
    df = pd.DataFrame(data['Q2']).T
    column_names = [
        "PelvisTranslation_X", "PelvisTranslation_Y", "PelvisTranslation_Z",
        "Pelvis_X", "Pelvis_Y", "Pelvis_Z",
        "Thorax_X", "Thorax_Y", "Thorax_Z",
        "Tete_X", "Tete_Y", "Tete_Z",
        "EpauleD_Y", "EpauleD_Z",
        "BrasD_X", "BrasD_Y", "BrasD_Z",
        "AvBrasD_X", "AvBrasD_Z",
        "MainD_X", "MainD_Y",
        "EpauleG_Y", "EpauleG_Z",
        "BrasG_X", "BrasG_Y", "BrasG_Z",
        "AvBrasG_X", "AvBrasG_Z",
        "MainG_X", "MainG_Y",
        "CuisseD_X", "CuisseD_Y", "CuisseD_Z",
        "JambeD_X",
        "PiedD_X", "PiedD_Z",
        "CuisseG_X", "CuisseG_Y", "CuisseG_Z",
        "JambeG_X",
        "PiedG_X", "PiedG_Z"
    ]
    df.columns = column_names

    # Select data in specify interval
    df_selected = df.iloc[interval[0]:interval[1]]

    # Interpolate each column to have a uniform number of points
    df_interpolated = df_selected.apply(lambda x: np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(x)), x))

    # Create OrderMatData instance and apply it to df_interpolated
    my_data_instance = OrderMatData(df_interpolated)
    return my_data_instance


def calculate_mean_std(data_instances, member, axis):
    """
    Calculates the mean and std for a given member and an axes on all data instances
    """
    data_arrays = [instance.get_column_by_index(member, axis) for instance in data_instances]
    Mean_Data = np.mean(data_arrays, axis=0)
    Std_Dev_Data = np.std(data_arrays, axis=0)
    return Mean_Data, Std_Dev_Data


# Load and interpolate all try
my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]


# Example for accessing data of Pelvis_X using index with OrderMatData class
# data_first_file = my_data_instances[0]  # First instance of OrderMatData
# pelvis_x_data = data_first_file.get_column_by_index("Pelvis", 0)


# List of members
members = ["Pelvis", "Thorax", "Tete", "EpauleD", "BrasD", "AvBrasD", "MainD", "EpauleG", "BrasG", "AvBrasG", "MainG",
           "CuisseD", "JambeD", "PiedD", "CuisseG", "JambeG", "PiedG"]

####### ONE COMPONENT BY GRAPH #######
axes = [0, 1, 2]  # 0 for X, 1 for Y, 2 for Z

for member in members:
    for axis in axes:
        try:
            # Calculate mean and std for all member and axis
            mean_data, std_dev_data = calculate_mean_std(my_data_instances, member, axis)

            # Trace graph of the mean with std area
            plt.figure(figsize=(10, 6))
            plt.plot(mean_data, label=f'{member} {["X", "Y", "Z"][axis]}')
            plt.fill_between(range(len(mean_data)), mean_data - std_dev_data, mean_data +
                             std_dev_data, color='gray', alpha=0.5)
            plt.title(f'Mean of data {member} {["X", "Y", "Z"][axis]} with standard deviation')
            plt.xlabel('Time (%)')
            plt.ylabel('Value')
            plt.legend()
            # plt.show()
            # Name of graph file
            file_name = f"{member}_{['X', 'Y', 'Z'][axis]}_graph.png"
            file_path = os.path.join(folder_path, file_name)

            # Register the graph in specific path
            plt.savefig(file_path)
            plt.close()

        except KeyError:
            # Handle the case where a member-axis combination does not exist
            print(f"The member {member} with axis {['X', 'Y', 'Z'][axis]} does not exist.")


####### ALL COMPONENT BY GRAPH  #######

# Define the colors for the axis X, Y, et Z
colors = ['red', 'green', 'blue']
dpi = 300  # High resolution
desired_width_px = 333  # Desired width in pixels
desired_height_px = 200  # Desired height in pixels
fig_width = desired_width_px / dpi  # Width in inches
fig_height = desired_height_px / dpi  # Height in inches
legend_created = False  # Verify that the legend wasn't created

for member in members:
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    for axis in axes:
        try:
            # Calculate mean and std for all member and axis
            mean_data, std_dev_data = calculate_mean_std(my_data_instances, member, axis)

            # Get matching color for current axis
            color = colors[axis]

            # Trace the graph of the mean with std area for all axis
            plt.plot(mean_data, label=f'{["X", "Y", "Z"][axis]}', color=color, linewidth=0.3)
            plt.fill_between(range(len(mean_data)), mean_data - std_dev_data, mean_data + std_dev_data, alpha=0.4,
                             color=color, edgecolor='none')

        except KeyError:
            # Handle the case where a combination member axis doesn't exist
            print(f"Le membre {member} avec l'axe {['X', 'Y', 'Z'][axis]} n'existe pas.")

    # Configure the graph
    plt.xlabel('Time (%)', fontsize=4)
    plt.ylabel('Value', fontsize=4)
    plt.tick_params(axis='both', labelsize=3, width=0.3, length=1.5)

    # Make the axis line thinner
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.3)

    # plt.legend()
    # plt.tick_params(axis='x', labelsize=3)  # Change 'labelsize'

    # Register the graph
    file_name = f"{member}_all_axes_graph.png"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format='png', bbox_inches='tight')
    # Create and register legend for only one member (Tete)
    if member == 'Tete' and not legend_created:
        # Get the handles and the labels for the legend
        handles, labels = ax.get_legend_handles_labels()

        # Create a new figure for the legend with the desired size
        fig_leg = plt.figure(figsize=(2, 2), dpi=100)  # Dimensions en pouces, dpi pour la résolution
        ax_leg = fig_leg.add_subplot(111)

        # Add a legend to the new figure with a bigger font size and bold
        legend = ax_leg.legend(handles, labels, loc='center',
                               prop={'size': 20, 'weight': 'bold'})  # Adjust size of font

        # Increase the width of color line in the legend
        legend.get_lines()[0].set_linewidth(4)  # For "X"
        legend.get_lines()[1].set_linewidth(4)  # For "Y"
        legend.get_lines()[2].set_linewidth(4)  # For "Z"

        ax_leg.axis('off')

        # Register the picture of the legend with the desired size
        leg_file_name = "legend.png"
        leg_file_path = os.path.join(folder_path, leg_file_name)
        fig_leg.savefig(leg_file_path, format='png', bbox_inches='tight',
                        pad_inches=0)  # Delete pad
        plt.close(fig_leg)

        legend_created = True

    plt.close('all')