import os
import glob
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from Function_Class_Basics import OrderMatData, load_and_interpolate, calculate_mean_std

# Path of the .mat file
file_path_mat = "/home/lim/Documents/StageMathieu/Data_propre/SaMi/Q/"

# Output path for the graph
folder_path = "/Graph_from_mot/SaMi/MeanSD"

# Create folder path
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Interval and name of different path
file_intervals = [
    (file_path_mat + "Sa_821_822_2_MOD200.00_GenderF_SaMig_Q.mat", (3299, 3591)),
    (file_path_mat + "Sa_821_822_3_MOD200.00_GenderF_SaMig_Q.mat", (3139, 3440)),
    # Other file
]

# Load and interpolate all try
my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]

# List of members
members = [
    "Pelvis",
    "Thorax",
    "Tete",
    "EpauleD",
    "BrasD",
    "AvBrasD",
    "MainD",
    "EpauleG",
    "BrasG",
    "AvBrasG",
    "MainG",
    "CuisseD",
    "JambeD",
    "PiedD",
    "CuisseG",
    "JambeG",
    "PiedG",
]

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
            plt.fill_between(
                range(len(mean_data)), mean_data - std_dev_data, mean_data + std_dev_data, color="gray", alpha=0.5
            )
            plt.title(f'Mean of data {member} {["X", "Y", "Z"][axis]} with standard deviation')
            plt.xlabel("Time (%)")
            plt.ylabel("Value")
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
colors = ["red", "green", "blue"]
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
            plt.fill_between(
                range(len(mean_data)),
                mean_data - std_dev_data,
                mean_data + std_dev_data,
                alpha=0.4,
                color=color,
                edgecolor="none",
            )

        except KeyError:
            # Handle the case where a combination member axis doesn't exist
            print(f"Le membre {member} avec l'axe {['X', 'Y', 'Z'][axis]} n'existe pas.")

    # Configure the graph
    plt.xlabel("Time (%)", fontsize=4)
    plt.ylabel("Value", fontsize=4)
    plt.tick_params(axis="both", labelsize=3, width=0.3, length=1.5)

    # Make the axis line thinner
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.3)

    # plt.legend()
    # plt.tick_params(axis='x', labelsize=3)  # Change 'labelsize'

    # Register the graph
    file_name = f"{member}_all_axes_graph.png"
    file_path = os.path.join(folder_path, file_name)
    plt.savefig(file_path, format="png", bbox_inches="tight")
    # Create and register legend for only one member (Tete)
    if member == "Tete" and not legend_created:
        # Get the handles and the labels for the legend
        handles, labels = ax.get_legend_handles_labels()

        # Create a new figure for the legend with the desired size
        fig_leg = plt.figure(figsize=(2, 2), dpi=100)  # Dimensions en pouces, dpi pour la résolution
        ax_leg = fig_leg.add_subplot(111)

        # Add a legend to the new figure with a bigger font size and bold
        legend = ax_leg.legend(
            handles, labels, loc="center", prop={"size": 20, "weight": "bold"}
        )  # Adjust size of font

        # Increase the width of color line in the legend
        legend.get_lines()[0].set_linewidth(4)  # For "X"
        legend.get_lines()[1].set_linewidth(4)  # For "Y"
        legend.get_lines()[2].set_linewidth(4)  # For "Z"

        ax_leg.axis("off")

        # Register the picture of the legend with the desired size
        leg_file_name = "legend.png"
        leg_file_path = os.path.join(folder_path, leg_file_name)
        fig_leg.savefig(leg_file_path, format="png", bbox_inches="tight", pad_inches=0)  # Delete pad
        plt.close(fig_leg)

        legend_created = True

    plt.close("all")
