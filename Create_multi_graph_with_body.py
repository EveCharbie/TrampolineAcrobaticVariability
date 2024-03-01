import pickle
import ezc3d
import os
import scipy.io
import biorbd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Function_Class_Basics import (
    OrderMatData,
    load_and_interpolate,
    calculate_mean_std,
    column_names,
    create_composite_image,
    add_lines_with_arrow_and_circle,
    graph_images_info,
    lines_info,
)

# Chemin du dossier contenant les fichiers .mat
file_path_mat = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Q/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Graphique/821_seul/"

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(folder_path):
    os.makedirs(folder_path)


# Liste des tuples (chemin du fichier, intervalle)
file_intervals = [
    (file_path_mat + "Sa_821_seul_1.mat", (0, 308)),
    (file_path_mat + "Sa_821_seul_2.mat", (0, 305)),
    (file_path_mat + "Sa_821_seul_3.mat", (0, 307)),
    (file_path_mat + "Sa_821_seul_4.mat", (0, 309)),
    (file_path_mat + "Sa_821_seul_5.mat", (0, 304)),
]

for file_path, interval in file_intervals:
    # Extraire le nom de base du fichier pour le nom du dossier
    base_name = os.path.basename(file_path).split(".")[0]

    # Créer un sous-dossier pour cet essai
    trial_folder_path = os.path.join(folder_path, base_name)
    if not os.path.exists(trial_folder_path):
        os.makedirs(trial_folder_path)

    # Charger les données depuis le fichier .mat
    data_loaded = scipy.io.loadmat(file_path)
    q2_data = data_loaded["Q2"]
    DataFrame_with_colname = pd.DataFrame(q2_data).T
    DataFrame_with_colname.columns = column_names
    my_data = OrderMatData(DataFrame_with_colname)

    selected_data = my_data.dataframe.iloc[interval[0] : interval[1]]

    member_groups = set([name.split("_")[0] for name in column_names])

    for group in member_groups:
        group_columns = [col for col in column_names if col.startswith(group)]
        group_data = selected_data[group_columns]

        plt.figure(figsize=(10, 6))
        for col in group_columns:
            plt.plot(group_data[col], label=col)

        plt.title(f"Graphique pour {group}")
        plt.xlabel("Index")
        plt.ylabel("Valeur")
        plt.legend()

        # Enregistrer le graphique dans le sous-dossier de l'essai
        file_name = f"{group}_graph.png"
        file_path_graph = os.path.join(trial_folder_path, file_name)

        plt.savefig(file_path_graph)
        plt.close()


# Chemin du nouveau dossier "mean"
mean_folder_path = os.path.join(folder_path, "MeanSD/")

# Créer le dossier "mean" s'il n'existe pas déjà
if not os.path.exists(mean_folder_path):
    os.makedirs(mean_folder_path)

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
            file_path = os.path.join(mean_folder_path, file_name)
            # Register the graph in specific path
            plt.savefig(file_path)
            plt.close()

        except KeyError:
            # Handle the case where a member-axis combination does not exist
            print(f"The member {member} with axis {['X', 'Y', 'Z'][axis]} doesn't exist.")


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
            print(f"The member {member} with axis {['X', 'Y', 'Z'][axis]} doesn't exist.")

    # Configure the graph
    plt.xlabel("Time (%)", fontsize=4)
    # plt.ylabel('Value', fontsize=4)
    plt.tick_params(axis="both", labelsize=3, width=0.3, length=1.5)

    # Make the axis line thinner
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.3)

    # plt.legend()
    # plt.tick_params(axis='x', labelsize=3)  # Change 'labelsize'

    # Register the graph
    file_name = f"{member}_all_axes_graph.png"
    file_path = os.path.join(mean_folder_path, file_name)
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
        leg_file_path = os.path.join(mean_folder_path, leg_file_name)
        fig_leg.savefig(leg_file_path, format="png", bbox_inches="tight", pad_inches=0)  # Delete pad
        plt.close(fig_leg)

        legend_created = True

    plt.close("all")


final_path = folder_path + "Graph_with_body.png"

# Call the function to create the composite image
composite_image_path = create_composite_image(graph_images_info, mean_folder_path, save_path=final_path)
# Call the function
output_image_path = add_lines_with_arrow_and_circle(composite_image_path, lines_info)
