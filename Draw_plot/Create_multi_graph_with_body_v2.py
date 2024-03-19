import pickle
import ezc3d
import os
import scipy.io
import biorbd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    OrderMatData,
    load_and_interpolate,
    calculate_mean_std
)
from TrampolineAcrobaticVariability.Function.Function_draw import (
    create_composite_image,
    add_lines_with_arrow_and_circle,
    graph_images_info,
    lines_info,
)

home_path = "/home/lim/Documents/StageMathieu/DataTrampo/"

csv_path = f"{home_path}Labelling_trampo.csv"
interval_name_tab = pd.read_csv(csv_path, sep=';', usecols=['Participant', 'Analyse', 'Essai', 'Durée'])
valide = ['O']
interval_name_tab = interval_name_tab[interval_name_tab["Analyse"] == 'O']

interval_name_tab['Essai'] = interval_name_tab['Essai'] + '.c3d'

# Obtenir la liste des participants
participant_names = interval_name_tab['Participant'].unique()

for participant in participant_names:
    # Chemin du dossier contenant les fichiers .mat
    file_path_participant = home_path + f"{participant}/"
    file_path_participant_mat = home_path + f"{participant}/Q/"

    directory_names = []
    for entry in os.listdir(file_path_participant_mat):
        entry_path = os.path.join(file_path_participant_mat, entry)
        if os.path.isdir(entry_path):
            directory_names.append(entry)

    for acrobatie in directory_names:
        file_path_participant_mat = home_path + f"{participant}/Q/{acrobatie}/"
        file_names = []
        for entry in os.listdir(file_path_participant_mat):
            entry_path = os.path.join(file_path_participant_mat, entry)
            if os.path.isfile(entry_path):
                file_names.append(entry)

        # Strip the .mat extension and append .c3d
        filenames_c3d = [f[:-4] + '.c3d' for f in file_names]
        matching_rows = interval_name_tab[interval_name_tab['Essai'].isin(filenames_c3d)]

        folder_path_participant = f"{home_path}{participant}/Graphique/"

        # essai_by_name = interval_name_tab[interval_name_tab["Participant"] == participant].copy()

        file_intervals = []

        for index, row in matching_rows.iterrows():
            filename = row['Essai']
            if filename.endswith('.c3d'):
                filename = filename[:-4] + '.mat'
            else:
                filename = filename + '.mat'

            file_path = file_path_participant_mat + filename

            # Get the duration and construct the interval (0, Durée)
            duration = row['Durée']
            interval = (0, int(duration))
            file_intervals.append((file_path, interval))

        # Créer le dossier de sortie s'il n'existe pas
        if not os.path.exists(folder_path_participant):
            os.makedirs(folder_path_participant)

    for file_path, interval in file_intervals:
        # Extraire le nom de base du fichier pour le nom du dossier
        base_name = os.path.basename(file_path).split(".")[0]

        # Créer un sous-dossier pour cet essai
        trial_folder_path = os.path.join(folder_path_participant, base_name)
        if not os.path.exists(trial_folder_path):
            os.makedirs(trial_folder_path)

        # Charger les données depuis le fichier .mat
        data_loaded = scipy.io.loadmat(file_path)
        q2_data = data_loaded["Q_ready_to_use"]
        Euler_Sequence = data_loaded["Euler_Sequence"]

        column_names = []
        for segment, sequence in Euler_Sequence:
            segment = segment.strip()
            for axis in sequence.strip():
                column_names.append(f"{segment}_{axis.upper()}")

        DataFrame_with_colname = pd.DataFrame(q2_data).T
        DataFrame_with_colname.columns = column_names
        my_data = OrderMatData(DataFrame_with_colname)

        selected_data = my_data.dataframe.iloc[interval[0] : interval[1]]

        member_groups = set([name.split("_")[0] for name in column_names])

        # for group in member_groups:
        #     group_columns = [col for col in column_names if col.startswith(group)]
        #     group_data = selected_data[group_columns]
        #
        #     plt.figure(figsize=(10, 6))
        #     for col in group_columns:
        #         plt.plot(group_data[col], label=col)
        #
        #     plt.title(f"Graphique pour {group}")
        #     plt.xlabel("Index")
        #     plt.ylabel("Valeur")
        #     plt.legend()
        #
        #     # Enregistrer le graphique dans le sous-dossier de l'essai
        #     file_name = f"{group}_graph.png"
        #     file_path_graph = os.path.join(trial_folder_path, file_name)
        #
        #     plt.savefig(file_path_graph)
        #     plt.close()


    # Chemin du nouveau dossier "mean"
    mean_folder_path = os.path.join(folder_path_participant, "MeanSD/")

    # Créer le dossier "mean" s'il n'existe pas déjà
    if not os.path.exists(mean_folder_path):
        os.makedirs(mean_folder_path)

    my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]

    ####### ONE COMPONENT BY GRAPH #######
    axes = [0, 1, 2]  # 0 for X, 1 for Y, 2 for Z


    def sequence_to_indices(sequence):
        # Define a mapping from axis characters to their indices
        axis_mapping = {'x': 0, 'y': 1, 'z': 2}

        # Convert the sequence string to a list of indices
        indices = [axis_mapping[axis] for axis in sequence.strip().lower()]

        return indices

    for index, member_raw in enumerate(Euler_Sequence[:, 0]):
        member = " ".join(member_raw.strip().split())
        axes_seq = Euler_Sequence[index][1]
        indices_single = sequence_to_indices(axes_seq)
        for axis in indices_single:
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

    for index, member_raw in enumerate(Euler_Sequence[:, 0]):
        member = " ".join(member_raw.strip().split())
        axes_seq = Euler_Sequence[index][1]
        indices_single = sequence_to_indices(axes_seq)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        for axis in indices_single:
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


    final_path = folder_path_participant + "Graph_with_body.png"

    # Call the function to create the composite image
    composite_image_path = create_composite_image(graph_images_info, mean_folder_path, save_path=final_path)
    # Call the function
    output_image_path = add_lines_with_arrow_and_circle(composite_image_path, lines_info)
