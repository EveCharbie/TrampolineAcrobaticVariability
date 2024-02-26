import os
import glob
import scipy.io
from scipy.interpolate import interp1d
import pickle
import ezc3d
import scipy.io
import biorbd
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import math


column_names = [
    "PelvisTranslation_X",
    "PelvisTranslation_Y",
    "PelvisTranslation_Z",
    "Pelvis_X",
    "Pelvis_Y",
    "Pelvis_Z",
    "Thorax_X",
    "Thorax_Y",
    "Thorax_Z",
    "Tete_X",
    "Tete_Y",
    "Tete_Z",
    "EpauleD_Y",
    "EpauleD_Z",
    "BrasD_X",
    "BrasD_Y",
    "BrasD_Z",
    "AvBrasD_X",
    "AvBrasD_Z",
    "MainD_X",
    "MainD_Y",
    "EpauleG_Y",
    "EpauleG_Z",
    "BrasG_X",
    "BrasG_Y",
    "BrasG_Z",
    "AvBrasG_X",
    "AvBrasG_Z",
    "MainG_X",
    "MainG_Y",
    "CuisseD_X",
    "CuisseD_Y",
    "CuisseD_Z",
    "JambeD_X",
    "PiedD_X",
    "PiedD_Z",
    "CuisseG_X",
    "CuisseG_Y",
    "CuisseG_Z",
    "JambeG_X",
    "PiedG_X",
    "PiedG_Z",
]


graph_images_info = {
    "Thorax_all_axes_graph.png": (623, -10),
    "Tete_all_axes_graph.png": (960, -10),
    "CuisseG_all_axes_graph.png": (1211, 558),
    "CuisseD_all_axes_graph.png": (358, 558),
    "EpauleG_all_axes_graph.png": (1298, -10),
    "EpauleD_all_axes_graph.png": (285, -10),
    "BrasG_all_axes_graph.png": (1554, 186),
    "BrasD_all_axes_graph.png": (16, 186),
    "AvBrasG_all_axes_graph.png": (1212, 340),
    "AvBrasD_all_axes_graph.png": (358, 340),
    "MainG_all_axes_graph.png": (1554, 443),
    "MainD_all_axes_graph.png": (16, 443),
    "JambeG_all_axes_graph.png": (1544, 804),
    "JambeD_all_axes_graph.png": (53, 804),
    "PiedG_all_axes_graph.png": (1153, 865),
    "PiedD_all_axes_graph.png": (443, 865),
    "Pelvis_all_axes_graph.png": (793, 865),
}


lines_info = {
    "line1": ((952, 400), (1130, 210)),  # Replace with actual coordinates
    "line2": ((952, 450), (794, 210)),  # Replace with actual coordinates
    "line3": ((935, 411), (570, 190)),  # Replace with actual coordinates
    "line4": ((903, 424), (370, 250)),  # Replace with actual coordinates
    "line5": ((898, 496), (720, 436)),  # Replace with actual coordinates
    "line6": ((879, 557), (360, 553)),  # Replace with actual coordinates
    "line7": ((929, 546), (720, 667)),  # Replace with actual coordinates
    "line8": ((929, 649), (400, 890)),  # Replace with actual coordinates
    "line9": ((930, 762), (609, 880)),  # Replace with actual coordinates
    "line10": ((953, 532), (960, 880)),  # Replace with actual coordinates
    "line11": ((976, 762), (1319, 880)),  # Replace with actual coordinates
    "line12": ((976, 649), (1543, 890)),  # Replace with actual coordinates
    "line13": ((976, 546), (1221, 667)),  # Replace with actual coordinates
    "line14": ((1024, 557), (1551, 553)),  # Replace with actual coordinates
    "line15": ((1008, 496), (1222, 436)),  # Replace with actual coordinates
    "line16": ((1000, 424), (1543, 250)),  # Replace with actual coordinates
    "line17": ((971, 411), (1410, 190)),  # Replace with actual coordinates
}


class OrderMatData:
    """
    A class for organize and access easily to the data of a dataframe, especially those from .mat
    files in biomechanical contexts
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Mapping des indices aux suffixes attendus
        self.index_suffix_map = {0: "X", 1: "Y", 2: "Z"}

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


def load_and_interpolate(file, interval, num_points=100):
    """
    Load and interpol the data from a MATLAB file(.mat).

    Args:
        file (str): Path to the .mat file to load.
        interval (tuple): A tuple of two elements specifying the interval of data to extract.
        num_points (int): Number of points for interpolation of data.

    Returns:
        OrderMatData: An instance of the OrderMatData class containing interpolate data.
    """
    # Load data with the DoF
    data = scipy.io.loadmat(file)
    df = pd.DataFrame(data["Q2"]).T

    df.columns = column_names

    # Select data in specify interval
    df_selected = df.iloc[interval[0]: interval[1]]

    # Interpolate each column to have a uniform number of points
    df_interpolated = df_selected.apply(
        lambda x: np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(x)), x)
    )
    # print(df_interpolated.shape)

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


def get_q(Xsens_orientation_per_move):
    """
    This function returns de generalized coordinates in the sequence XYZ (biorbd) from the quaternion of the orientation
    of the Xsens segments.
    The translation is left empty as it has to be computed otherwise.
    I am not sure if I would use this for kinematics analysis, but for visualisation it is not that bad.
    """

    parent_idx_list = {
        "Pelvis": None,  # 0
        "L5": [0, "Pelvis"],  # 1
        "L3": [1, "L5"],  # 2
        "T12": [2, "L3"],  # 3
        "T8": [3, "T12"],  # 4
        "Neck": [4, "T8"],  # 5
        "Head": [5, "Neck"],  # 6
        "ShoulderR": [4, "T8"],  # 7
        "UpperArmR": [7, "ShoulderR"],  # 8
        "LowerArmR": [8, "UpperArmR"],  # 9
        "HandR": [9, "LowerArmR"],  # 10
        "ShoulderL": [4, "T8"],  # 11
        "UpperArmL": [11, "ShoulderR"],  # 12
        "LowerArmL": [12, "UpperArmR"],  # 13
        "HandL": [13, "LowerArmR"],  # 14
        "UpperLegR": [0, "Pelvis"],  # 15
        "LowerLegR": [15, "UpperLegR"],  # 16
        "FootR": [16, "LowerLegR"],  # 17
        "ToesR": [17, "FootR"],  # 18
        "UpperLegL": [0, "Pelvis"],  # 19
        "LowerLegL": [19, "UpperLegL"],  # 20
        "FootL": [20, "LowerLegL"],  # 21
        "ToesL": [21, "FootL"],  # 22
    }

    nb_frames = Xsens_orientation_per_move.shape[0]
    Q = np.zeros((23 * 3, nb_frames))
    rotation_matrices = np.zeros((23, nb_frames, 3, 3))
    for i_segment, key in enumerate(parent_idx_list):
        for i_frame in range(nb_frames):
            Quat_normalized = Xsens_orientation_per_move[i_frame, i_segment * 4 : (i_segment + 1) * 4] / np.linalg.norm(
                Xsens_orientation_per_move[i_frame, i_segment * 4 : (i_segment + 1) * 4]
            )
            Quat = biorbd.Quaternion(Quat_normalized[0], Quat_normalized[1], Quat_normalized[2], Quat_normalized[3])

            RotMat_current = biorbd.Quaternion.toMatrix(Quat).to_array()
            z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi / 2]), "z").to_array()
            RotMat_current = z_rotation @ RotMat_current

            if parent_idx_list[key] is None:
                RotMat = np.eye(3)
            else:
                RotMat = rotation_matrices[parent_idx_list[key][0], i_frame, :, :]

            RotMat_between = np.linalg.inv(RotMat) @ RotMat_current
            RotMat_between = biorbd.Rotation(
                RotMat_between[0, 0],
                RotMat_between[0, 1],
                RotMat_between[0, 2],
                RotMat_between[1, 0],
                RotMat_between[1, 1],
                RotMat_between[1, 2],
                RotMat_between[2, 0],
                RotMat_between[2, 1],
                RotMat_between[2, 2],
            )
            Q[i_segment * 3 : (i_segment + 1) * 3, i_frame] = biorbd.Rotation.toEulerAngles(
                RotMat_between, "xyz"
            ).to_array()

            rotation_matrices[i_segment, i_frame, :, :] = RotMat_current
    return Q


def create_composite_image(
    graph_images_info,
    base_graph_path,
    save_path,
    bg_size=(1920, 1082),
    body_size=(383, 669),
    body_position=(761, 228),
    graph_size=(366, 220),
    border_thickness=0,
):
    background = Image.new("RGB", bg_size, color="white")

    body_image_path = "/home/lim/Documents/StageMathieu/Graph_from_mot/DALL_E_Body.png"
    try:
        body_image = Image.open(body_image_path)
        body_image = body_image.resize(body_size, Image.Resampling.LANCZOS)
        background.paste(body_image, body_position, body_image)
    except FileNotFoundError:
        print("Body image file not found. Please upload the file and try again.")
        return None

    full_graph_images_info = {base_graph_path + filename: position for filename, position in graph_images_info.items()}

    for graph_image_filename, graph_position in full_graph_images_info.items():
        try:
            graph_image = Image.open(graph_image_filename)
            graph_image = graph_image.resize(graph_size, Image.Resampling.LANCZOS)

            if border_thickness > 0:
                border_image = Image.new(
                    "RGB", (graph_size[0] + 2 * border_thickness, graph_size[1] + 2 * border_thickness), color="black"
                )
                border_position = (graph_position[0] - border_thickness, graph_position[1] - border_thickness)
                background.paste(border_image, border_position)
                background.paste(graph_image, graph_position, graph_image)
            else:
                background.paste(graph_image, graph_position, graph_image)
        except FileNotFoundError:
            print(f"Graph image file {graph_image_filename} not found.")
    legend_image_path = base_graph_path + "legend.png"
    try:
        legend_image = Image.open(legend_image_path)
        background.paste(legend_image, (1723, 0))
    except FileNotFoundError:
        print("Legend image file not found")
    try:
        background.save(save_path)
    except Exception as e:
        print(f"Error saving the image: {e}")
        return None
    return save_path


def add_lines_with_arrow_and_circle(
    image_path, lines_info, line_width=2, arrow_size=15, circle_radius=5, scale_factor=4
):
    """
    Draw smooth lines with arrows on one end and circles on the other on the image using a scaling technique for
    anti-aliasing.

    :param image_path: Path to the image where lines will be drawn.
    :param lines_info: A dictionary with keys as line identifiers and values as tuples containing
                       start and end coordinates (x1, y1, x2, y2).
    :param line_width: Width of the lines.
    :param arrow_size: Size of the arrow.
    :param circle_radius: Radius of the circle.
    :param scale_factor: Factor by which to scale the image for drawing.
    :return: Path to the saved image with smooth drawn lines and circles.
    """
    # Load the image and scale it up
    with Image.open(image_path) as img:
        large_img = img.resize((img.width * scale_factor, img.height * scale_factor), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(large_img)

        for start, end in lines_info.values():
            # Scale up coordinates and dimensions
            start_scaled = tuple([x * scale_factor for x in start])
            end_scaled = tuple([x * scale_factor for x in end])
            line_width_scaled = line_width * scale_factor
            arrow_size_scaled = arrow_size * scale_factor
            circle_radius_scaled = circle_radius * scale_factor

            # Draw the line
            draw.line((start_scaled, end_scaled), fill="black", width=line_width_scaled)

            # Calculate arrow direction
            dx = end_scaled[0] - start_scaled[0]
            dy = end_scaled[1] - start_scaled[1]
            angle = math.atan2(dy, dx)

            # Calculate arrow points
            arrow_tip = end_scaled
            arrow_left = (
                end_scaled[0] - arrow_size_scaled * math.cos(angle - math.pi / 6),
                end_scaled[1] - arrow_size_scaled * math.sin(angle - math.pi / 6),
            )
            arrow_right = (
                end_scaled[0] - arrow_size_scaled * math.cos(angle + math.pi / 6),
                end_scaled[1] - arrow_size_scaled * math.sin(angle + math.pi / 6),
            )

            # Draw the arrow
            draw.polygon([arrow_tip, arrow_left, arrow_right], fill="black")

            # Draw a filled circle
            draw.ellipse(
                [
                    (start_scaled[0] - circle_radius_scaled, start_scaled[1] - circle_radius_scaled),
                    (start_scaled[0] + circle_radius_scaled, start_scaled[1] + circle_radius_scaled),
                ],
                fill="black",
            )

        # Resize the image back down with anti-aliasing
        smooth_img = large_img.resize(img.size, Image.Resampling.LANCZOS)

        # Save the modified image
        output_path = image_path.replace(".png", ".png")
        smooth_img.save(output_path)

    return output_path


def recons_kalman(n_frames, num_markers, markers_xsens, model, initial_guess):
    markersOverFrames = []
    for i in range(n_frames):
        node_segment = []
        for j in range(num_markers):
            node_segment.append(biorbd.NodeSegment(markers_xsens[:, j, i].T))
        markersOverFrames.append(node_segment)

    freq = 200
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)
    kalman.setInitState(initial_guess[0], initial_guess[1], initial_guess[2])

    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)
    q_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    qdot_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()
    return q_recons, qdot_recons


def find_index(name, list):
    return list.index(name)


def calculate_rmsd(markers, pos_recons):
    # Vérifier que les formes des tableaux sont identiques
    assert markers.shape == pos_recons.shape, "Les tableaux doivent avoir la même forme."

    n_frames = markers.shape[2]
    rmsd_per_frame = np.zeros(n_frames)

    for i in range(n_frames):
        # Calculer la différence entre les ensembles de marqueurs pour le cadre i
        diff = markers[:, :, i] - pos_recons[:, :, i]
        # Calculer la norme au carré de la différence pour chaque marqueur
        squared_diff = np.nansum(diff ** 2, axis=0)
        # Calculer la moyenne des différences au carré
        mean_squared_diff = np.mean(squared_diff)
        # Calculer la racine carrée de la moyenne des différences au carré pour obtenir la RMSD
        rmsd_per_frame[i] = np.sqrt(mean_squared_diff)

    return rmsd_per_frame


def normalise_vecteurs(vecteurs):
    normes = np.linalg.norm(vecteurs, axis=1)[:, np.newaxis]
    vecteurs_normalises = vecteurs / normes
    return vecteurs_normalises


def dessiner_vecteurs(ax, origine, vecteur_x, vecteur_y, vecteur_z, longueur=0.1):
    ax.quiver(origine[0], origine[1], origine[2], vecteur_x[0], vecteur_x[1], vecteur_x[2], color="r",
              length=longueur, normalize=True)
    ax.quiver(origine[0], origine[1], origine[2], vecteur_y[0], vecteur_y[1], vecteur_y[2], color="g",
              length=longueur, normalize=True)
    ax.quiver(origine[0], origine[1], origine[2], vecteur_z[0], vecteur_z[1], vecteur_z[2], color="b",
              length=longueur, normalize=True)


def transform_point(local_point, rotation_matrix, origin):
    local_point_np = np.array(local_point).T
    local_point_np_reshaped = local_point_np[:, :, np.newaxis]
    rotated_points = rotation_matrix @ local_point_np_reshaped
    rotated_points = rotated_points.squeeze(-1)

    origin_np = np.array(origin)

    global_point = rotated_points + origin_np

    return global_point


def calculate_hjc(pos_marker, EIASD_index, EIASG_index, condintd_index, condintg_index, malintd_index,
                  malintg_index, is_right_side):
    if is_right_side:
        # Utilisation des indices pour le côté droit
        condint_index = condintd_index
        malint_index = malintd_index
    else:
        # Utilisation des indices pour le côté gauche
        condint_index = condintg_index
        malint_index = malintg_index

    diffs_ASIS = (pos_marker[:, EIASD_index, :].T - pos_marker[:, EIASG_index, :].T)
    inter_ASIS_distance = np.linalg.norm(diffs_ASIS, axis=1)

    diffs_length_lower_leg = (pos_marker[:, EIASD_index, :].T - pos_marker[:, condint_index, :].T) if is_right_side \
        else (pos_marker[:, EIASG_index, :].T - pos_marker[:, condint_index, :].T)

    lower_leg_length = np.linalg.norm(diffs_length_lower_leg, axis=1)
    diffs_length_upper_leg = (pos_marker[:, condint_index, :].T - pos_marker[:, malint_index, :].T)
    upper_leg_length = np.linalg.norm(diffs_length_upper_leg, axis=1)

    leg_length_total = upper_leg_length + lower_leg_length

    hip_joint_center_x = (11 - 0.063 * (leg_length_total * 1000)) / 1000
    hip_joint_center_y = (8 + 0.086 * (leg_length_total * 1000)) / 1000
    hip_joint_center_z = (-8 - 0.038 * inter_ASIS_distance - 0.071 * (leg_length_total * 1000)) / 1000 if is_right_side\
        else - ((-8 - 0.038 * inter_ASIS_distance - 0.071 * (leg_length_total * 1000)) / 1000)

    # Création du tableau numpy pour les coordonnées HJC
    hip_joint_center_local = np.array([hip_joint_center_x, hip_joint_center_y, hip_joint_center_z])

    return hip_joint_center_local


def get_orientation_knee_left(pos_marker, marker_name_list):
    condintg_index = find_index("CONDINTG", marker_name_list)
    conextg_index = find_index("CONEXTG", marker_name_list)
    malintg_index = find_index("MALINTG", marker_name_list)
    malextg_index = find_index("MALEXTG", marker_name_list)


    axe_z_knee = (pos_marker[:, conextg_index, :]).T - (pos_marker[:, condintg_index, :]).T
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    mid_cond = ((pos_marker[:, conextg_index, :]).T + (pos_marker[:, condintg_index, :]).T) / 2
    mid_mal = ((pos_marker[:, malextg_index, :]).T + (pos_marker[:, malintg_index, :]).T) / 2

    axe_y_knee = mid_cond - mid_mal
    axe_y_knee = normalise_vecteurs(axe_y_knee)

    axe_x_knee = np.cross(axe_z_knee, axe_y_knee)
    axe_x_knee = normalise_vecteurs(axe_x_knee)

    axe_z_knee = np.cross(axe_x_knee, axe_y_knee)
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_knee, axe_y_knee, axe_z_knee)])

    return matrices_rotation, mid_cond


def get_orientation_knee_right(pos_marker, marker_name_list):
    condintd_index = find_index("CONDINTD", marker_name_list)
    condextd_index = find_index("CONDEXTD", marker_name_list)
    malintd_index = find_index("MALINTD", marker_name_list)
    malextd_index = find_index("MALEXTD", marker_name_list)

    axe_z_knee = (pos_marker[:, condintd_index, :]).T - (pos_marker[:, condextd_index, :]).T
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    mid_cond = ((pos_marker[:, condextd_index, :]).T + (pos_marker[:, condintd_index, :]).T) / 2
    mid_mal = ((pos_marker[:, malextd_index, :]).T + (pos_marker[:, malintd_index, :]).T) / 2

    axe_y_knee = mid_cond - mid_mal
    axe_y_knee = normalise_vecteurs(axe_y_knee)

    axe_x_knee = np.cross(axe_z_knee, axe_y_knee)
    axe_x_knee = normalise_vecteurs(axe_x_knee)

    axe_z_knee = np.cross(axe_x_knee, axe_y_knee)
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_knee, axe_y_knee, axe_z_knee)])

    return matrices_rotation, mid_cond


def predictive_hip_joint_center_location(pos_marker, marker_name_list):
    EIPSD_index = find_index("EIPSD", marker_name_list)
    EIPSG_index = find_index("EIPSG", marker_name_list)
    EIASD_index = find_index("EIASD", marker_name_list)
    EIASG_index = find_index("EIASG", marker_name_list)
    condintd_index = find_index("CONDINTD", marker_name_list)
    condintg_index = find_index("CONDINTG", marker_name_list)
    malintd_index = find_index("MALINTD", marker_name_list)
    malintg_index = find_index("MALINTG", marker_name_list)

    mid_EIAS = ((pos_marker[:, EIASD_index, :]).T + (pos_marker[:, EIASG_index, :]).T) / 2
    mid_EIPS = ((pos_marker[:, EIPSD_index, :]).T + (pos_marker[:, EIPSG_index, :]).T) / 2

    axe_z_pelvic = (pos_marker[:, EIASD_index, :]).T - (pos_marker[:, EIASG_index, :]).T
    axe_z_pelvic = normalise_vecteurs(axe_z_pelvic)

    axe_x_pelvic = mid_EIAS - mid_EIPS
    axe_x_pelvic = normalise_vecteurs(axe_x_pelvic)

    axe_y_pelvic = np.cross(axe_z_pelvic, axe_x_pelvic)
    axe_y_pelvic = normalise_vecteurs(axe_y_pelvic)

    axe_x_pelvic = np.cross(axe_z_pelvic, axe_y_pelvic)
    axe_x_pelvic = normalise_vecteurs(axe_x_pelvic)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_pelvic, axe_y_pelvic, axe_z_pelvic)])

    diffs_pelvic_depth = mid_EIAS + mid_EIPS
    pelvic_depth = np.linalg.norm(diffs_pelvic_depth, axis=1)

    hip_right_joint_center_local = calculate_hjc(pos_marker, EIASD_index, EIASG_index, condintd_index, condintg_index,
                                                 malintd_index, malintg_index, True)

    hip_left_joint_center_local = calculate_hjc(pos_marker, EIASD_index, EIASG_index, condintd_index, condintg_index,
                                                malintd_index, malintg_index, False)

    hip_right_joint_center = transform_point(hip_right_joint_center_local, matrices_rotation, mid_EIAS)
    hip_left_joint_center = transform_point(hip_left_joint_center_local, matrices_rotation, mid_EIAS)

    return hip_right_joint_center, hip_left_joint_center, mid_EIAS, matrices_rotation


def get_orientation_hip(pos_marker, marker_name_list, hjc_center, is_right_side):
    if is_right_side:
        condint_index = find_index("CONDINTD", marker_name_list)
        condext_index = find_index("CONDEXTD", marker_name_list)

    else:
        condint_index = find_index("CONDINTG", marker_name_list)
        condext_index = find_index("CONEXTG", marker_name_list)

    mid_cond = ((pos_marker[:, condext_index, :]).T + (pos_marker[:, condint_index, :]).T) / 2

    axe_y_hip = mid_cond - hjc_center
    axe_y_hip = normalise_vecteurs(axe_y_hip)

    V1 = hjc_center - (pos_marker[:, condext_index, :]).T
    V2 = hjc_center - (pos_marker[:, condint_index, :]).T

    plan_center_cond = np.cross(V1, V2)
    axe_z_hip = np.cross(axe_y_hip, plan_center_cond)
    axe_z_hip = normalise_vecteurs(axe_z_hip)
    axe_z_hip = axe_z_hip if is_right_side else [-i for i in axe_z_hip]

    axe_x_hip = np.cross(axe_y_hip, axe_z_hip)
    axe_x_hip = normalise_vecteurs(axe_x_hip)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_hip, axe_y_hip, axe_z_hip)])

    return matrices_rotation


def get_orientation_ankle(pos_marker, marker_name_list, is_right_side):

    if is_right_side:
        calc_index = find_index("CALCD", marker_name_list)
        malint_index = find_index("MALINTD", marker_name_list)
        malext_index = find_index("MALEXTD", marker_name_list)
        metat1_index = find_index("METAT1D", marker_name_list)
        metat5_index = find_index("METAT5D", marker_name_list)

    else:
        calc_index = find_index("CALCG", marker_name_list)
        malint_index = find_index("MALINTG", marker_name_list)
        malext_index = find_index("MALEXTG", marker_name_list)
        metat1_index = find_index("METAT1G", marker_name_list)
        metat5_index = find_index("METAT5G", marker_name_list)

    mid_meta1 = ((pos_marker[:, metat1_index, :]).T + (pos_marker[:, metat5_index, :]).T) / 2
    mid_mal = ((pos_marker[:, malint_index, :]).T + (pos_marker[:, malext_index, :]).T) / 2

    axe_z_ankle = (pos_marker[:, malext_index, :]).T - (pos_marker[:, malint_index, :]).T if is_right_side \
        else (pos_marker[:, malint_index, :].T - pos_marker[:, malext_index, :].T)
    axe_z_ankle = normalise_vecteurs(axe_z_ankle)

    axe_x_ankle = mid_meta1 - (pos_marker[:, calc_index, :]).T
    axe_x_ankle = normalise_vecteurs(axe_x_ankle)

    axe_y_ankle = np.cross(axe_x_ankle, axe_z_ankle)
    axe_y_ankle = normalise_vecteurs(axe_y_ankle)

    axe_x_ankle = np.cross(axe_z_ankle, axe_y_ankle)
    axe_x_ankle = normalise_vecteurs(axe_x_ankle)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_ankle, axe_y_ankle, axe_z_ankle)])

    return matrices_rotation, mid_meta1


def get_orientation_thorax(pos_marker, marker_name_list):

    manu_index = find_index("MANU", marker_name_list)
    c7_index = find_index("MALINTD", marker_name_list)
    xiphoide_index = find_index("MALEXTD", marker_name_list)
    d10_index = find_index("METAT1D", marker_name_list)

    manu = pos_marker[:, manu_index, :].T
    mid_lower_stern = ((pos_marker[:, xiphoide_index, :]).T + (pos_marker[:, d10_index, :]).T) / 2
    mid_upper_stern = ((pos_marker[:, manu_index, :]).T + (pos_marker[:, c7_index, :]).T) / 2

    axe_y_stern = mid_upper_stern - mid_lower_stern
    axe_y_stern = normalise_vecteurs(axe_y_stern)

    V1 = mid_lower_stern - (pos_marker[:, c7_index, :]).T
    V2 = mid_lower_stern - (pos_marker[:, manu_index, :]).T

    axe_z_stern = np.cross(V2, V1)
    axe_z_stern = normalise_vecteurs(axe_z_stern)

    axe_x_stern = np.cross(axe_z_stern, axe_y_stern)
    axe_x_stern = normalise_vecteurs(axe_x_stern)

    axe_y_stern = np.cross(axe_z_stern, axe_x_stern)
    axe_y_stern = normalise_vecteurs(axe_y_stern)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_stern, axe_y_stern, axe_z_stern)])

    return matrices_rotation, manu


def get_orientation_elbow(pos_marker, marker_name_list, is_right_side):

    if is_right_side:
        epicon_index = find_index("EPICOND", marker_name_list)
        epitro_index = find_index("EPITROD", marker_name_list)
        ulna_index = find_index("ULNAD", marker_name_list)
        radius_index = find_index("RADIUSD", marker_name_list)

    else:
        epicon_index = find_index("EPICONG", marker_name_list)
        epitro_index = find_index("EPITROG", marker_name_list)
        ulna_index = find_index("ULNAG", marker_name_list)
        radius_index = find_index("RADIUSG", marker_name_list)

    mid_epi = ((pos_marker[:, epicon_index, :]).T + (pos_marker[:, epitro_index, :]).T) / 2
    mid_ul_rad = ((pos_marker[:, ulna_index, :]).T + (pos_marker[:, radius_index, :]).T) / 2

    axe_z_elbow = (pos_marker[:, epitro_index, :]).T - (pos_marker[:, epicon_index, :]).T if is_right_side \
        else (pos_marker[:, epicon_index, :].T - pos_marker[:, epitro_index, :].T)
    axe_z_elbow = normalise_vecteurs(axe_z_elbow)

    axe_y_elbow = mid_epi - mid_ul_rad
    axe_y_elbow = normalise_vecteurs(axe_y_elbow)

    axe_x_elbow = np.cross(axe_y_elbow, axe_z_elbow)
    axe_x_elbow = normalise_vecteurs(axe_x_elbow)

    axe_z_elbow = np.cross(axe_x_elbow, axe_y_elbow)
    axe_z_elbow = normalise_vecteurs(axe_z_elbow)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_elbow, axe_y_elbow, axe_z_elbow)])

    return matrices_rotation, mid_epi


def get_orientation_wrist(pos_marker, marker_name_list, is_right_side):

    if is_right_side:
        ulna_index = find_index("ULNAD", marker_name_list)
        radius_index = find_index("RADIUSD", marker_name_list)
        midmetac3_index = find_index("MIDMETAC3D", marker_name_list)

    else:
        ulna_index = find_index("ULNAG", marker_name_list)
        radius_index = find_index("RADIUSG", marker_name_list)
        midmetac3_index = find_index("MIDMETAC3G", marker_name_list)

    mid_ul_rad = ((pos_marker[:, ulna_index, :]).T + (pos_marker[:, radius_index, :]).T) / 2

    axe_z_wrist = (pos_marker[:, radius_index, :]).T - (pos_marker[:, ulna_index, :]).T if is_right_side \
        else (pos_marker[:, ulna_index, :].T - pos_marker[:, radius_index, :].T)
    axe_z_wrist = normalise_vecteurs(axe_z_wrist)

    axe_y_wrist = pos_marker[:, midmetac3_index, :].T - mid_ul_rad
    axe_y_wrist = normalise_vecteurs(axe_y_wrist)

    axe_x_wrist = np.cross(axe_y_wrist, axe_z_wrist)
    axe_x_wrist = normalise_vecteurs(axe_x_wrist)

    axe_z_wrist = np.cross(axe_x_wrist, axe_y_wrist)
    axe_z_wrist = normalise_vecteurs(axe_z_wrist)

    matrices_rotation = np.array([np.column_stack([x, y, z]) for x, y, z in zip(axe_x_wrist, axe_y_wrist, axe_z_wrist)])

    return matrices_rotation, mid_ul_rad

