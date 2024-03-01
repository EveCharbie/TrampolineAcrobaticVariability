import numpy as np
import matplotlib.pyplot as plt
import os
import biorbd
import ezc3d
import bioviz
import pandas as pd
from Draw_function import dessiner_vecteurs

from Function_Class_Basics import (
    find_index,
    calculate_rmsd,
)

from Build_model_function import (
    recons_kalman_with_marker,
    get_orientation_knee_left,
    get_orientation_knee_right,
    predictive_hip_joint_center_location,
    get_orientation_ankle,
    get_orientation_hip,
    get_orientation_thorax,
    get_orientation_elbow,
    get_orientation_wrist,
    get_orientation_head,
    get_orientation_shoulder,
)
from matplotlib.animation import FuncAnimation

model = biorbd.Model("/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Sarah.s2mMod")
# Chemin du dossier contenant les fichiers .c3d
file_path_c3d = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Tests/"

# Chemin du dossier de sortie pour les graphiques
folder_path = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/"

file_intervals = [
    (file_path_c3d + "Relax.c3d", (0, 50)),
]

##
for file_path, interval in file_intervals:
    file_name = os.path.basename(file_path).split(".")[0]
    print(f"{file_name} is running")
    c = ezc3d.c3d(file_path)
    point_data = c["data"]["points"][:, :, interval[0] : interval[1]]
    n_markers = point_data.shape[1]
    nf_mocap = point_data.shape[2]
    f_mocap = c["parameters"]["POINT"]["RATE"]["value"][0]
    point_labels = c["parameters"]["POINT"]["LABELS"]
    # Extract useful marker name from point_labels
    useful_labels = [
        label for label in point_labels["value"] if not label.startswith("*")
    ]

    sample_label = useful_labels[0]
    typical_dimensions = point_data[0][
        find_index(sample_label, point_labels["value"])
    ].shape[0]
# Extract marker name order from model
    desired_order = [
        model.markerNames()[i].to_string() for i in range(model.nbMarkers())
    ]

    n_markers_desired = len(desired_order)
    reordered_point_data = np.full((4, n_markers_desired, typical_dimensions), np.nan)
# Reorder marker based on list from model
    for i, marker in enumerate(desired_order):
        marker_found = False
        for label in useful_labels:
            if marker in label:  # Check if "marker" is a substring of "label".
                original_index = find_index(label, point_labels["value"])
                reordered_point_data[:, i, :] = point_data[:, original_index, :]
                marker_found = True
                break

        if not marker_found:
            print(
                f"Le marqueur '{marker}' n'a pas été trouvé et a été initialisé avec NaN."
            )
            pass

    n_markers_reordered = reordered_point_data.shape[1]

    # Take 3 first col, convert mm in m and possibility to reverse direction
    markers = np.zeros((3, n_markers_reordered, nf_mocap))
    for i in range(nf_mocap):
        for j in range(n_markers_reordered):
            markers[:, j, i] = reordered_point_data[:3, j, i]
            # markers[:, j, nf_mocap-1-i] = reordered_point_data[:3, j, i]
    markers = markers / 1000

    frame_index = 0
    # frame_index = nf_mocap-1
    start_frame = markers[:, :, frame_index : frame_index + 1]
    if start_frame.shape != (3, n_markers_reordered, 1):
        raise ValueError(
            f"Dimension incorrecte pour 'specific_frame'. Attendu: (3, {n_markers_reordered}, 1), Obtenu: "
            f"{start_frame.shape}"
        )

    ik = biorbd.InverseKinematics(model, start_frame)
    ik.solve("only_lm")
    Q = ik.q
    Q_1d = Q.flatten() if Q.ndim > 1 else Q
    Qdot_1d = np.zeros(Q_1d.shape)
    Qddot_1d = np.zeros(Q_1d.shape)

    # Create initial guess with 1D vector
    initial_guess = (Q_1d, Qdot_1d, Qddot_1d)

    q_recons, qdot_recons, pos_recons = recons_kalman_with_marker(
        nf_mocap, n_markers_reordered, markers, model, initial_guess
    )
    # b = bioviz.Viz(loaded_model=model)
    # b.load_movement(q_recons)
    # b.load_experimental_markers(markers[:, :, :])
    # b.exec()

    rmsd_by_frame = calculate_rmsd(markers, pos_recons)

    origine = np.zeros((q_recons.shape[1], 3))
    matrice_origin = np.array([np.eye(3) for _ in range(q_recons.shape[1])])

    (
        hip_right_joint_center,
        hip_left_joint_center,
        pelvic_origin,
        matrices_rotation_pelvic,
    ) = predictive_hip_joint_center_location(pos_recons, desired_order)

    matrices_rotation_hip_right = get_orientation_hip(
        pos_recons, desired_order, hip_right_joint_center, True
    )
    matrices_rotation_hip_left = get_orientation_hip(
        pos_recons, desired_order, hip_left_joint_center, False
    )

    matrices_rotation_knee_right, mid_cond_right = get_orientation_knee_right(
        pos_recons, desired_order
    )
    matrices_rotation_knee_left, mid_cond_left = get_orientation_knee_left(
        pos_recons, desired_order
    )

    matrices_rotation_ankle_right, mid_mal_right = get_orientation_ankle(
        pos_recons, desired_order, True
    )
    matrices_rotation_ankle_left, mid_mal_left = get_orientation_ankle(
        pos_recons, desired_order, False
    )

    matrices_rotation_thorax, manu = get_orientation_thorax(pos_recons, desired_order)

    matrices_rotation_head, head_joint_center = get_orientation_head(
        pos_recons, desired_order
    )

    matrices_rotation_shoulder_right, mid_acr_right = get_orientation_shoulder(
        pos_recons, desired_order, True
    )
    matrices_rotation_shoulder_left, mid_acr_left = get_orientation_shoulder(
        pos_recons, desired_order, False
    )

    matrices_rotation_elbow_right, mid_epi_right = get_orientation_elbow(
        pos_recons, desired_order, True
    )
    matrices_rotation_elbow_left, mid_epi_left = get_orientation_elbow(
        pos_recons, desired_order, False
    )

    matrices_rotation_wrist_right, mid_ul_rad_right = get_orientation_wrist(
        pos_recons, desired_order, True
    )
    matrices_rotation_wrist_left, mid_ul_rad_left = get_orientation_wrist(
        pos_recons, desired_order, False
    )

    rot_mat = np.stack(
        [
            matrices_rotation_pelvic,
            matrices_rotation_hip_right,
            matrices_rotation_hip_left,
            matrices_rotation_knee_right,
            matrices_rotation_knee_left,
            matrices_rotation_ankle_right,
            matrices_rotation_ankle_left,
            matrices_rotation_thorax,
            matrices_rotation_head,
            matrices_rotation_shoulder_right,
            matrices_rotation_shoulder_left,
            matrices_rotation_elbow_right,
            matrices_rotation_elbow_left,
            matrices_rotation_wrist_right,
            matrices_rotation_wrist_left,
        ],
        axis=0,
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])

    def init():
        ax.clear()
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([-3, 3])

    def update(frame):
        ax.clear()
        origin_left = mid_cond_left[frame]
        origin_right = mid_cond_right[frame]
        origin_pelvic = pelvic_origin[frame]
        origin_hjc_right = hip_right_joint_center[frame]
        origin_hjc_left = hip_left_joint_center[frame]
        origin_ankle_right = mid_mal_right[frame]
        origin_ankle_left = mid_mal_left[frame]
        origin_thorax = manu[frame]
        origin = origine[frame]
        origin_ul_rad_right = mid_ul_rad_right[frame]
        origin_ul_rad_left = mid_ul_rad_left[frame]
        origin_epi_right = mid_epi_right[frame]
        origin_epi_left = mid_epi_left[frame]
        origin_head = head_joint_center[frame]
        origin_acr_left = mid_acr_left[frame]
        origin_acr_right = mid_acr_right[frame]

        dessiner_vecteurs(
            ax,
            origin_left,
            matrices_rotation_knee_left[frame][:, 0],
            matrices_rotation_knee_left[frame][:, 1],
            matrices_rotation_knee_left[frame][:, 2],
        )
        dessiner_vecteurs(
            ax,
            origin_right,
            matrices_rotation_knee_right[frame][:, 0],
            matrices_rotation_knee_right[frame][:, 1],
            matrices_rotation_knee_right[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_pelvic,
            matrices_rotation_pelvic[frame][:, 0],
            matrices_rotation_pelvic[frame][:, 1],
            matrices_rotation_pelvic[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_hjc_right,
            matrices_rotation_hip_right[frame][:, 0],
            matrices_rotation_hip_right[frame][:, 1],
            matrices_rotation_hip_right[frame][:, 2],
        )
        dessiner_vecteurs(
            ax,
            origin_hjc_left,
            matrices_rotation_hip_left[frame][:, 0],
            matrices_rotation_hip_left[frame][:, 1],
            matrices_rotation_hip_left[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_ankle_right,
            matrices_rotation_ankle_right[frame][:, 0],
            matrices_rotation_ankle_right[frame][:, 1],
            matrices_rotation_ankle_right[frame][:, 2],
        )
        dessiner_vecteurs(
            ax,
            origin_ankle_left,
            matrices_rotation_ankle_left[frame][:, 0],
            matrices_rotation_ankle_left[frame][:, 1],
            matrices_rotation_ankle_left[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_thorax,
            matrices_rotation_thorax[frame][:, 0],
            matrices_rotation_thorax[frame][:, 1],
            matrices_rotation_thorax[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_ul_rad_right,
            matrices_rotation_wrist_right[frame][:, 0],
            matrices_rotation_wrist_right[frame][:, 1],
            matrices_rotation_wrist_right[frame][:, 2],
        )
        dessiner_vecteurs(
            ax,
            origin_ul_rad_left,
            matrices_rotation_wrist_left[frame][:, 0],
            matrices_rotation_wrist_left[frame][:, 1],
            matrices_rotation_wrist_left[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_epi_right,
            matrices_rotation_elbow_right[frame][:, 0],
            matrices_rotation_elbow_right[frame][:, 1],
            matrices_rotation_elbow_right[frame][:, 2],
        )
        dessiner_vecteurs(
            ax,
            origin_epi_left,
            matrices_rotation_elbow_left[frame][:, 0],
            matrices_rotation_elbow_left[frame][:, 1],
            matrices_rotation_elbow_left[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_acr_right,
            matrices_rotation_shoulder_right[frame][:, 0],
            matrices_rotation_shoulder_right[frame][:, 1],
            matrices_rotation_shoulder_right[frame][:, 2],
        )
        dessiner_vecteurs(
            ax,
            origin_acr_left,
            matrices_rotation_shoulder_left[frame][:, 0],
            matrices_rotation_shoulder_left[frame][:, 1],
            matrices_rotation_shoulder_left[frame][:, 2],
        )

        dessiner_vecteurs(
            ax,
            origin_head,
            matrices_rotation_head[frame][:, 0],
            matrices_rotation_head[frame][:, 1],
            matrices_rotation_head[frame][:, 2],
        )

        for ma in range(pos_recons.shape[1]):
            xpos, ypos, zpos = pos_recons[:, ma, frame]
            ax.scatter(xpos, ypos, zpos, s=10, c="b")

        # Création de l'animation

    ani = FuncAnimation(
        fig, update, frames=range(pos_recons.shape[2]), init_func=init, blit=False
    )

    plt.show()

    # markers_by_segment = {}
    # for i in range(model.nbMarkers()):
    #     marker_name = model.markerNames()[i].to_string()
    #     marker_indice = model.marker(i)
    #     parent_index = marker_indice.parentId()
    #     # Vérifier que l'indice du parent est valide
    #     if 0 <= parent_index < model.nbSegment():
    #         segment_name = model.segment(parent_index).name().to_string()
    #         if segment_name not in markers_by_segment:
    #             markers_by_segment[segment_name] = []
    #         markers_by_segment[segment_name].append(marker_name)
    #     else:
    #         print(f"Indice de segment invalide pour le marqueur {marker_name}: {parent_index}")
    # # Afficher les marqueurs regroupés par segment
    # for segment, markers in markers_by_segment.items():
    #     print(f"Segment: {segment}, Marqueurs: {markers}")

    # Select frame
    frame = 0
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    origin_left = mid_cond_left[frame]
    origin_right = mid_cond_right[frame]
    origin_pelvic = pelvic_origin[frame]
    origin_hjc_right = hip_right_joint_center[frame]
    origin_hjc_left = hip_left_joint_center[frame]
    origin_ankle_right = mid_mal_right[frame]
    origin_ankle_left = mid_mal_left[frame]
    origin_thorax = manu[frame]
    # origin = origine[frame]
    origin_ul_rad_right = mid_ul_rad_right[frame]
    origin_ul_rad_left = mid_ul_rad_left[frame]
    origin_epi_right = mid_epi_right[frame]
    origin_epi_left = mid_epi_left[frame]
    origin_head = head_joint_center[frame]
    origin_acr_left = mid_acr_left[frame]
    origin_acr_right = mid_acr_right[frame]

    dessiner_vecteurs(
        ax,
        origin_left,
        matrices_rotation_knee_left[frame][:, 0],
        matrices_rotation_knee_left[frame][:, 1],
        matrices_rotation_knee_left[frame][:, 2],
    )
    dessiner_vecteurs(
        ax,
        origin_right,
        matrices_rotation_knee_right[frame][:, 0],
        matrices_rotation_knee_right[frame][:, 1],
        matrices_rotation_knee_right[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_pelvic,
        matrices_rotation_pelvic[frame][:, 0],
        matrices_rotation_pelvic[frame][:, 1],
        matrices_rotation_pelvic[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_hjc_right,
        matrices_rotation_hip_right[frame][:, 0],
        matrices_rotation_hip_right[frame][:, 1],
        matrices_rotation_hip_right[frame][:, 2],
    )
    dessiner_vecteurs(
        ax,
        origin_hjc_left,
        matrices_rotation_hip_left[frame][:, 0],
        matrices_rotation_hip_left[frame][:, 1],
        matrices_rotation_hip_left[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_ankle_right,
        matrices_rotation_ankle_right[frame][:, 0],
        matrices_rotation_ankle_right[frame][:, 1],
        matrices_rotation_ankle_right[frame][:, 2],
    )
    dessiner_vecteurs(
        ax,
        origin_ankle_left,
        matrices_rotation_ankle_left[frame][:, 0],
        matrices_rotation_ankle_left[frame][:, 1],
        matrices_rotation_ankle_left[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_thorax,
        matrices_rotation_thorax[frame][:, 0],
        matrices_rotation_thorax[frame][:, 1],
        matrices_rotation_thorax[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_ul_rad_right,
        matrices_rotation_wrist_right[frame][:, 0],
        matrices_rotation_wrist_right[frame][:, 1],
        matrices_rotation_wrist_right[frame][:, 2],
    )
    dessiner_vecteurs(
        ax,
        origin_ul_rad_left,
        matrices_rotation_wrist_left[frame][:, 0],
        matrices_rotation_wrist_left[frame][:, 1],
        matrices_rotation_wrist_left[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_epi_right,
        matrices_rotation_elbow_right[frame][:, 0],
        matrices_rotation_elbow_right[frame][:, 1],
        matrices_rotation_elbow_right[frame][:, 2],
    )
    dessiner_vecteurs(
        ax,
        origin_epi_left,
        matrices_rotation_elbow_left[frame][:, 0],
        matrices_rotation_elbow_left[frame][:, 1],
        matrices_rotation_elbow_left[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_acr_right,
        matrices_rotation_shoulder_right[frame][:, 0],
        matrices_rotation_shoulder_right[frame][:, 1],
        matrices_rotation_shoulder_right[frame][:, 2],
    )
    dessiner_vecteurs(
        ax,
        origin_acr_left,
        matrices_rotation_shoulder_left[frame][:, 0],
        matrices_rotation_shoulder_left[frame][:, 1],
        matrices_rotation_shoulder_left[frame][:, 2],
    )

    dessiner_vecteurs(
        ax,
        origin_head,
        matrices_rotation_head[frame][:, 0],
        matrices_rotation_head[frame][:, 1],
        matrices_rotation_head[frame][:, 2],
    )

    for m in range(pos_recons.shape[1]):
        x, y, z = pos_recons[:, m, frame]
        ax.scatter(x, y, z, s=10, c="b")

    plt.show()
