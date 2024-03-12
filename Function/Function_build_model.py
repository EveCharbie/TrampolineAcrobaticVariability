import os
import numpy as np
import biorbd
import ezc3d
from .Function_Class_Basics import find_index, normalise_vecteurs, calculate_rmsd
from scipy.linalg import svd


def recons_kalman_with_marker(
    n_frames, num_markers, markers_xsens, model, initial_guess
):
    # Préparation comme avant
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
    markers_recons = np.ndarray(
        (3, num_markers, len(markersOverFrames))
    )  # Pour stocker les positions reconstruites des marqueurs

    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()

        # Nouveau : Calculer et stocker les positions reconstruites des marqueurs pour ce cadre
        markers_reconstructed = model.markers(Q)
        for m, marker_recons in enumerate(markers_reconstructed):
            markers_recons[:, m, i] = marker_recons.to_array()

    return q_recons, qdot_recons, markers_recons


def transform_point(local_point, rotation_matrix, origin):
    local_point_np = np.array(local_point).T
    local_point_np_reshaped = local_point_np[:, :, np.newaxis]
    rotated_points = rotation_matrix @ local_point_np_reshaped
    rotated_points = rotated_points.squeeze(-1)

    origin_np = np.array(origin)

    global_point = rotated_points + origin_np

    return global_point


def calculate_hjc(
    pos_marker,
    EIASD_index,
    EIASG_index,
    condintd_index,
    condintg_index,
    malintd_index,
    malintg_index,
    is_right_side,
):
    if is_right_side:
        # Utilisation des indices pour le côté droit
        condint_index = condintd_index
        malint_index = malintd_index
        EIAS_index = EIASD_index
    else:
        # Utilisation des indices pour le côté gauche
        condint_index = condintg_index
        malint_index = malintg_index
        EIAS_index = EIASG_index

    diffs_ASIS = pos_marker[:, EIASD_index, :].T - pos_marker[:, EIASG_index, :].T
    inter_ASIS_distance = np.linalg.norm(diffs_ASIS, axis=1)

    diffs_length_lower_leg = (
        (pos_marker[:, EIAS_index, :].T - pos_marker[:, condint_index, :].T)
    )
    lower_leg_length = np.linalg.norm(diffs_length_lower_leg, axis=1)

    diffs_length_upper_leg = (
        pos_marker[:, condint_index, :].T - pos_marker[:, malint_index, :].T
    )
    upper_leg_length = np.linalg.norm(diffs_length_upper_leg, axis=1)

    leg_length_total = upper_leg_length + lower_leg_length

    hip_joint_center_x = (11 - 0.063 * (leg_length_total * 1000)) / 1000
    hip_joint_center_y = (
        -(8 + 0.086 * (leg_length_total * 1000)) / 1000
        if is_right_side
        else (8 + 0.086 * (leg_length_total * 1000)) / 1000
    )
    # Other predictive method
    # hip_joint_center_z = (
    #     -8 - 0.038 * inter_ASIS_distance - 0.071 * (leg_length_total * 1000)
    # ) / 1000

    hip_joint_center_z = (
        -9 - 0.078 * (leg_length_total * 1000)
    ) / 1000

    hip_joint_center_local = np.array(
        [hip_joint_center_x, hip_joint_center_y, hip_joint_center_z]
    )

    return hip_joint_center_local


def get_orientation_knee_left(pos_marker, marker_name_list):
    condintg_index = find_index("CONDINTG", marker_name_list)
    conextg_index = find_index("CONEXTG", marker_name_list)
    malintg_index = find_index("MALINTG", marker_name_list)
    malextg_index = find_index("MALEXTG", marker_name_list)

    axe_z_knee = (pos_marker[:, conextg_index, :]).T - (
        pos_marker[:, condintg_index, :]
    ).T
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    mid_cond = (
        (pos_marker[:, conextg_index, :]).T + (pos_marker[:, condintg_index, :]).T
    ) / 2
    mid_mal = (
        (pos_marker[:, malextg_index, :]).T + (pos_marker[:, malintg_index, :]).T
    ) / 2

    axe_y_knee = mid_cond - mid_mal
    axe_y_knee = normalise_vecteurs(axe_y_knee)

    axe_x_knee = np.cross(axe_z_knee, axe_y_knee)
    axe_x_knee = normalise_vecteurs(axe_x_knee)

    axe_z_knee = np.cross(axe_x_knee, axe_y_knee)
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_x_knee, axe_y_knee, axe_z_knee)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_knee, axe_x_knee, axe_y_knee)
        ]
    )

    return matrices_rotation, mid_cond


def get_orientation_knee_right(pos_marker, marker_name_list):
    condintd_index = find_index("CONDINTD", marker_name_list)
    condextd_index = find_index("CONDEXTD", marker_name_list)
    malintd_index = find_index("MALINTD", marker_name_list)
    malextd_index = find_index("MALEXTD", marker_name_list)

    axe_z_knee = (pos_marker[:, condintd_index, :]).T - (
        pos_marker[:, condextd_index, :]
    ).T
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    mid_cond = (
        (pos_marker[:, condextd_index, :]).T + (pos_marker[:, condintd_index, :]).T
    ) / 2
    mid_mal = (
        (pos_marker[:, malextd_index, :]).T + (pos_marker[:, malintd_index, :]).T
    ) / 2

    axe_y_knee = mid_cond - mid_mal
    axe_y_knee = normalise_vecteurs(axe_y_knee)

    axe_x_knee = np.cross(axe_z_knee, axe_y_knee)
    axe_x_knee = normalise_vecteurs(axe_x_knee)

    axe_z_knee = np.cross(axe_x_knee, axe_y_knee)
    axe_z_knee = normalise_vecteurs(axe_z_knee)

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_x_knee, axe_y_knee, axe_z_knee)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_knee, axe_x_knee, axe_y_knee)
        ]
    )

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

    mid_EIAS = (
        (pos_marker[:, EIASD_index, :]).T + (pos_marker[:, EIASG_index, :]).T
    ) / 2
    mid_EIPS = (
        (pos_marker[:, EIPSD_index, :]).T + (pos_marker[:, EIPSG_index, :]).T
    ) / 2

    # Repere pelvis classique
    # axe_z_pelvic = (pos_marker[:, EIASD_index, :]).T - (pos_marker[:, EIASG_index, :]).T
    # axe_z_pelvic = normalise_vecteurs(axe_z_pelvic)
    # axe_x_pelvic = mid_EIAS - mid_EIPS
    # axe_x_pelvic = normalise_vecteurs(axe_x_pelvic)
    # axe_y_pelvic = np.cross(axe_z_pelvic, axe_x_pelvic)
    # axe_y_pelvic = normalise_vecteurs(axe_y_pelvic)
    # # axe_y_pelvic = [-i for i in axe_y_pelvic]
    # axe_x_pelvic = np.cross(axe_z_pelvic, axe_y_pelvic)
    # axe_x_pelvic = normalise_vecteurs(axe_x_pelvic)
    # axe_x_pelvic = [-i for i in axe_x_pelvic]

    # Repere pelvis selon article pour estimation des HJC
    B1 = mid_EIAS - mid_EIPS
    B2 = (pos_marker[:, EIASG_index, :]).T - (pos_marker[:, EIASD_index, :]).T
    axe_y_pelvic = normalise_vecteurs(B2)

    axe_x_pelvic = np.cross(B1, axe_y_pelvic)
    axe_x_pelvic = normalise_vecteurs(axe_x_pelvic)

    axe_z_pelvic = np.cross(axe_y_pelvic, axe_x_pelvic)
    axe_z_pelvic = normalise_vecteurs(axe_z_pelvic)

    matrices_rotation_for_hjc = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_x_pelvic, axe_y_pelvic, axe_z_pelvic)
        ]
    )
    axe_y_pelvic = [-i for i in axe_y_pelvic]

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_z_pelvic, axe_x_pelvic, axe_y_pelvic)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_y_pelvic, axe_z_pelvic, axe_x_pelvic)
        ]
    )

    hip_right_joint_center_local = calculate_hjc(
        pos_marker,
        EIASD_index,
        EIASG_index,
        condintd_index,
        condintg_index,
        malintd_index,
        malintg_index,
        True,
    )

    hip_left_joint_center_local = calculate_hjc(
        pos_marker,
        EIASD_index,
        EIASG_index,
        condintd_index,
        condintg_index,
        malintd_index,
        malintg_index,
        False,
    )

    hip_right_joint_center = transform_point(
        hip_right_joint_center_local, matrices_rotation_for_hjc, mid_EIAS
    )
    hip_left_joint_center = transform_point(
        hip_left_joint_center_local, matrices_rotation_for_hjc, mid_EIAS
    )

    return hip_right_joint_center, hip_left_joint_center, mid_EIAS, matrices_rotation


def get_orientation_hip(pos_marker, marker_name_list, hjc_center, is_right_side):

    if is_right_side:
        condint_index = find_index("CONDINTD", marker_name_list)
        condext_index = find_index("CONDEXTD", marker_name_list)

    else:
        condint_index = find_index("CONDINTG", marker_name_list)
        condext_index = find_index("CONEXTG", marker_name_list)

    mid_cond = (
        (pos_marker[:, condext_index, :]).T + (pos_marker[:, condint_index, :]).T
    ) / 2

    axe_y_hip = mid_cond - hjc_center
    axe_y_hip = normalise_vecteurs(axe_y_hip)
    axe_y_hip = [-i for i in axe_y_hip]

    V1_hjc = hjc_center - (pos_marker[:, condext_index, :]).T
    V2_hjc = hjc_center - (pos_marker[:, condint_index, :]).T

    plan_center_cond = np.cross(V1_hjc, V2_hjc)
    axe_z_hip = np.cross(axe_y_hip, plan_center_cond)
    axe_z_hip = normalise_vecteurs(axe_z_hip)
    axe_z_hip = [-i for i in axe_z_hip] if is_right_side else axe_z_hip

    axe_x_hip = np.cross(axe_z_hip, axe_y_hip)
    axe_x_hip = normalise_vecteurs(axe_x_hip)
    axe_x_hip = [-i for i in axe_x_hip]

    # matrices_rotation = np.array(
    #     [np.column_stack([x, y, z]) for x, y, z in zip(axe_x_hip, axe_y_hip, axe_z_hip)]
    # )
    matrices_rotation = np.array(
        [np.column_stack([x, y, z]) for x, y, z in zip(axe_z_hip, axe_x_hip, axe_y_hip)]
    )

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

    mid_meta1 = (
        (pos_marker[:, metat1_index, :]).T + (pos_marker[:, metat5_index, :]).T
    ) / 2
    mid_mal = (
        (pos_marker[:, malint_index, :]).T + (pos_marker[:, malext_index, :]).T
    ) / 2

    axe_z_ankle = (
        (pos_marker[:, malext_index, :]).T - (pos_marker[:, malint_index, :]).T
        if is_right_side
        else (pos_marker[:, malint_index, :].T - pos_marker[:, malext_index, :].T)
    )
    axe_z_ankle = normalise_vecteurs(axe_z_ankle)

    axe_x_ankle = mid_meta1 - (pos_marker[:, calc_index, :]).T
    axe_x_ankle = normalise_vecteurs(axe_x_ankle)

    axe_y_ankle = np.cross(axe_x_ankle, axe_z_ankle)
    axe_y_ankle = normalise_vecteurs(axe_y_ankle)
    axe_y_ankle = [-i for i in axe_y_ankle]

    axe_x_ankle = np.cross(axe_z_ankle, axe_y_ankle)
    axe_x_ankle = normalise_vecteurs(axe_x_ankle)

# reverse x and y to have rot int ext on y
#     matrices_rotation = np.array(
#         [
#             np.column_stack([x, y, z])
#             for x, y, z in zip(axe_y_ankle, axe_x_ankle, axe_z_ankle)
#         ]
#     )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_ankle, axe_y_ankle, axe_x_ankle)
        ]
    )

    return matrices_rotation, mid_mal


def get_orientation_thorax(pos_marker, marker_name_list):

    manu_index = find_index("MANU", marker_name_list)
    c7_index = find_index("C7", marker_name_list)
    xiphoide_index = find_index("XIPHOIDE", marker_name_list)
    d10_index = find_index("D10", marker_name_list)

    manu = pos_marker[:, manu_index, :].T
    mid_lower_stern = (
        (pos_marker[:, xiphoide_index, :]).T + (pos_marker[:, d10_index, :]).T
    ) / 2
    mid_upper_stern = (
        (pos_marker[:, manu_index, :]).T + (pos_marker[:, c7_index, :]).T
    ) / 2

    axe_y_stern = mid_upper_stern - mid_lower_stern
    axe_y_stern = normalise_vecteurs(axe_y_stern)

    V1_thorax = mid_lower_stern - (pos_marker[:, c7_index, :]).T
    V2_thorax = mid_lower_stern - manu

    axe_z_stern = np.cross(V2_thorax, V1_thorax)
    axe_z_stern = normalise_vecteurs(axe_z_stern)

    axe_x_stern = np.cross(axe_z_stern, axe_y_stern)
    axe_x_stern = normalise_vecteurs(axe_x_stern)
    axe_x_stern = [-i for i in axe_x_stern]

    # axe_y_stern = np.cross(axe_z_stern, axe_x_stern)
    # axe_y_stern = normalise_vecteurs(axe_y_stern)

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_x_stern, axe_y_stern, axe_z_stern)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_stern, axe_x_stern, axe_y_stern)
        ]
    )

    return matrices_rotation, mid_lower_stern


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

    mid_epi = (
        (pos_marker[:, epicon_index, :]).T + (pos_marker[:, epitro_index, :]).T
    ) / 2
    mid_ul_rad = (
        (pos_marker[:, ulna_index, :]).T + (pos_marker[:, radius_index, :]).T
    ) / 2

    axe_z_elbow = (
        (pos_marker[:, epitro_index, :]).T - (pos_marker[:, epicon_index, :]).T
        if is_right_side
        else (pos_marker[:, epicon_index, :].T - pos_marker[:, epitro_index, :].T)
    )
    axe_z_elbow = normalise_vecteurs(axe_z_elbow)

    axe_y_elbow = mid_epi - mid_ul_rad if is_right_side else mid_ul_rad - mid_epi
    axe_y_elbow = normalise_vecteurs(axe_y_elbow)
    axe_y_elbow = axe_y_elbow if is_right_side else [-i for i in axe_y_elbow]

    axe_x_elbow = np.cross(axe_y_elbow, axe_z_elbow)
    axe_x_elbow = normalise_vecteurs(axe_x_elbow)
    axe_x_elbow = [-i for i in axe_x_elbow]

    axe_z_elbow = np.cross(axe_x_elbow, axe_y_elbow)
    axe_z_elbow = normalise_vecteurs(axe_z_elbow)

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_x_elbow, axe_y_elbow, axe_z_elbow)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_elbow, axe_x_elbow, axe_y_elbow)
        ]
    )

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

    mid_ul_rad = (
        (pos_marker[:, ulna_index, :]).T + (pos_marker[:, radius_index, :]).T
    ) / 2

    axe_z_wrist = (
        (pos_marker[:, radius_index, :]).T - (pos_marker[:, ulna_index, :]).T
        if is_right_side
        else (pos_marker[:, ulna_index, :].T - pos_marker[:, radius_index, :].T)
    )
    axe_z_wrist = normalise_vecteurs(axe_z_wrist)

    axe_y_wrist = pos_marker[:, midmetac3_index, :].T - mid_ul_rad
    axe_y_wrist = normalise_vecteurs(axe_y_wrist)

    axe_x_wrist = np.cross(axe_y_wrist, axe_z_wrist)
    axe_x_wrist = normalise_vecteurs(axe_x_wrist)
    axe_x_wrist = [-i for i in axe_x_wrist]

    axe_z_wrist = np.cross(axe_x_wrist, axe_y_wrist)
    axe_z_wrist = normalise_vecteurs(axe_z_wrist)
    axe_z_wrist = [-i for i in axe_z_wrist]

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_x_wrist, axe_y_wrist, axe_z_wrist)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_wrist, axe_y_wrist, axe_x_wrist)
        ]
    )

    return matrices_rotation, mid_ul_rad


def get_orientation_head(pos_marker, marker_name_list):

    glabelle_index = find_index("GLABELLE", marker_name_list)
    tempd_index = find_index("TEMPD", marker_name_list)
    tempg_index = find_index("TEMPG", marker_name_list)
    c7_index = find_index("C7", marker_name_list)
    zygd_index = find_index("ZYGD", marker_name_list)
    zygg_index = find_index("ZYGG", marker_name_list)

    jc_head = (
        (pos_marker[:, c7_index, :]).T + (pos_marker[:, glabelle_index, :]).T
    ) / 2
    mid_zyg = ((pos_marker[:, zygd_index, :]).T + (pos_marker[:, zygg_index, :]).T) / 2
    mid_temp = (
        (pos_marker[:, tempd_index, :]).T + (pos_marker[:, tempg_index, :]).T
    ) / 2

    axe_z_head = (pos_marker[:, tempd_index, :]).T - (pos_marker[:, tempg_index, :]).T
    axe_z_head = normalise_vecteurs(axe_z_head)

    axe_y_head = mid_zyg - mid_temp
    axe_y_head = normalise_vecteurs(axe_y_head)

    axe_x_head = np.cross(axe_y_head, axe_z_head)
    axe_x_head = normalise_vecteurs(axe_x_head)
    axe_x_head = [-i for i in axe_x_head]

    axe_y_head = np.cross(axe_x_head, axe_z_head)
    axe_y_head = normalise_vecteurs(axe_y_head)
    axe_y_head = [-i for i in axe_y_head]

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_x_head, axe_y_head, axe_z_head)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_head, axe_x_head, axe_y_head)
        ]
    )

    return matrices_rotation, jc_head


def get_orientation_shoulder(pos_marker, marker_name_list, is_right_side):

    if is_right_side:
        epicon_index = find_index("EPICOND", marker_name_list)
        epitro_index = find_index("EPITROD", marker_name_list)
        acrant_index = find_index("ACRANTD", marker_name_list)
        acrpost_index = find_index("ACRPOSTD", marker_name_list)

    else:
        epicon_index = find_index("EPICONG", marker_name_list)
        epitro_index = find_index("EPITROG", marker_name_list)
        acrant_index = find_index("ACRANTG", marker_name_list)
        acrpost_index = find_index("ACRPOSTG", marker_name_list)

    mid_acr = (
        (pos_marker[:, acrant_index, :]).T + (pos_marker[:, acrpost_index, :]).T
    ) / 2
    mid_epi = (
        (pos_marker[:, epicon_index, :]).T + (pos_marker[:, epitro_index, :]).T
    ) / 2

    axe_y_shoulder = mid_acr - mid_epi
    axe_y_shoulder = normalise_vecteurs(axe_y_shoulder)

    V1 = mid_acr - (pos_marker[:, epitro_index, :]).T
    V2 = mid_acr - (pos_marker[:, epicon_index, :]).T

    axe_x_shoulder = np.cross(V2, V1) if is_right_side else np.cross(V1, V2)
    axe_x_shoulder = normalise_vecteurs(axe_x_shoulder)

    axe_z_shoulder = np.cross(axe_x_shoulder, axe_y_shoulder)
    axe_z_shoulder = normalise_vecteurs(axe_z_shoulder)

    axe_x_shoulder = np.cross(axe_y_shoulder, axe_z_shoulder)
    axe_x_shoulder = normalise_vecteurs(axe_x_shoulder)

    # matrices_rotation = np.array(
    #     [
    #         np.column_stack([x, y, z])
    #         for x, y, z in zip(axe_x_shoulder, axe_y_shoulder, axe_z_shoulder)
    #     ]
    # )
    matrices_rotation = np.array(
        [
            np.column_stack([x, y, z])
            for x, y, z in zip(axe_z_shoulder, axe_x_shoulder, axe_y_shoulder)
        ]
    )

    return matrices_rotation, mid_acr


def get_all_matrice(file_path, interval, model):
    file_name = os.path.basename(file_path).split(".")[0]
    print(f"{file_name} is running")
    c = ezc3d.c3d(file_path)
    point_data = c["data"]["points"][:, :, interval[0] : interval[1]]
    n_markers = point_data.shape[1]
    nf_mocap = point_data.shape[2]
    f_mocap = c["parameters"]["POINT"]["RATE"]["value"][0]
    point_labels = c["parameters"]["POINT"]["LABELS"]
    # Extraire les noms de marqueurs utiles de 'point_labels'
    useful_labels = [
        label for label in point_labels["value"] if not label.startswith("*")
    ]

    sample_label = useful_labels[0]
    typical_dimensions = point_data[0][
        find_index(sample_label, point_labels["value"])
    ].shape[0]

    desired_order = [
        model.markerNames()[i].to_string() for i in range(model.nbMarkers())
    ]

    n_markers_desired = len(desired_order)
    reordered_point_data = np.full((4, n_markers_desired, typical_dimensions), np.nan)

    for i, marker in enumerate(desired_order):
        marker_found = False
        for label in useful_labels:
            if marker in label:  # Vérifie si marker est une sous-chaîne de label.
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

    # Ne prendre que les 3 premiere colonnes et divise par 1000
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
    # Assurez-vous que Q est un vecteur 1D
    Q_1d = Q.flatten() if Q.ndim > 1 else Q

    # Initialiser Qdot et Qddot en tant que vecteurs 1D
    Qdot_1d = np.zeros(Q_1d.shape)
    Qddot_1d = np.zeros(Q_1d.shape)

    # Créer initial_guess avec ces vecteurs 1D
    initial_guess = (Q_1d, Qdot_1d, Qddot_1d)

    q_recons, qdot_recons, pos_recons = recons_kalman_with_marker(
        nf_mocap, n_markers_reordered, markers, model, initial_guess
    )

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
            matrices_rotation_thorax,
            matrices_rotation_head,
            matrices_rotation_shoulder_right,
            matrices_rotation_elbow_right,
            matrices_rotation_wrist_right,
            matrices_rotation_shoulder_left,
            matrices_rotation_elbow_left,
            matrices_rotation_wrist_left,
            matrices_rotation_hip_right,
            matrices_rotation_knee_right,
            matrices_rotation_ankle_right,
            matrices_rotation_hip_left,
            matrices_rotation_knee_left,
            matrices_rotation_ankle_left,
        ],
        axis=0,
    )

    articular_joint_center = np.stack(
        [
            pelvic_origin,
            manu,
            head_joint_center,
            mid_acr_right,
            mid_epi_right,
            mid_ul_rad_right,
            mid_acr_left,
            mid_epi_left,
            mid_ul_rad_left,
            hip_right_joint_center,
            mid_cond_right,
            mid_mal_right,
            hip_left_joint_center,
            mid_cond_left,
            mid_mal_left,
        ]
    )

    return rot_mat, articular_joint_center, pos_recons


def convert_to_local_frame(P1, R1, P2, R2):
    """
    - P1: Position of the first point in the global frame
    - R1: Rotation matrix of the first point in the global frame
    - P2: Position of the second point in the global frame
    - R2: Rotation matrix of the second point in the global frame

    - P2_prime: Position of the second point in the frame of the first point.
    - R2_prime: Rotation matrix of the second point relative to the first point.
    """

    P2_prime = P2 - P1
    P2_prime = R1.T @ P2_prime

    R2_prime = R1.T @ R2

    return P2_prime, R2_prime


def convert_marker_to_local_frame(P1, R1, P2):

    P2_prime = P2 - P1
    P2_prime = R1.T @ P2_prime

    return P2_prime


def average_rotation_matrix(matrices):
    # Calcul de la moyenne arithmétique des matrices
    mean_matrix = np.mean(matrices, axis=0)

    # Application de la SVD pour garantir l'orthogonalité
    U, _, VT = svd(mean_matrix)
    return np.dot(U, VT)


def calculer_rotation_et_angle(i_segment, quat_array, z_rotation):
    # Normalisation et conversion des quaternions en matrices de rotation
    quat_normalized = quat_array[i_segment * 4: (i_segment + 1) * 4] / np.linalg.norm(
        quat_array[i_segment * 4: (i_segment + 1) * 4]
    )
    quat = biorbd.Quaternion(quat_normalized[0], quat_normalized[1], quat_normalized[2], quat_normalized[3])
    rot_mat_segment = biorbd.Quaternion.toMatrix(quat).to_array()
    return z_rotation @ rot_mat_segment


def calculer_rotation_relative(rot_mat_parent, rot_mat_child):
    return np.linalg.inv(rot_mat_parent) @ rot_mat_child
