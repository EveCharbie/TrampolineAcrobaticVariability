"""
The goal of this program is to convert the joint angles from xsens to biorbd.
"""

import numpy as np
import pickle
from IPython import embed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
<<<<<<< HEAD:get_kinematics_and_gaze_from_xsens.py
import biorbd
import bioviz
import scipy

=======

# import biorbd
# import bioviz
import scipy


>>>>>>> main:Draw/get_kinematics_and_gaze_from_xsens.py
def get_q(Xsens_orientation_per_move, move_orientation):
    """
    This function returns de generalized coordinates in the sequence XYZ (biorbd) from the quaternion of the orientation
    of the Xsens segments.
    The translation is left empty as it has to be computed otherwise.
    I am not sure if I would use this for kinematics analysis, but for visualisation it is not that bad.
    """

    parent_idx_list = {"Pelvis": None,  # 0
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
    Q = np.zeros((23*3, nb_frames))
    rotation_matrices = np.zeros((23, nb_frames, 3, 3))
    for i_segment, key in enumerate(parent_idx_list):
        for i_frame in range(nb_frames):
            Quat_normalized = Xsens_orientation_per_move[i_frame, i_segment*4: (i_segment+1)*4] / np.linalg.norm(
                Xsens_orientation_per_move[i_frame, i_segment*4: (i_segment+1)*4]
            )
            Quat = biorbd.Quaternion(Quat_normalized[0],
                                     Quat_normalized[1],
                                     Quat_normalized[2],
                                     Quat_normalized[3])

            RotMat_current = biorbd.Quaternion.toMatrix(Quat).to_array()
            if move_orientation == 1:
                z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-np.pi/2]), 'z').to_array()
            else:
                z_rotation = biorbd.Rotation.fromEulerAngles(np.array([-3*np.pi/2]), 'z').to_array()
            RotMat_current = z_rotation @ RotMat_current

            if parent_idx_list[key] is None:
                RotMat = np.eye(3)
            else:
                RotMat = rotation_matrices[parent_idx_list[key][0], i_frame, :, :]

            RotMat_between = np.linalg.inv(RotMat) @ RotMat_current
            RotMat_between = biorbd.Rotation(RotMat_between[0, 0], RotMat_between[0, 1], RotMat_between[0, 2],
                            RotMat_between[1, 0], RotMat_between[1, 1], RotMat_between[1, 2],
                            RotMat_between[2, 0], RotMat_between[2, 1], RotMat_between[2, 2])
            Q[i_segment*3:(i_segment+1)*3, i_frame] = biorbd.Rotation.toEulerAngles(RotMat_between, 'xyz').to_array()
            rotation_matrices[i_segment, i_frame, :, :] = RotMat_current

    return Q


joint_labels = [
    "Pelvs_x",
    "Pelvs_y",
    "Pelvs_z",

    "jL5S1_x",  # 0
    "jL5S1_y",  # 1
    "jL5S1_z",  # 2

    "jL4L3_x",  # 3
    "jL4L3_y",  # 4
    "jL4L3_z",  # 5

    "jL1T12_x",  # 6
    "jL1T12_y",  # 7
    "jL1T12_z",  # 8
    "jT9T8_x",  # 9
    "jT9T8_y",  # 10
    "jT9T8_z",  # 11

    "jT1C7_x",  # 12
    "jT1C7_y",  # 13
    "jT1C7_z",  # 14

    "jC1Head_x",  # 15
    "jC1Head_y",  # 16
    "jC1Head_z",  # 17

    "jRightT4Shoulder…",  # 18
    "jRightT4Shoulder…",  # 19
    "jRightT4Shoulder…",  # 20
    "jRightShoulder_x",  # 21
    "jRightShoulder_y",  # 22
    "jRightShoulder_z",  # 23
    "jRightElbow_x",  # 24
    "jRightElbow_y",  # 25
    "jRightElbow_z",  # 26
    "jRightWrist_x",  # 27
    "jRightWrist_y",  # 28
    "jRightWrist_z",  # 29

    "jLeftT4Shoulder_x",  # 30
    "jLeftT4Shoulder_y",  # 31
    "jLeftT4Shoulder_z",  # 32
    "jLeftShoulder_x",  # 33
    "jLeftShoulder_y",  # 34
    "jLeftShoulder_z",  # 35
    "jLeftElbow_x",  # 36
    "jLeftElbow_y",  # 37
    "jLeftElbow_z",  # 38
    "jLeftWrist_x",  # 39
    "jLeftWrist_y",  # 40
    "jLeftWrist_z",  # 41

    "jRightHip_x",  # 42
    "jRightHip_y",  # 43
    "jRightHip_z",  # 44
    "jRightKnee_x",  # 45
    "jRightKnee_y",  # 46
    "jRightKnee_z",  # 47
    "jRightAnkle_x",  # 48
    "jRightAnkle_y",  # 49
    "jRightAnkle_z",  # 50
    "jRightBallFoot_x",  # 51
    "jRightBallFoot_y",  # 52
    "jRightBallFoot_z",  # 53

    "jLeftHip_x",  # 54
    "jLeftHip_y",  # 55
    "jLeftHip_z",  # 56
    "jLeftKnee_x",  # 57
    "jLeftKnee_y",  # 58
    "jLeftKnee_z",  # 59
    "jLeftAnkle_x",  # 60
    "jLeftAnkle_y",  # 61
    "jLeftAnkle_z",  # 62
    "jLeftBallFoot_x",  # 63
    "jLeftBallFoot_y",  # 64
    "jLeftBallFoot_z",  # 65
]  # 66


### ------------------------ Code beginnig ------------------------ ###

GENRATE_DATA_FRAME_FLAG = True
name_results = ""
save_path = "/home/charbie/Documents/Programmation/TrampolineAcrobaticVariability/XsensReconstructions/"


move_list = ["4-", "41", "42", "43"]

if os.path.exists("/home/user"):
    home_path = "/home/user"
elif os.path.exists("/home/charbie"):
    home_path = "/home/charbie"

if name_results:
    results_path = f"{home_path}/disk/Eye-tracking/Results_{name_results}"
    plot_path = home_path + f"/disk/Eye-tracking/plots_{name_results}"
else:
    results_path = f"{home_path}/disk/Eye-tracking/Results"
    plot_path = home_path + f"/disk/Eye-tracking/plots"

# This section is only to answer questions from the reviewers
pelvis_orientations = {}
pelvis_orientations_interpolated = {}
time_vector = {}
if GENRATE_DATA_FRAME_FLAG:
    for folder_subject in os.listdir(results_path):
        # biorbd_model_path = f"models/{folder_subject}_Xsens_Model_rotated.bioMod"
        biorbd_model_path = "/home/charbie/Documents/Programmation/VisionOCP/models/SoMe_Xsens_Model_rotated_without_cone.bioMod"
        pelvis_orientations[folder_subject] = {}
<<<<<<< HEAD:get_kinematics_and_gaze_from_xsens.py
        pelvis_orientations_interpolated[folder_subject] = {}
        time_vector[folder_subject] = {}
        for folder_move in os.listdir(results_path + '/' + folder_subject):
            if folder_move in move_list:
                pelvis_orientations[folder_subject][folder_move] = []
                pelvis_orientations_interpolated[folder_subject][folder_move] = []
                time_vector[folder_subject][folder_move] = []
                for file in os.listdir(results_path + '/' + folder_subject + '/' + folder_move):
=======
        for folder_move in os.listdir(results_path + "/" + folder_subject):
            if folder_move in move_list:
                pelvis_orientations[folder_subject][folder_move] = []
                for file in os.listdir(results_path + "/" + folder_subject + "/" + folder_move):
>>>>>>> main:Draw/get_kinematics_and_gaze_from_xsens.py
                    if len(file) > 23:
                        if file[-23:] == "eyetracking_metrics.pkl":

                            path = results_path + "/" + folder_subject + "/" + folder_move + "/"
                            move_filename = path + file
                            with open(move_filename, "rb") as f:
                                eye_tracking_metrics = pickle.load(f)

                                expertise = eye_tracking_metrics["subject_expertise"]
                                subject_name = eye_tracking_metrics["subject_name"]
                                move_orientation = eye_tracking_metrics["move_orientation"]

                                acrobatics = folder_move

                                gaze_position_temporal_evolution_projected = eye_tracking_metrics[
                                    "gaze_position_temporal_evolution_projected"
                                ]
                                gaze_position_temporal_evolution_projected_facing_front_wall = eye_tracking_metrics[
                                    "gaze_position_temporal_evolution_projected_facing_front_wall"
                                ]
                                wall_index = eye_tracking_metrics["wall_index"]
                                wall_index_facing_front_wall = eye_tracking_metrics["wall_index_facing_front_wall"]
                                fixation_index = eye_tracking_metrics["fixation_index"]
                                anticipatory_index = eye_tracking_metrics["anticipatory_index"]
                                compensatory_index = eye_tracking_metrics["compensatory_index"]
                                spotting_index = eye_tracking_metrics["spotting_index"]
                                movement_detection_index = eye_tracking_metrics["movement_detection_index"]
                                blinks_index = eye_tracking_metrics["blinks_index"]

                                fixation_positions = eye_tracking_metrics["fixation_positions"]

                                Xsens_head_position_calculated = eye_tracking_metrics["Xsens_head_position_calculated"]
                                eye_position = eye_tracking_metrics["eye_position"]
                                gaze_orientation = eye_tracking_metrics["gaze_orientation"]
                                EulAngles_head_global = eye_tracking_metrics["EulAngles_head_global"]
                                EulAngles_neck = eye_tracking_metrics["EulAngles_neck"]
                                eye_angles = eye_tracking_metrics["eye_angles"]
                                Xsens_orthogonal_thorax_position = eye_tracking_metrics[
                                    "Xsens_orthogonal_thorax_position"
                                ]
                                Xsens_orthogonal_head_position = eye_tracking_metrics["Xsens_orthogonal_head_position"]
                                Xsens_position_no_level_CoM_corrected_rotated_per_move = eye_tracking_metrics[
                                    "Xsens_position_no_level_CoM_corrected_rotated_per_move"
                                ]
                                Xsens_jointAngle_per_move = eye_tracking_metrics["Xsens_jointAngle_per_move"]
                                Xsens_orientation_per_move = eye_tracking_metrics["Xsens_orientation_per_move"]
                                Xsens_CoM_per_move = eye_tracking_metrics["Xsens_CoM_per_move"]
                                time_vector_pupil_per_move = eye_tracking_metrics["time_vector_pupil_per_move"]

                            ### ------------- Computations begin here ------------- ###
                            # model = biorbd.Model(biorbd_model_path)
                            num_dofs = 69 + 3 + 2  # model.nbQ()

                            eye_angles_without_nans = np.zeros(eye_angles.shape)
                            eye_angles_without_nans[:, :] = eye_angles[:, :]
                            blink_index = np.isnan(eye_angles[0, :]).astype(int)
                            end_of_blinks = np.where(blink_index[1:] - blink_index[:-1] == -1)[0]
                            start_of_blinks = np.where(blink_index[1:] - blink_index[:-1] == 1)[0]

                            if blink_index[0] == 1:
                                start_of_blinks = np.hstack((-1, start_of_blinks))

                            for i in range(len(start_of_blinks)):
                                if len(end_of_blinks) == i:
                                    eye_angles_without_nans[0, start_of_blinks[i] + 1:] = eye_angles_without_nans[0,
                                                                                          start_of_blinks[i]]
                                    eye_angles_without_nans[1, start_of_blinks[i] + 1:] = eye_angles_without_nans[1,
                                                                                          start_of_blinks[i]]
                                else:
                                    eye_angles_without_nans[:, start_of_blinks[i]+1:end_of_blinks[i]+1] = np.linspace(eye_angles_without_nans[:, start_of_blinks[i]], eye_angles_without_nans[:, end_of_blinks[i]+1], end_of_blinks[i]-start_of_blinks[i]).T

                            DoFs = np.zeros((num_dofs, len(Xsens_jointAngle_per_move)))
                            DoFs[3:-2, :] = get_q(Xsens_orientation_per_move, move_orientation)
                            DoFs[-2:, :] = eye_angles_without_nans
                            for i in range(DoFs.shape[0]):
                                DoFs[i, :] = np.unwrap(DoFs[i, :])

                            time_vector_pupil_per_move = time_vector_pupil_per_move - time_vector_pupil_per_move[0]
                            duration = time_vector_pupil_per_move[-1]
                            vz_init = 9.81 * duration / 2

<<<<<<< HEAD:get_kinematics_and_gaze_from_xsens.py
                            trans = np.zeros((3, len(Xsens_jointAngle_per_move)))
                            trans[2, :] = vz_init * time_vector_pupil_per_move - 0.5 * 9.81 * time_vector_pupil_per_move ** 2

                            model = biorbd.Model(biorbd_model_path)
                            for i in range(DoFs.shape[1]):
                                CoM = model.CoM(DoFs[:, i]).to_array()
                                trans[:, i] = trans[:, i] - CoM
                            DoFs[:3, :] = trans

                            # b = bioviz.Viz(biorbd_model_path,
                            #                mesh_opacity=0.8,
                            #                show_global_center_of_mass=False,
                            #                show_gravity_vector=False,
                            #                show_segments_center_of_mass=False,
                            #                show_global_ref_frame=False,
                            #                show_local_ref_frame=False,
                            #                experimental_markers_color=(1, 1, 1),
                            #                background_color=(1.0, 1.0, 1.0),
                            #                )
                            # b.set_camera_zoom(0.25)
                            # b.set_camera_focus_point(0, 0, 2.5)
                            # b.maximize()
                            # b.update()
                            # b.load_movement(DoFs)
                            #
                            # b.set_camera_zoom(0.25)
                            # b.set_camera_focus_point(0, 0, 2.5)
                            # b.maximize()
                            # b.update()
                            # b.exec()

                            pelvis_interpolated = np.zeros((3, 100))
                            for i in range(3):
                                interp_fun = scipy.interpolate.interp1d(time_vector_pupil_per_move, DoFs[3 + i, :], kind="linear")
                                pelvis_interpolated[i, :] = interp_fun(np.linspace(0, duration, 100))
                            pelvis_orientations[folder_subject][folder_move].append(DoFs[3:6, :])
                            pelvis_interpolated[2, :] += np.pi / 2
                            if pelvis_interpolated[2, -1] < 0.2:
                                pelvis_interpolated[2, :] *= -1
                            pelvis_orientations_interpolated[folder_subject][folder_move].append(pelvis_interpolated/2/np.pi)
                            time_vector[folder_subject][folder_move].append(time_vector_pupil_per_move/duration)
=======
                pelvis_orientations[folder_subject][folder_move].append(DoFs[3:6, :])
>>>>>>> main:Draw/get_kinematics_and_gaze_from_xsens.py


elite_names = ["AlAd", "GuSe", "JeCa", "JeCh", "MaBo", "SaBe", "SaMi", "SoMe"]
subelite_names = ["AlLe", "AnBe", "AnSt", "ArMa", "JaNo", "JaSh", "JoBu", "LeJa", "LiDu"]
colors_subelites = [cm.get_cmap("plasma")(k) for k in np.linspace(0, 0.4, len(subelite_names))]
colors_elites = [cm.get_cmap("plasma")(k) for k in np.linspace(0.6, 1, len(elite_names))]

fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for subject in pelvis_orientations:
    for i_move, move in enumerate(move_list):
        if move in pelvis_orientations[subject]:
            for i_trial in range(len(pelvis_orientations[subject][move])):
                if subject in elite_names:
                    color = colors_elites[elite_names.index(subject)]
                    axs[0, i_move].plot(time_vector[subject][move][i_trial],
                                        pelvis_orientations[subject][move][i_trial][0, :], color=color)
                    axs[2, i_move].plot(time_vector[subject][move][i_trial],
                                        pelvis_orientations[subject][move][i_trial][2, :] + np.pi / 2, color=color)

                elif subject in subelite_names:
                    color = colors_subelites[subelite_names.index(subject)]
                    axs[1, i_move].plot(time_vector[subject][move][i_trial],
                                        pelvis_orientations[subject][move][i_trial][0, :], color=color)
                    axs[3, i_move].plot(time_vector[subject][move][i_trial],
                                        pelvis_orientations[subject][move][i_trial][2, :] + np.pi / 2, color=color)

                else:
                    raise RuntimeError(f"Subject {subject} not found")

for i_move, move in enumerate(move_list):
    axs[0, i_move].set_title(move)

for i_athlete, athlete in enumerate(elite_names):
    axs[0, 3].plot([], [], color=colors_elites[i_athlete], label=athlete)
for i_athlete, athlete in enumerate(subelite_names):
    axs[1, 3].plot([], [], color=colors_subelites[i_athlete], label=athlete)
axs[0, 3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
axs[1, 3].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


axs[0, 0].set_ylabel("Somersault")
axs[1, 0].set_ylabel("Somersault")
axs[2, 0].set_ylabel("Twist")
axs[3, 0].set_ylabel("Twist")

for i in range(4):
    axs[i, 0].set_ylim(-6, 6)

plt.tight_layout()
plt.savefig("SomersaultsTwist.png", dpi=300)
plt.show()
<<<<<<< HEAD:get_kinematics_and_gaze_from_xsens.py


fig, axs = plt.subplots(2, 2, figsize=(6, 6))
axs = axs.ravel()
mean_elites_pelvis_orientations = np.zeros((4, len(pelvis_orientations_interpolated.keys()), 100, 3))
mean_subelites_pelvis_orientations = np.zeros((4, len(pelvis_orientations_interpolated.keys()), 100, 3))
mean_elites_pelvis_orientations[:, :, :, :] = np.nan
mean_subelites_pelvis_orientations[:, :, :, :] = np.nan
for i_subject, subject in enumerate(pelvis_orientations_interpolated):
    for i_move, move in enumerate(pelvis_orientations_interpolated[subject]):
        for i in range(3):
            if subject in elite_names:
                if len(pelvis_orientations_interpolated[subject][move]) != 0:
                    mean_elites_pelvis_orientations[move_list.index(move), i_subject, :, i] = np.nanmean(
                        np.array(pelvis_orientations_interpolated[subject][move])[:, i, :], axis=0)
            else:
                if len(pelvis_orientations_interpolated[subject][move]) != 0:
                    mean_subelites_pelvis_orientations[move_list.index(move), i_subject, :, i] = np.nanmean(
                        np.array(pelvis_orientations_interpolated[subject][move])[:, i, :], axis=0)

for i_move, move in enumerate(move_list):
    mean_somersault_elites = np.nanmean(mean_elites_pelvis_orientations[i_move, :, :, 0], axis=0)
    mean_twist_elites = np.nanmean(mean_elites_pelvis_orientations[i_move, :, :, 2], axis=0)
    std_somersault_elites = np.nanstd(mean_elites_pelvis_orientations[i_move, :, :, 0], axis=0)
    std_twist_elites = np.nanstd(mean_elites_pelvis_orientations[i_move, :, :, 2], axis=0)
    if i_move == 0:
        axs[i_move].plot(mean_somersault_elites, mean_twist_elites, color=colors_elites[3], label="Elites")
    else:
        axs[i_move].plot(mean_somersault_elites, mean_twist_elites, color=colors_elites[3])
    axs[i_move].fill_between(mean_somersault_elites, mean_twist_elites - std_twist_elites,
                             mean_twist_elites + std_twist_elites, color=colors_elites[3], alpha=0.5)
    axs[i_move].plot(mean_somersault_elites[0], mean_twist_elites[0], color=colors_elites[3], marker='o')

    mean_somersault_subelites = np.nanmean(mean_subelites_pelvis_orientations[i_move, :, :, 0], axis=0)
    mean_twist_subelites = np.nanmean(mean_subelites_pelvis_orientations[i_move, :, :, 2], axis=0)
    std_somersault_subelites = np.nanstd(mean_subelites_pelvis_orientations[i_move, :, :, 0], axis=0)
    std_twist_subelites = np.nanstd(mean_subelites_pelvis_orientations[i_move, :, :, 2], axis=0)
    if i_move == 0:
        axs[i_move].plot(mean_somersault_subelites, mean_twist_subelites, color=colors_subelites[3], label="Subelites")
    else:
        axs[i_move].plot(mean_somersault_subelites, mean_twist_subelites, color=colors_subelites[3])
    axs[i_move].fill_between(mean_somersault_subelites, mean_twist_subelites - std_twist_subelites,
                             mean_twist_subelites + std_twist_subelites, color=colors_subelites[3], alpha=0.5)
    axs[i_move].plot(mean_somersault_subelites[0], mean_twist_subelites[0], color=colors_subelites[3], marker='o')


axs[0].set_title("0 twist", fontweight="bold")
axs[1].set_title("0.5 twist", fontweight="bold")
axs[2].set_title("1 twist", fontweight="bold")
axs[3].set_title("1.5 twists", fontweight="bold")

axs[0].legend(bbox_to_anchor=(1.2, 1.25), loc='upper center', borderaxespad=0., ncol=2)

for i in range(4):
    axs[i].set_xlabel("Somersault")
    axs[i].set_ylabel("Twist")
axs[0].set_ylim(-0.1, 0.5)

plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.savefig("SomersaultsTwist_angle-angle.png", dpi=300)
plt.show()
=======
>>>>>>> main:Draw/get_kinematics_and_gaze_from_xsens.py
