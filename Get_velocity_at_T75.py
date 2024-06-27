import scipy
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import glob
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    load_and_interpolate_for_point,
    find_index
)

nombre_lignes_minimum = 10
n_points = 100
next_index = 0
time_values = np.linspace(0, n_points-1, num=n_points)

home_path = "/DataTrampo/Xsens_pkl/"
movement_to_analyse = ['41', '42', '43', '41o', '8-1<', '8-1o', '8-3<', '811<', '822', '831<']
path75 = "/home/lim/Documents/StageMathieu/Tab_result3/"

liste_name = [name for name in os.listdir(home_path) if os.path.isdir(os.path.join(home_path, name))]

list_name_for_movement = []
all_velocities = []
for id_mvt, mvt_name in enumerate(movement_to_analyse):

    pattern_file = f"*_{mvt_name}_*_75.csv"
    file75 = glob.glob(os.path.join(path75, pattern_file))
    timestamp75 = pd.read_csv(file75[0])


    temp_liste_name = []
    for name in liste_name:
        home_path_subject = f"{home_path}{name}/Pos_JC/{mvt_name}"
        if not os.path.exists(home_path_subject):
            print(f"Subject {name} didn't realize {mvt_name}")
        else:
            temp_liste_name.append(name)

    list_name_for_movement.append(temp_liste_name)
    pelvis_X_velocity_by_subject = []
    pelvis_Y_velocity_by_subject = []
    pelvis_Z_velocity_by_subject = []
    pelvis_global_velocity_by_subject = []
    acrobatics_velocity_each_subject_T75 = []

    for id_name, name in enumerate(temp_liste_name):
        print(f"{name} {mvt_name} is running")
        home_path_subject = f"{home_path}{name}/Pos_JC/{mvt_name}"

        fichiers_mat_subject = []
        for root, dirs, files in os.walk(home_path_subject):
            for file in files:
                if file.endswith(".mat"):
                    full_path = os.path.join(root, file)
                    fichiers_mat_subject.append(full_path)

        data_subject = []
        subject_info_dict = {}
        gaze_position_temporal_evolution_projected_subject = []
        velocity_by_subject = []

        T75_by_name = timestamp75[name].dropna().mean().round()

        for file in fichiers_mat_subject:
            (data,
             subject_expertise,
             laterality,
             length_segment,
             wall_index,
             gaze_position_temporal_evolution_projected,
             total_duration) = load_and_interpolate_for_point(file, include_expertise_laterality_length=True)

            pelvis_data = data[['Pelvis_X', 'Pelvis_Y', 'Pelvis_Z']]
            pelvis_data_degrees = np.degrees(pelvis_data)

            pelvis_data_degrees['Pelvis_X'] = savgol_filter(pelvis_data_degrees['Pelvis_X'], window_length=11,
                                                            polyorder=2)
            pelvis_data_degrees['Pelvis_Y'] = savgol_filter(pelvis_data_degrees['Pelvis_Y'], window_length=11,
                                                            polyorder=2)
            pelvis_data_degrees['Pelvis_Z'] = savgol_filter(pelvis_data_degrees['Pelvis_Z'], window_length=11,
                                                            polyorder=2)

            time = np.arange(100)

            num_points = 100
            dt = total_duration / (num_points - 1)

            dPelvis_X = np.diff(pelvis_data_degrees['Pelvis_X']) / dt
            dPelvis_Y = np.diff(pelvis_data_degrees['Pelvis_Y']) / dt
            dPelvis_Z = np.diff(pelvis_data_degrees['Pelvis_Z']) / dt

            dPelvis_X = np.insert(dPelvis_X, 0, 0)
            dPelvis_Y = np.insert(dPelvis_Y, 0, 0)
            dPelvis_Z = np.insert(dPelvis_Z, 0, 0)

            global_velocity = np.sqrt(dPelvis_X ** 2 + dPelvis_Y ** 2 + dPelvis_Z ** 2)

            velocities = np.column_stack((dPelvis_X, dPelvis_Y, np.sqrt(dPelvis_Z**2), global_velocity))
            velocity_by_subject.append(velocities)

        subject_velocities = np.mean(np.array(velocity_by_subject), axis=0)

        pelvis_X_velocity_by_subject.append(subject_velocities[:, 0])
        pelvis_Y_velocity_by_subject.append(subject_velocities[:, 1])
        pelvis_Z_velocity_by_subject.append(subject_velocities[:, 2])
        pelvis_global_velocity_by_subject.append(subject_velocities[:, 3])

        subject_velocityT75 = subject_velocities[int(T75_by_name), 3]
        acrobatics_velocity_each_subject_T75.append(subject_velocityT75)

    acrobatics_pelvis_X_velocity = np.mean(np.array(pelvis_X_velocity_by_subject), axis=0)
    acrobatics_pelvis_Y_velocity = np.mean(np.array(pelvis_Y_velocity_by_subject), axis=0)
    acrobatics_pelvis_Z_velocity = np.mean(np.array(pelvis_Z_velocity_by_subject), axis=0)
    acrobatics_pelvis_global_velocity = np.mean(np.array(pelvis_global_velocity_by_subject), axis=0)

    std_pelvis_X_velocity = np.std(np.array(pelvis_X_velocity_by_subject), axis=0)
    std_pelvis_Y_velocity = np.std(np.array(pelvis_Y_velocity_by_subject), axis=0)
    std_pelvis_Z_velocity = np.std(np.array(pelvis_Z_velocity_by_subject), axis=0)
    std_pelvis_global_velocity = np.std(np.array(pelvis_global_velocity_by_subject), axis=0)

    time = time[1:]
    acrobatics_pelvis_X_velocity = acrobatics_pelvis_X_velocity[1:]
    acrobatics_pelvis_Y_velocity = acrobatics_pelvis_Y_velocity[1:]
    acrobatics_pelvis_Z_velocity = acrobatics_pelvis_Z_velocity[1:]
    acrobatics_pelvis_global_velocity = acrobatics_pelvis_global_velocity[1:]
    std_pelvis_X_velocity = std_pelvis_X_velocity[1:]
    std_pelvis_Y_velocity = std_pelvis_Y_velocity[1:]
    std_pelvis_Z_velocity = std_pelvis_Z_velocity[1:]
    std_pelvis_global_velocity = std_pelvis_global_velocity[1:]

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(time, acrobatics_pelvis_X_velocity, label='dPelvis_X')
    plt.fill_between(time, acrobatics_pelvis_X_velocity - std_pelvis_X_velocity,
                     acrobatics_pelvis_X_velocity + std_pelvis_X_velocity, color='gray', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (degrees/s)')
    plt.title('Mean Angular Velocity of Pelvis_X')
    plt.legend()

    plt.subplot(4, 1, 2)
    plt.plot(time, acrobatics_pelvis_Y_velocity, label='dPelvis_Y')
    plt.fill_between(time, acrobatics_pelvis_Y_velocity - std_pelvis_Y_velocity,
                     acrobatics_pelvis_Y_velocity + std_pelvis_Y_velocity, color='gray', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (degrees/s)')
    plt.title('Mean Angular Velocity of Pelvis_Y')
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(time, acrobatics_pelvis_Z_velocity, label='dPelvis_Z')
    plt.fill_between(time, acrobatics_pelvis_Z_velocity - std_pelvis_Z_velocity,
                     acrobatics_pelvis_Z_velocity + std_pelvis_Z_velocity, color='gray', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Angular Velocity (degrees/s)')
    plt.title('Mean Angular Velocity of Pelvis_Z')
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(time, acrobatics_pelvis_global_velocity, label='Global Velocity', color='m')
    plt.fill_between(time, acrobatics_pelvis_global_velocity - std_pelvis_global_velocity,
                     acrobatics_pelvis_global_velocity + std_pelvis_global_velocity, color='gray', alpha=0.5)
    plt.xlabel('Time')
    plt.ylabel('Global Velocity (degrees/s)')
    plt.title('Mean Global Angular Velocity')
    plt.legend()

    plt.tight_layout()
    # plt.show()
    plt.close()

    velocity_acrobatic_at_T75 = np.mean(acrobatics_velocity_each_subject_T75)
    print(f"{velocity_acrobatic_at_T75} for {mvt_name}")
    all_velocities.append(velocity_acrobatic_at_T75.round())

result_df = pd.DataFrame({
    'Velocity at T75': all_velocities,
    'Movement Name': movement_to_analyse
})

result_df_sorted = result_df.sort_values(by='Velocity at T75')
print(result_df_sorted)

