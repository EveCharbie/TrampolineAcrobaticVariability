import pickle
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from TrampolineAcrobaticVariability.Function.Function_stat import safe_interpolate

file_path = "/home/lim/Documents/StageMathieu/Tab_result3/sd_pelvis_and_gaze_orientation.mat"
data_loaded = scipy.io.loadmat(file_path)
mean_SD_pelvis_all_subjects_acrobatics = data_loaded["mean_SD_pelvis_all_subjects_acrobatics"]
movement_to_analyse = data_loaded["movement_to_analyse"]
movement_to_analyse = np.char.strip(movement_to_analyse)

wall_index_all_subjects_acrobatics = data_loaded["wall_index_all_subjects_acrobatics"]
liste_name = data_loaded["liste_name"]
gaze_position_temporal_evolution_projected_all_subject_acrobatics = data_loaded["gaze_position_temporal_evolution_projected_all_subject_acrobatics"]
list_name_for_movement = data_loaded["list_name_for_movement"]

X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])

name_to_color = {
    'GuSe': '#1f77b4',
    'JaSh': '#ff7f0e',
    'JeCa': '#2ca02c',
    'AnBe': '#d62728',
    'AnSt': '#9467bd',
    'SaBe': '#8c564b',
    'JoBu': '#e377c2',
    'JaNo': '#7f7f7f',
    'SaMi': '#bcbd22',
    'AlLe': '#17becf',
    'MaBo': '#aec7e8',
    'SoMe': '#ffbb78',
    'JeCh': '#98df8a',
    'LiDu': '#ff9896',
    'LeJa': '#c5b0d5',
    'ArMa': '#c49c94',
    'AlAd': '#dbdb8d'
}

name_to_linestyle = {
    'GuSe': '--',
    'JaSh': ':',
    'JeCa': '--',
    'AnBe': ':',
    'AnSt': ':',
    'SaBe': '--',
    'JoBu': ':',
    'JaNo': ':',
    'SaMi': '--',
    'AlLe': '--',
    'MaBo': '--',
    'SoMe': '--',
    'JeCh': '--',
    'LiDu': ':',
    'LeJa': ':',
    'ArMa': ':',
    'AlAd': '--'
}

anonyme_name = {
    'GuSe': '1',
    'JaSh': '2',
    'JeCa': '3',
    'AnBe': '4',
    'AnSt': '5',
    'SaBe': '6',
    'JoBu': '7',
    'JaNo': '8',
    'SaMi': '9',
    'AlLe': '10',
    'MaBo': '11',
    'SoMe': '12',
    'JeCh': '13',
    'LiDu': '14',
    'LeJa': '15',
    'ArMa': '16',
    'AlAd': '17'
}

full_name_acrobatics = {
    '4-': '4-/',
    '4-o':  '4-o',
    '8--o': '8--o',
    '8-1<': '8-1<',
    '8-1o': '8-1o',
    '41': '41/',
    '811<': '811<',
    '41o': '41o',
    '8-3<': '8-3<',
    '42': '42/',
    '822': '822/',
    '831<': '831<',
    '43': '43/',

}

num_points = 100

time_spent_looking = {
    "gymnasium_floor": {mvt: {name_subject: [] for name_subject in list_name_for_movement[0][idx_mvt]}
                        for idx_mvt, mvt in enumerate(movement_to_analyse)},
    "trampoline_bed": {mvt: {name_subject: [] for name_subject in list_name_for_movement[0][idx_mvt]}
                       for idx_mvt, mvt in enumerate(movement_to_analyse)}}
mean_time_spent_looking = {
    "gymnasium_floor": {mvt: {name_subject: 0 for name_subject in list_name_for_movement[0][idx_mvt]}
                        for idx_mvt, mvt in enumerate(movement_to_analyse)},
    "trampoline_bed": {mvt: {name_subject: 0 for name_subject in list_name_for_movement[0][idx_mvt]}
                        for idx_mvt, mvt in enumerate(movement_to_analyse)}}
std_time_spent_looking = {
    "gymnasium_floor": {mvt: {name_subject: 0 for name_subject in list_name_for_movement[0][idx_mvt]}
                        for idx_mvt, mvt in enumerate(movement_to_analyse)},
    "trampoline_bed": {mvt: {name_subject: 0 for name_subject in list_name_for_movement[0][idx_mvt]}
                       for idx_mvt, mvt in enumerate(movement_to_analyse)}}
min_time_gymnasium = ['none', 100]
max_time_gymnasium = ['none', 0]
time_spent_looking_gymnasium_array = []
min_time_bed = ['none', 100]
max_time_bed = ['none', 0]
time_spent_looking_bed_array = []
for idx_mvt, mvt in enumerate(movement_to_analyse):
    name_acro = full_name_acrobatics[mvt]

    up_subject = 1.5
    max_value = 60

    fig, ax = plt.subplots(figsize=(280 / 90, 396 / 80))
    initial_ticks = np.arange(0, 60, 10)
    current_ticks = list(initial_ticks)
    current_labels = [f"{tick:.0f}" for tick in initial_ticks]

    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):
        color = name_to_color[name_subject]
        line_style = name_to_linestyle[name_subject]
        anonyme = anonyme_name[name_subject]
        up_line = max_value + up_subject
        trial_count = 0
        for idx_trials in range(
                len(gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0])):
            gaze_position = \
                gaze_position_temporal_evolution_projected_all_subject_acrobatics[0][idx_mvt][0][idx_subject][0][idx_trials]

            data_ground = np.zeros(len(gaze_position), dtype=int)
            tolerance = 1e-6

            for idx_ligne, ligne in enumerate(gaze_position):
                if abs((ligne[2]) <= tolerance):  # Ground
                    data_ground[idx_ligne] = 0
                else:
                    data_ground[idx_ligne] = 1
            data_ground_pd = None
            data_ground_pd = pd.DataFrame(data_ground[:])

            data_mat = np.zeros(len(gaze_position), dtype=int)

            for idx_ligne, ligne in enumerate(gaze_position):
                if (X[0][0] <= ligne[0] <= X[0][1] and Y[:, 1][0] <= ligne[1] <= Y[:, 1][1] and abs(ligne[2]) <= tolerance):  # Trampo
                    data_mat[idx_ligne] = 0
                else:
                    data_mat[idx_ligne] = 1
            data_mat_pd = None
            data_mat_pd = pd.DataFrame(data_mat[:])

            pd.set_option('display.max_rows', None)

            data_norm_ground = data_ground_pd.apply(lambda x: safe_interpolate(x, num_points))
            data_norm_mat = data_mat_pd.apply(lambda x: safe_interpolate(x, num_points))

            y_line_position = up_line
            y_values_ground = np.full(len(data_norm_ground[0]), np.nan)
            y_values_ground[data_norm_ground[0] == 0] = y_line_position
            ax.plot(y_values_ground, '-', color=color, alpha=0.1, label='Presence of Zero' if idx_trials == 0 else "")

            y_values_mat = np.full(len(data_norm_mat[0]), np.nan)
            y_values_mat[data_norm_mat[0] == 0] = y_line_position
            ax.plot(y_values_mat, '-', color=color, alpha=1, label='Presence of Zero' if idx_trials == 0 else "")

            tick_with_offset = max_value + up_subject
            if max_value + up_subject not in current_ticks:
                current_ticks.append(tick_with_offset)
                current_labels.append(str(anonyme))
            current_ticks.sort()
            special_index = current_ticks.index(tick_with_offset)
            current_labels[special_index] = str(anonyme)

            up_line += 0.04
            trial_count += 0.08

            gym_data = np.where(data_ground == 0)[0].shape[0] / np.shape(data_ground)[0]
            bed_data = np.where(data_mat == 0)[0].shape[0] / np.shape(data_mat)[0]
            
            time_spent_looking["gymnasium_floor"][mvt][name_subject].append(gym_data)
            time_spent_looking_gymnasium_array += [gym_data]
            if np.min(gym_data) < min_time_gymnasium[1]:
                min_time_gymnasium[1] = np.min(gym_data)
                min_time_gymnasium[0] = mvt
            if np.max(gym_data) > 0.99999:
                print(mvt)
            if np.max(gym_data) > max_time_gymnasium[1]:
                max_time_gymnasium[1] = np.max(gym_data)
                max_time_gymnasium[0] = mvt

            time_spent_looking["trampoline_bed"][mvt][name_subject].append(bed_data)
            time_spent_looking_bed_array += [bed_data]
            if np.min(bed_data) < min_time_bed[1]:
                min_time_bed[1] = np.min(bed_data)
                min_time_bed[0] = mvt
            if np.max(bed_data) > 0.99999:
                print('-------------------', mvt)
            if np.max(bed_data) > max_time_bed[1]:
                max_time_bed[1] = np.max(bed_data)
                max_time_bed[0] = mvt
                
        plt.plot(np.degrees(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt][idx_subject]), label=f'Subject {idx_subject + 1}',
                 color=color, linestyle=line_style)

        array_this_time_bed = np.array(time_spent_looking["trampoline_bed"][mvt][name_subject])
        mean_time_spent_looking["trampoline_bed"][mvt][name_subject] = np.mean(array_this_time_bed)
        std_time_spent_looking["trampoline_bed"][mvt][name_subject] = np.std(array_this_time_bed)
        array_this_time_gymnasium = np.array(time_spent_looking["gymnasium_floor"][mvt][name_subject])
        mean_time_spent_looking["gymnasium_floor"][mvt][name_subject] = np.mean(array_this_time_gymnasium)
        std_time_spent_looking["gymnasium_floor"][mvt][name_subject] = np.std(array_this_time_gymnasium)
        plt.text(101, up_line-1, f"{round(mean_time_spent_looking["gymnasium_floor"][mvt][name_subject] * 100)}%", fontsize=6)

        up_subject += 2

    mean_array = np.mean(mean_SD_pelvis_all_subjects_acrobatics[0][idx_mvt], axis=0)
    plt.plot(np.degrees(mean_array), color="black", linestyle= '-', linewidth=2)
    if mvt in ["8--o", "831<"]:
        mean_array[:50] = 0
    horz_max = np.argmax(mean_array)
    plt.plot(np.array([horz_max, horz_max]), np.array([0, 1000]), linewidth=0.5, color="black")

    ax.set_yticks(current_ticks)
    ax.set_yticklabels(current_labels, fontweight='normal')

    yticks = ax.get_yticks()
    yticklabels = ax.get_yticklabels()

    for tick, label in zip(yticks, yticklabels):
        try:
            if tick < max_value:
                label.set_fontsize(9)
            else:
                label.set_fontsize(7)
        except ValueError:
            continue

    ax.set_title(f"Horizontal Line Indicating Presence of Zeros {mvt}")
    ax.set_xlim(0, len(data_norm_ground[0]))
    ax.set_ylim(0, 95)
    ax.tick_params(axis='x', labelsize=9)

    line1, = plt.plot([], [], color='black', label='Gaze on trampoline')
    line2, = plt.plot([], [], color='black', linestyle='--', label='SDtotal on pelvic rotation')

    plt.title(f'{name_acro}', fontsize=11)
    plt.subplots_adjust(left=0.102, right=0.9, top=0.945, bottom=0.047)
    plt.savefig(f"/home/lim/Documents/StageMathieu/Gaze_ground/{mvt}_gaze.png", dpi=1000)
    # plt.show()

# Legend
fig, ax = plt.subplots()
line1, = ax.plot([], [], color='#9467bd', label='Gaze on the \ntrampoline bed')
line2, = ax.plot([], [], color='#9467bd', alpha=0.2, label='Gaze on the \ngymnasium floor')
line3, = ax.plot([], [], color='#9467bd', linestyle='--', label='SDtotal on pelvis \norientation')
line4, = ax.plot([], [], color='black', linestyle='-', label='Mean SDtotal across \nacrobatics on \npelvis orientation')

figlegend = plt.figure(figsize=(4, 1.5), facecolor='white')
plt.figlegend(handles=[line1, line2, line3, line4], loc='center', fontsize='small', handlelength=4)

# plt.show()

figlegend.savefig('/home/lim/Documents/StageMathieu/Gaze_ground/legend.png', bbox_inches='tight', pad_inches=0, dpi=1000)

print("\n\nGymnasium floor : \nMin : ", min_time_gymnasium, "\nMax : ", max_time_gymnasium, "\nMean : ", np.mean(time_spent_looking_gymnasium_array))
print("\n\nTrampoline bed : \nMin : ", min_time_bed, "\nMax : ", max_time_bed, "\nMean : ", np.mean(time_spent_looking_bed_array))

with open("/home/lim/Documents/StageMathieu/Tab_result3/time_spent_looking.pkl", 'wb') as f:
    pickle.dump([time_spent_looking, mean_time_spent_looking, std_time_spent_looking], f)