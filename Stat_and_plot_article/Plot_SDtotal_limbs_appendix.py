import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from TrampolineAcrobaticVariability.Function.Function_stat import safe_interpolate

file_path = "/home/lim/Documents/StageMathieu/Tab_result3/sd_pelvis_and_gaze_orientation.mat"
data_loaded = scipy.io.loadmat(file_path)
mean_members_all_subjects_acrobatics = data_loaded["members_data_all_subjects_acrobatics"]
movement_to_analyse = data_loaded["movement_to_analyse"]
movement_to_analyse = np.char.strip(movement_to_analyse)

liste_name = data_loaded["liste_name"]
list_name_for_movement = data_loaded["list_name_for_movement"]

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

members = ["Elbow R", "Wrist R", "Elbow L", "Wrist L", "Knee R", "Ankle R", "Knee L", "Ankle L"]

num_points = 100

for idx_mvt, mvt in enumerate(movement_to_analyse):
    name_acro = full_name_acrobatics[mvt]

    fig, axs = plt.subplots(8, 1, figsize=(280 / 90 + 1, 396 / 80))
    initial_ticks = np.arange(0, 60, 10)
    current_ticks = list(initial_ticks)
    current_labels = [f"{tick:.0f}" for tick in initial_ticks]

    for idx_subject, name_subject in enumerate(list_name_for_movement[0][idx_mvt]):
        color = name_to_color[name_subject]
        line_style = name_to_linestyle[name_subject]
        anonyme = anonyme_name[name_subject]

        for i_member, member_name in enumerate(members):
            axs[i_member].plot(mean_members_all_subjects_acrobatics[:, idx_mvt][0][idx_subject, i_member, :], label=f'Subject {idx_subject + 1}',
                     color=color, linestyle=line_style)

    for i_member, member_name in enumerate(members):
        max_members = np.nanmax(mean_members_all_subjects_acrobatics[:, idx_mvt][0])
        axs[i_member].text(105, max_members / 2, f"{member_name}")
        axs[i_member].set_ylim(0, max_members)
        if i_member != 7:
            axs[i_member].get_xaxis().set_visible(False)

    axs[0].set_title(f'{name_acro}', fontsize=11)
    plt.subplots_adjust(left=0.102, right=0.85, top=0.945, bottom=0.047)
    plt.savefig(f"/home/lim/Documents/StageMathieu/limbs_variability/{mvt}.png", dpi=1000)
    # plt.show()
