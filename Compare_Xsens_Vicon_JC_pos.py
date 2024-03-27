import scipy
import os
import pandas as pd
import matplotlib.pyplot as plt
from TrampolineAcrobaticVariability.Function.Function_Class_Basics import (
    OrderMatData,
    load_and_interpolate,
    calculate_mean_std
)

home_path = "/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/GuSe/Q/822/"

fichiers_mat = []
for root, dirs, files in os.walk(home_path):
    for file in files:
        if file.endswith(".mat"):
            full_path = os.path.join(root, file)
            fichiers_mat.append(full_path)

for file_path in fichiers_mat:
    data_loaded = scipy.io.loadmat(file_path)
    JC_Xsens = data_loaded["Jc_in_pelvis_frame"]
    Order_JC_Xsens = data_loaded["JC_order"]

    n_frames = JC_Xsens.shape[2]
    n_JC = JC_Xsens.shape[1]

    # Xsens_position = JC_Xsens.reshape(n_frames, n_JC * 3)
    
    JC_Xsens_transposed = JC_Xsens.transpose(1, 2, 0)
    Xsens_position = JC_Xsens_transposed.reshape(-1, JC_Xsens_transposed.shape[2] * JC_Xsens_transposed.shape[0])

    complete_order = []
    for joint_center in Order_JC_Xsens:
        complete_order.append(f"{joint_center.strip()}_X")
        complete_order.append(f"{joint_center.strip()}_Y")
        complete_order.append(f"{joint_center.strip()}_Z")

    DataFrame_with_colname = pd.DataFrame(Xsens_position)
    DataFrame_with_colname.columns = complete_order
    my_data = OrderMatData(DataFrame_with_colname)

    member_groups = set([name.split("_")[0] for name in complete_order])

    for group in member_groups:
        group_columns = [col for col in complete_order if col.startswith(group)]
        group_data = my_data[group_columns]

        plt.figure(figsize=(10, 6))
        for col in group_columns:
            plt.plot(group_data[col], label=col)

        plt.title(f"Graphique pour {group}")
        plt.xlabel("Index")
        plt.ylabel("Valeur")
        plt.legend()
        plt.show()
