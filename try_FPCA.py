import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Function_Class_Graph import (
    load_and_interpolate,
)
file_path_mat = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Q/"

file_intervals = [
    (file_path_mat + "Sa_821_seul_1.mat", (0, 308)),
    (file_path_mat + "Sa_821_seul_2.mat", (0, 305)),
    (file_path_mat + "Sa_821_seul_3.mat", (0, 307)),
    (file_path_mat + "Sa_821_seul_4.mat", (0, 309)),
    (file_path_mat + "Sa_821_seul_5.mat", (0, 304)),
]

my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]


data_arrays = [instance.get_column_by_index("BrasD", 1) for instance in my_data_instances]
data_arrays = np.degrees(data_arrays)
plt.figure(figsize=(10, 6))
for i in range(0, len(data_arrays)):
    if data_arrays[i][0] > 340:
        data_arrays[i] -= 360
    elif data_arrays[i][0] < -340:
        data_arrays[i] += 360

    if 140 < data_arrays[i][0] <= 360:
        data_arrays[i] -= 180
    elif -360 <= data_arrays[i][0] < -140:
        data_arrays[i] += 180
    plt.plot(data_arrays[i], label=f"Essai{i}")
plt.xlabel("Time (%)")
plt.ylabel("Value")
plt.legend()
plt.show()
