import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Function_Class_Basics import (
    load_and_interpolate,
)
file_path_mat = "/home/lim/Documents/StageMathieu/DataTrampo/Sarah/Q/"
file_path_mat2 = "/home/lim/Documents/StageMathieu/DataTrampo/Jeremy/Q/"

file_intervals = [
    (file_path_mat + "Sa_821_seul_1.mat", (0, 308)),
    (file_path_mat + "Sa_821_seul_2.mat", (0, 305)),
    (file_path_mat + "Sa_821_seul_3.mat", (0, 307)),
    (file_path_mat + "Sa_821_seul_4.mat", (0, 309)),
    (file_path_mat + "Sa_821_seul_5.mat", (0, 304)),

    (file_path_mat + "Sa_821_822_2.mat", (0, 307)),
    (file_path_mat + "Sa_821_822_3.mat", (0, 307)),
    (file_path_mat + "Sa_821_822_4.mat", (0, 302)),
    (file_path_mat + "Sa_821_822_5.mat", (0, 306)),

    # (file_path_mat2 + "Je_821_821_1.mat", (0, 336)),
    # (file_path_mat2 + "Je_821_821_2.mat", (0, 335)),
    # (file_path_mat2 + "Je_821_821_3.mat", (0, 342)),
    # (file_path_mat2 + "Je_821_821_4.mat", (0, 342)),
    # (file_path_mat2 + "Je_821_821_5.mat", (0, 339)),

]

my_data_instances = [load_and_interpolate(file, interval) for file, interval in file_intervals]


data_arrays = [instance.get_column_by_index("CuisseG", 0) for instance in my_data_instances]
plt.figure(figsize=(10, 6))
for i in range(0, len(data_arrays)):
    if data_arrays[i][0] > 2*np.pi:
        data_arrays[i] -= 2*np.pi
    elif data_arrays[i][0] < -2*np.pi:
        data_arrays[i] += 2*np.pi

    if np.pi < data_arrays[i][0] <= 2*np.pi:
        data_arrays[i] -= np.pi
    elif -2*np.pi <= data_arrays[i][0] < -np.pi:
        data_arrays[i] += np.pi
    plt.plot(data_arrays[i], label=f"Essai{i}")
    df = pd.DataFrame(data_arrays).T  # .T pour transposer si nécessaire (chaque essai en colonne)

    # Enregistrement du DataFrame en fichier CSV
    csv_file_path = '/home/lim/Documents/StageMathieu/DataTrampo/Sarah/my_adjusted_data.csv'  # Chemin du fichier CSV
    df.to_csv(csv_file_path, index=False)

plt.xlabel("Time (%)")
plt.ylabel("Value")
plt.legend()
plt.show()

