import scipy.io
import pandas as pd
import numpy as np
import biorbd
from scipy.integrate import simpson
from scipy.interpolate import interp1d
import pickle


data_loaded = scipy.io.loadmat("/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/JoBu/Pos_JC/41/de035f64_0_0-37_556__41__0__eyetracking_metrics.mat")

for key in data_loaded.keys():
    print(key, type(data_loaded[key]))

if 'wall_index' in data_loaded:
    print("Wall index is present for participant")
else:
    print("Wall index missing for participant")

with open("/home/lim/Documents/StageMathieu/DataTrampo/Xsens_pkl/JoBu/42/de035f64_0_0-37_556__42__0__eyetracking_metrics.pkl", "rb") as fichier_pkl:
    # Charger les données à partir du fichier ".pkl"
    eye_tracking_metrics = pickle.load(fichier_pkl)

subject_expertise = eye_tracking_metrics["subject_expertise"]
subject_name = eye_tracking_metrics["subject_name"]
move_orientation = eye_tracking_metrics["move_orientation"]
Xsens_orientation_per_move = eye_tracking_metrics["Xsens_orientation_per_move"]
Xsens_position_rotated_per_move = eye_tracking_metrics["Xsens_position_rotated_per_move"]
laterality = eye_tracking_metrics["laterality"]
wall_index = eye_tracking_metrics["wall_index"]



# Enregistrement dans un fichier .mat
scipy.io.savemat("/home/lim/Documents/de035f64_0_0-37_556__41__0__eyetracking_metrics.mat", mat_data)