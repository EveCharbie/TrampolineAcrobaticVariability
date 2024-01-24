"""
The goal of this program is to reconstruct the kinematics of the motion capture data using a Kalman filter.
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import biorbd
import scipy
import ezc3d


def recons_kalman(num_frames, num_markers, markers_xsens, model):
    markersOverFrames = []
    for i in range(num_frames):
        node_segment = []
        for j in range(num_markers):
            node_segment.append(biorbd.NodeSegment(markers_xsens[:, j, i].T))
        markersOverFrames.append(node_segment)

    freq = 200
    noise_factor = 0.01
    error_factor = 0.1
    params = biorbd.KalmanParam(freq, noise_factor, error_factor)

    kalman = biorbd.KalmanReconsMarkers(model, params)

    # Perform the kalman filter for each frame (the first frame is much longer than the next)
    Q = biorbd.GeneralizedCoordinates(model)
    Qdot = biorbd.GeneralizedVelocity(model)
    Qddot = biorbd.GeneralizedAcceleration(model)
    q_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    qdot_recons = np.ndarray((model.nbQ(), len(markersOverFrames)))
    for i, targetMarkers in enumerate(markersOverFrames):
        kalman.reconstructFrame(model, targetMarkers, Q, Qdot, Qddot)
        q_recons[:, i] = Q.to_array()
        qdot_recons[:, i] = Qdot.to_array()
    return q_recons, qdot_recons


c = ezc3d.c3d('/media/lim/My Passport/labelling/2019-08-30/Sarah/Tests/Sa_821_seul_1.c3d')
point_data = c['data']['points']
n_markers = point_data.shape[1]
nf_mocap = point_data.shape[2]
f_mocap = c['parameters']['POINT']['RATE']['value'][0]

markers = np.zeros((3, n_markers, nf_mocap))
for i in range(nf_mocap):
    for j in range(n_markers):
        markers[:, j, i] = point_data[:3, j, i]

model = biorbd.Model('/home/lim/Documents/StageMathieu/Data_propre/SaMi/SaMi.bioMod')

q_recons, qdot_recons = recons_kalman(nf_mocap, n_markers, markers, model)

q_recons = q_recons/1000

# Création d'un dictionnaire pour le stockage
mat_data = {'Q2': q_recons}

# Enregistrement dans un fichier .mat
scipy.io.savemat('/home/lim/Documents/StageMathieu/Data_propre/SaMi/fichier.mat', mat_data)
