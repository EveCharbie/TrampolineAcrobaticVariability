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

c = ezc3d.c3d('/media/lim/My Passport/labelling/2019-08-30/Sarah/Tests/Sa_821_seul_2.c3d')
print('Nombre de marqueurs présents:',c['parameters']['POINT']['USED']['value'][0]); # nombre de marqueurs

point_data = c['data']['points'] #récupération trajectoires marqueurs
point_labels = c['parameters']['POINT']['LABELS']# récupération labels marqueurs
point_rate = c['parameters']['POINT']['RATE'] # récupération fréquence mocap
analog_data = c['data']['analogs'] # récupération données analogiques
analog_labels =c['parameters']['ANALOG']['LABELS'] # récupération labels analogiques
analog_rate = c['parameters']['ANALOG']['RATE'] # récupération fréquence analogique

print('fréquence échantillonnage capture de mouvement:', point_rate['value'][0],'Hz')
print('fréquence échantillonnage données analogiques:', analog_rate['value'][0],'Hz')

# nombre d'échantillons mocap et analog
nf_mocap=len(point_data[0][0][:])
n_markers=len(point_data[0][:])
n_dims=len(point_data[:])
nf_analog=len(analog_data[0][:][:])

#reconstruction vecteur temps capture et signaux analogiques
t_point=np.linspace(0., nf_mocap/point_rate['value'][0], num=nf_mocap)
t_analog=np.linspace(0., nf_analog/analog_rate['value'][0], num=nf_analog)
f_mocap=point_rate['value'][0] #fréquence capture


print('Noms des marqueurs',point_labels)
print('Nombre de marqueurs',n_markers)
print('Nombre de frames',nf_mocap)

def recons_kalman(num_frames, num_markers, markers_xsens, model):
    markersOverFrames = []
    for i in range(num_frames):
        node_segment = []
        for j in range(num_markers):
            node_segment.append(biorbd.NodeSegment(markers_xsens[:, j, i].T))
        markersOverFrames.append(node_segment)

    # Create a Kalman filter structure
    freq = 200  # Hz
    params = biorbd.KalmanParam(freq)
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

model = "/home/lim/Documents/StageMathieu/Data_propre/SaMi/SaMi.bioMod"

q_recons, qdot_recons= recons_kalman(nf_mocap, n_markers,point_data,model)

