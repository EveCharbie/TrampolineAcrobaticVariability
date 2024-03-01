import scipy.io
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/home/lim/Documents/StageMathieu/XsensData/GuSe/exports_shoulder_height/"

alldata = []
elements = os.listdir(file_path)

for folder in enumerate(elements):
    file_path_complete = f"{file_path}{folder[1]}/angularVelocity.mat"
    data = scipy.io.loadmat(file_path_complete)
    df = data['angularVelocity']
    alldata.append(df)

nb_art = alldata[0].shape[1]

for file in range(len(alldata)):
    mydata = alldata[file]

    for dof in range(nb_art):
        plt.figure(figsize=(5, 3))
        plt.plot(mydata[:, dof], label=f'Segment {dof+1}')
        plt.title(f'Segment {dof+1}')
        plt.xlabel('Frame')
        plt.ylabel('Angle (rad)')
        plt.legend()
        plt.show()

