import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

file = "/media/lim/My Passport/XsensData/SaBe/exports_shoulder_height/SaBe_01/angularVelocity.mat"

data = scipy.io.loadmat(file)
df = data['angularVelocity']

print(df.shape)

nb_art = df.shape[1]

for i in range(nb_art):
    plt.figure(figsize=(5, 3))
    plt.plot(df[:, i], label=f'Segment {i+1}')
    plt.title(f'Segment {i+1}')
    plt.xlabel('Frame')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.show()