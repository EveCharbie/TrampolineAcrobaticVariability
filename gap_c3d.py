import ezc3d
import numpy as np
import matplotlib.pyplot as plt

c = ezc3d.c3d('/media/lim/My Passport/collecte_MoCap/2019-08-30/Sarah/Tests/Sa_831_831_1.c3d')
print('Nombre de marqueurs présents:',c['parameters']['POINT']['USED']['value'][0]) # nombre de marqueurs

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

point_labels = c['parameters']['POINT']['LABELS']['value']

markers = [label for label in point_labels if not label.startswith('*')]

markers_validity = {marker: True for marker in markers}


for i, marker in enumerate(markers):
    marker_data = point_data[:, i, 3099:3747]
    is_missing = np.any(np.isnan(marker_data))
    markers_validity[marker] = not is_missing

for marker, is_valid in markers_validity.items():
    if not is_valid:
        print(f"Des données sont manquantes pour le marqueur {marker}.")


for i, marker in enumerate(markers):
    # Extraire les données X, Y, Z pour le marqueur actuel
    # Assurez-vous que l'indexation est correcte pour accéder aux données X, Y, Z
    x_data = point_data[0, i, :]
    y_data = point_data[1, i, :]
    z_data = point_data[2, i, :]

    # Créer un graphique pour le marqueur actuel
    plt.figure(figsize=(10, 4))
    plt.plot(x_data, label='X', color='r')
    plt.plot(y_data, label='Y', color='g')
    plt.plot(z_data, label='Z', color='b')
    plt.axvline(x=3099, color='gray', linestyle='--')
    plt.axvline(x=3403, color='gray', linestyle='--')
    plt.axvline(x=3466, color='gray', linestyle='--')
    plt.axvline(x=3747, color='gray', linestyle='--')
    plt.title(f"Composantes X, Y, Z pour le marqueur {marker}")
    plt.xlabel("Frame")
    plt.ylabel("Valeur")
    plt.legend()
    plt.show()