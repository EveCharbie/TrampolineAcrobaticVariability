import bioviz
import biorbd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

model_path = "/home/lim/Documents/StageMathieu/Data_propre/SaMi/SaMi.bioMod"
model = biorbd.Model(model_path)
# data_loaded = scipy.io.loadmat('/home/lim/Documents/StageMathieu/Data_propre/SaMi/fichier.mat')
# q_data = data_loaded['Q2']
#
# b = bioviz.Viz(loaded_model=model)
# b.load_movement(q_data)
# # b.load_experimental_markers(markers)
# b.exec()

# Configuration neutre
q = np.zeros(model.nbQ())

# Obtenir les positions des marqueurs et les noms
markers = [model.marker(q, i) for i in range(model.nbMarkers())]
markers_positions = np.array([marker.to_array() for marker in markers])
markers_names = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]

# Création du graphique
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(markers_positions[:, 0], markers_positions[:, 1], markers_positions[:, 2])

# Annotation pour afficher le nom du marqueur
annot = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(20, 20),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="w"),
    arrowprops=dict(arrowstyle="->"),
)
annot.set_visible(False)


# Fonction pour mettre à jour l'annotation en fonction de la position de la souris
def update_annot(ind):
    pos = sc.get_offsets()[ind["ind"][0]]
    annot.xy = pos
    text = "{}".format(" ".join([markers_names[n] for n in ind["ind"]]))
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.4)


# Fonction pour gérer l'événement de mouvement de la souris
def hover(event):
    vis = annot.get_visible()
    if event.inaxes == ax:
        cont, ind = sc.contains(event)
        if cont:
            update_annot(ind)
            annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            if vis:
                annot.set_visible(False)
                fig.canvas.draw_idle()


# Connecter l'événement de mouvement de la souris à la fonction de gestion
fig.canvas.mpl_connect("motion_notify_event", hover)

# Afficher le graphique
plt.show()
