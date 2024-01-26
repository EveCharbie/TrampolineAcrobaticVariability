import bioviz
import biorbd
import scipy

model = biorbd.Model('/home/lim/Documents/StageMathieu/Data_propre/SaMi/SaMi.bioMod')

data_loaded = scipy.io.loadmat('/home/lim/Documents/StageMathieu/Data_propre/SaMi/fichier.mat')
q_data = data_loaded['Q2']

b = bioviz.Viz(loaded_model=model)
b.load_movement(q_data)
# b.load_experimental_markers(markers)
b.exec()
