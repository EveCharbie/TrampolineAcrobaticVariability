import scipy.io
import pandas as pd
import numpy as np
import biorbd
from Draw_function import column_names
from scipy.integrate import simpson


class OrderMatData:
    """
    A class for organize and access easily to the data of a dataframe, especially those from .mat
    files in biomechanical contexts
    """

    def __init__(self, dataframe):
        self.dataframe = dataframe
        # Mapping des indices aux suffixes attendus
        self.index_suffix_map = {0: "X", 1: "Y", 2: "Z"}

    def __getitem__(self, key):
        matching_columns = [
            col for col in self.dataframe.columns if col.startswith(key)
        ]
        if not matching_columns:
            raise KeyError(f"Variable {key} not found.")
        return self.dataframe[matching_columns]

    def get_column_by_index(self, key, index):
        # Vérifie si l'index est valide
        if index not in self.index_suffix_map:
            raise KeyError(f"Invalid index {index}.")

        expected_suffix = self.index_suffix_map[index]
        column_name = f"{key}_{expected_suffix}"

        if column_name not in self.dataframe.columns:
            raise KeyError(f"Column {column_name} does not exist.")

        return self.dataframe[column_name]


def load_and_interpolate(file, interval, num_points=100):
    """
    Load and interpol the data from a MATLAB file(.mat).

    Args:
        file (str): Path to the .mat file to load.
        interval (tuple): A tuple of two elements specifying the interval of data to extract.
        num_points (int): Number of points for interpolation of data.

    Returns:
        OrderMatData: An instance of the OrderMatData class containing interpolate data.
    """
    # Load data with the DoF
    data = scipy.io.loadmat(file)
    df = pd.DataFrame(data["Q2"]).T

    df.columns = column_names

    # Select data in specify interval
    df_selected = df.iloc[interval[0] : interval[1]]

    # Interpolate each column to have a uniform number of points
    df_interpolated = df_selected.apply(
        lambda x: np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(x)), x)
    )
    # print(df_interpolated.shape)

    # Create OrderMatData instance and apply it to df_interpolated
    my_data_instance = OrderMatData(df_interpolated)
    return my_data_instance


def calculate_mean_std(data_instances, member, axis):
    """
    Calculates the mean and std for a given member and an axes on all data instances
    """
    data_arrays = [
        instance.get_column_by_index(member, axis) for instance in data_instances
    ]
    Mean_Data = np.mean(data_arrays, axis=0)
    Std_Dev_Data = np.std(data_arrays, axis=0)
    return Mean_Data, Std_Dev_Data


def get_q(Xsens_orientation_per_move):
    """
    This function returns de generalized coordinates in the sequence XYZ (biorbd) from the quaternion of the orientation
    of the Xsens segments.
    The translation is left empty as it has to be computed otherwise.
    I am not sure if I would use this for kinematics analysis, but for visualisation it is not that bad.
    """

    parent_idx_list = {
        "Pelvis": None,  # 0
        "L5": [0, "Pelvis"],  # 1
        "L3": [1, "L5"],  # 2
        "T12": [2, "L3"],  # 3
        "T8": [3, "T12"],  # 4
        "Neck": [4, "T8"],  # 5
        "Head": [5, "Neck"],  # 6
        "ShoulderR": [4, "T8"],  # 7
        "UpperArmR": [7, "ShoulderR"],  # 8
        "LowerArmR": [8, "UpperArmR"],  # 9
        "HandR": [9, "LowerArmR"],  # 10
        "ShoulderL": [4, "T8"],  # 11
        "UpperArmL": [11, "ShoulderR"],  # 12
        "LowerArmL": [12, "UpperArmR"],  # 13
        "HandL": [13, "LowerArmR"],  # 14
        "UpperLegR": [0, "Pelvis"],  # 15
        "LowerLegR": [15, "UpperLegR"],  # 16
        "FootR": [16, "LowerLegR"],  # 17
        "ToesR": [17, "FootR"],  # 18
        "UpperLegL": [0, "Pelvis"],  # 19
        "LowerLegL": [19, "UpperLegL"],  # 20
        "FootL": [20, "LowerLegL"],  # 21
        "ToesL": [21, "FootL"],  # 22
    }

    nb_frames = Xsens_orientation_per_move.shape[0]
    Q = np.zeros((23 * 3, nb_frames))
    rotation_matrices = np.zeros((23, nb_frames, 3, 3))
    for i_segment, key in enumerate(parent_idx_list):
        for i_frame in range(nb_frames):
            Quat_normalized = Xsens_orientation_per_move[
                i_frame, i_segment * 4 : (i_segment + 1) * 4
            ] / np.linalg.norm(
                Xsens_orientation_per_move[i_frame, i_segment * 4 : (i_segment + 1) * 4]
            )
            Quat = biorbd.Quaternion(
                Quat_normalized[0],
                Quat_normalized[1],
                Quat_normalized[2],
                Quat_normalized[3],
            )

            RotMat_current = biorbd.Quaternion.toMatrix(Quat).to_array()
            z_rotation = biorbd.Rotation.fromEulerAngles(
                np.array([-np.pi / 2]), "z"
            ).to_array()
            RotMat_current = z_rotation @ RotMat_current

            if parent_idx_list[key] is None:
                RotMat = np.eye(3)
            else:
                RotMat = rotation_matrices[parent_idx_list[key][0], i_frame, :, :]

            RotMat_between = np.linalg.inv(RotMat) @ RotMat_current
            RotMat_between = biorbd.Rotation(
                RotMat_between[0, 0],
                RotMat_between[0, 1],
                RotMat_between[0, 2],
                RotMat_between[1, 0],
                RotMat_between[1, 1],
                RotMat_between[1, 2],
                RotMat_between[2, 0],
                RotMat_between[2, 1],
                RotMat_between[2, 2],
            )
            Q[
                i_segment * 3 : (i_segment + 1) * 3, i_frame
            ] = biorbd.Rotation.toEulerAngles(RotMat_between, "xyz").to_array()

            rotation_matrices[i_segment, i_frame, :, :] = RotMat_current
    return Q


def recons_kalman(n_frames, num_markers, markers_xsens, model, initial_guess):
    markersOverFrames = []
    for i in range(n_frames):
        node_segment = []
        for j in range(num_markers):
            node_segment.append(biorbd.NodeSegment(markers_xsens[:, j, i].T))
        markersOverFrames.append(node_segment)

    freq = 200
    params = biorbd.KalmanParam(freq)
    kalman = biorbd.KalmanReconsMarkers(model, params)
    kalman.setInitState(initial_guess[0], initial_guess[1], initial_guess[2])

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


def find_index(name, list):
    return list.index(name)


def calculate_rmsd(markers, pos_recons):
    # Vérifier que les formes des tableaux sont identiques
    assert (
        markers.shape == pos_recons.shape
    ), "Les tableaux doivent avoir la même forme."

    n_frames = markers.shape[2]
    rmsd_per_frame = np.zeros(n_frames)

    for i in range(n_frames):
        # Calculer la différence entre les ensembles de marqueurs pour le cadre i
        diff = markers[:, :, i] - pos_recons[:, :, i]
        # Calculer la norme au carré de la différence pour chaque marqueur
        squared_diff = np.nansum(diff**2, axis=0)
        # Calculer la moyenne des différences au carré
        mean_squared_diff = np.mean(squared_diff)
        # Calculer la racine carrée de la moyenne des différences au carré pour obtenir la RMSD
        rmsd_per_frame[i] = np.sqrt(mean_squared_diff)

    return rmsd_per_frame


def normalise_vecteurs(vecteurs):
    normes = np.linalg.norm(vecteurs, axis=1)[:, np.newaxis]
    vecteurs_normalises = vecteurs / normes
    return vecteurs_normalises


parent_list = {
    "Pelvis": None,  # 0
    "Thorax": [0, "Pelvis"],  # 1
    "Tete": [1, "Thorax"],  # 2
    "BrasD": [1, "Thorax"],  # 3
    "ABrasD": [3, "BrasD"],  # 4
    "MainD": [4, "ABrasD"],  # 5
    "BrasG": [1, "Thorax"],  # 6
    "ABrasG": [6, "BrasG"],  # 7
    "MainG": [7, "ABrasG"],  # 8
    "CuisseD": [0, "Pelvis"],  # 9
    "JambeD": [9, "CuisseD"],  # 10
    "PiedD": [10, "JambeD"],  # 11
    "CuisseG": [0, "Pelvis"],  # 12
    "JambeG": [12, "CuisseG"],  # 13
    "PiedG": [13, "JambeG"],  # 14
}


def trouver_index_parent(nom_parent):
    # Créer une liste des clés de parent_list pour obtenir les index
    keys_list = list(parent_list.keys())
    # Trouver l'index du nom du parent dans cette liste
    index_parent = keys_list.index(nom_parent) if nom_parent in keys_list else None
    return index_parent


def calculate_scores(fd, fpca_components, dx):
    # Nombre d'essais, nombre de points par essai, nombre de FPC
    n_essais, n_points, _ = fd.data_matrix.shape
    n_fpc = fpca_components.shape[0]

    # Initialiser un tableau pour stocker les scores
    scores = np.zeros((n_essais, n_fpc))

    # Itérer sur chaque essai
    for i in range(n_essais):
        essai_values = fd.data_matrix[i, :, 0]  # Extraire les valeurs pour l'essai courant

        # Itérer sur chaque FPC
        for j in range(n_fpc):
            fpc_values = fpca_components[j, :, 0]  # Extraire les valeurs pour la FPC courante

            # Calculer le produit et intégrer pour obtenir le score
            produit = essai_values * fpc_values
            score = simpson(produit, dx=dx)

            scores[i, j] = score  # Stocker le score calculé

    return scores
