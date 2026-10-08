import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

def temporal_association(window_size):
    #ouvrir les données
    with open("data_pose.json", "r") as f:
        dico = json.load(f)

    #formater les données
    array = []
    for frame, humans in dico.items():
        for _, position in humans.items():
            x, y = position
            array.append([int(frame), x, y])
    array = np.array(array, dtype=np.float64)

    #réaliser clustering
    nb_frame = int(array[:, 0].max())

    lst_cluster = []
    for window_start in range(0, nb_frame + 1, window_size):
        window_end = window_start + window_size
        bloc = array[(array[:, 0] >= window_start) &(array[:, 0] < window_end)]
        X = bloc[:, 1:3]

        #clustering dbscan
        dbscan = DBSCAN(eps=1, min_samples=5)
        labels = dbscan.fit_predict(X)
        bloc = np.column_stack((bloc, labels))

        for c in np.unique(labels):
            if c == -1:
                continue
            cluster = bloc[bloc[:,-1] == c]
            x_center = np.median(cluster[:, 1])
            y_center = np.median(cluster[:, 2])

            lst_cluster.append([window_start, 
                                float(x_center),
                                float(y_center),
                                int(len(cluster)),
                                int(c),
                                -1])
    array_cluster = np.array(lst_cluster)

    #faire un tri dans les clusters
    lst_suppr = []
    for i in range(len(array_cluster)):
        #si plus de 30 points dans le cluster on le supprime
        if array_cluster[i, 3] > window_size:
            lst_suppr.append(i)
    array_cluster = np.delete(array_cluster, lst_suppr, axis=0)

    #garder que les séquences avec 4 clusters
    frames_valides = []
    for f in np.unique(array_cluster[:, 0]):
        clusters_frame = array_cluster[array_cluster[:, 0] == f]
        if len(clusters_frame) == 4:
            frames_valides.append(f)

    array_cluster = array_cluster[np.isin(array_cluster[:, 0], frames_valides)]
    array_cluster = array_cluster[np.lexsort((array_cluster[:, -1], array_cluster[:, 0]))]

    #association temporelle
    frames = np.unique(array_cluster[:, 0])
    frame_0 = frames[0]
    mask = array_cluster[:, 0] == frame_0
    array_cluster[mask, 5] = np.arange(4)
    mask = array_cluster[:, 0] == frame_0
    array_cluster[mask, 5] = np.arange(4)

    for i in range(len(frames) - 1):
        frame_t = frames[i]
        frame_t1 = frames[i + 1]
        clusters_t = array_cluster[array_cluster[:, 0] == frame_t]
        clusters_t1 = array_cluster[array_cluster[:, 0] == frame_t1]

        #trier par ID temporel
        clusters_t = clusters_t[np.argsort(clusters_t[:, 5])]
        #trier t1 par ID DBSCAN pour avoir une référence stable
        clusters_t1 = clusters_t1[np.argsort(clusters_t1[:, 4])]

        #coordonnées
        points_t = clusters_t[:, 1:3]
        points_t1 = clusters_t1[:, 1:3]

        #matrice de distance
        distance_matrix = np.sqrt(
                ((points_t[:, None, :] - points_t1[None, :, :]) ** 2).sum(axis=2)
            )
        #association 1-à-1
        rows, cols = linear_sum_assignment(distance_matrix)

        #propagation des temporelle des id
        for row, col in zip(rows, cols):
                id_heib = clusters_t[row, 5]
                id_dbscan_t1 = clusters_t1[col, 4]
        
                mask = (
                    (array_cluster[:, 0] == frame_t1) &
                    (array_cluster[:, 4] == id_dbscan_t1)
                )
                array_cluster[mask, 5] = id_heib

    #enregistrement
    np.save("temporal_array.npy", array_cluster)