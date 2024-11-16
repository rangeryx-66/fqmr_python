import numpy as np


def knn(barycenters, k, batch_size):

    n_points = barycenters.shape[0]
    indices_knn = np.empty((n_points, k), dtype=int)

    modulo = n_points % batch_size
    nb_iter = (n_points - modulo) // batch_size

    for i in range(nb_iter):
        i_start, i_end = i * batch_size, (i + 1) * batch_size
        distances = np.linalg.norm(barycenters[i_start:i_end][:, None] - barycenters, axis=-1)
        neighbors = np.argsort(distances, axis=1)[:, :k]
        indices_knn[i_start:i_end] = neighbors

    if modulo != 0:
        # Last piece of computation
        distances = np.linalg.norm(barycenters[-modulo:][:, None] - barycenters, axis=-1)
        neighbors = np.argsort(distances, axis=1)[:, :k]
        indices_knn[-modulo:] = neighbors

    return indices_knn