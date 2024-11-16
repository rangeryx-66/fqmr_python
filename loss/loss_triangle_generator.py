import numpy as np
from loss.knn import knn

def _numpy_d_f_S_Ss(p_b_hat, b_hat, b):
    """
    Compute the final term using NumPy.

    Parameters:
    - p_b_hat: numpy array of shape (N,)
    - b_hat: numpy array of shape (N, D)
    - b: numpy array of shape (M, D)

    Returns:
    - final_term: scalar value representing the final term
    """
    # 计算所有点对之间的距离矩阵
    distances = np.linalg.norm(b_hat[:, None] - b, axis=-1)

    # 将距离为0的地方设置为无穷大
    distances_filtered = np.where(distances != 0, distances, np.inf)

    # 找到每个 b_hat 点到最近 b 点的距离
    min_b = distances_filtered.min(axis=1)

    # 计算最终结果
    final_term = np.sum(p_b_hat * min_b)

    return final_term

def _sample_points_in_triangles(triangles, num_points_per_triangle=50):
    """
    Sample points uniformly within N triangles in 3D space.

    Args:
    triangles (numpy.ndarray): Array of triangle vertices, shape (N, 3, 3)
                               where N is the number of triangles
    num_points_per_triangle (int): Number of points to sample per triangle

    Returns:
    numpy.ndarray: Sampled points, shape (N * num_points_per_triangle, 3)
    """
    N = triangles.shape[0]

    # Generate random barycentric coordinates
    r1 = np.sqrt(np.random.rand(N, num_points_per_triangle))
    r2 = np.random.rand(N, num_points_per_triangle)

    # Convert barycentric coordinates to Cartesian coordinates
    a = 1 - r1
    b = r1 * (1 - r2)
    c = r1 * r2

    # Reshape for broadcasting
    a = a[:, :, np.newaxis]
    b = b[:, :, np.newaxis]
    c = c[:, :, np.newaxis]

    # Compute the sampled points
    sampled_points = (a * triangles[:, 0:1, :] +
                      b * triangles[:, 1:2, :] +
                      c * triangles[:, 2:3, :])

    # Reshape to (N * num_points_per_triangle, 3)
    sampled_points = sampled_points.reshape(-1, 3)
    return sampled_points


def _numpy_d_r_S_Ss(original_nodes, generated_triangles, probability_generated_triangles,
                    barycenters_generated_triangles, k=20, batch_size=1000):
    """
    Compute the loss function using NumPy.

    Args:
    original_nodes (numpy.ndarray): Original nodes, shape (M, 3)
    generated_triangles (numpy.ndarray): Generated triangles, shape (N, 3, 3)
    probability_generated_triangles (numpy.ndarray): Probabilities for generated triangles, shape (N,)
    barycenters_generated_triangles (numpy.ndarray): Barycenters of generated triangles, shape (N, 3)

    Returns:
    float: Loss value
    """
    sampled_points_generated_triangles = _sample_points_in_triangles(generated_triangles)

    # First term
    distances = np.linalg.norm(original_nodes[:, None] - sampled_points_generated_triangles, axis=-1)
    min_distances = distances.min(axis=1)  # Shape should be (M,)
    min_mean_distances = min_distances.mean()  # No need for axis=1 here
    first_term = probability_generated_triangles * min_mean_distances

    # Second term
    indices_knn = knn(barycenters_generated_triangles, k, batch_size)

    # Compute distances, excluding self-connections
    x_tk = sampled_points_generated_triangles[indices_knn[:, 1:]].reshape(indices_knn.shape[0], k - 1, -1, 3)
    x_y = sampled_points_generated_triangles[indices_knn[:, 0]].reshape(indices_knn.shape[0], 1, -1, 3)
    knn_distances = np.linalg.norm(x_tk - x_y, axis=-1)  # shape: (nb_triangles, nb_neighbors, nb_points_per_triangles)

    knn_distances_mean = knn_distances.mean(axis=2)  # shape: (nb_triangles, nb_neighbors)
    probabilities_tk = probability_generated_triangles[indices_knn[:, 1:]]  # shape: (nb_triangles, nb_neighbors)

    # Ensure shapes match for broadcasting
    dist_times_probability = knn_distances_mean * probabilities_tk  # shape: (nb_triangles, nb_neighbors)
    sum_over_k = dist_times_probability.sum(axis=1)  # shape: (nb_triangles,)
    mean_points_per_triangle_again = sum_over_k.mean(axis=0)  # shape: ()
    normed = mean_points_per_triangle_again / k
    second_term = normed * (1 - probability_generated_triangles)

    loss_d_r_S_Ss = np.sum(first_term + second_term)

    return loss_d_r_S_Ss


def triangle_generator_loss(original_nodes, original_barycenters, selected_triangles, selected_triangles_probabilities):
    """
    Compute the triangle generator loss using NumPy.

    Args:
    original_nodes (numpy.ndarray): Original nodes, shape (M, 3)
    original_barycenters (numpy.ndarray): Original barycenters, shape (M, 3)
    selected_triangles (numpy.ndarray): Selected triangles, shape (N, 3, 3)
    selected_triangles_probabilities (numpy.ndarray): Probabilities for selected triangles, shape (N,)

    Returns:
    tuple: Loss values (d_f_S_Ss, d_r_S_Ss)
    """
    b = original_barycenters
    b_hat = selected_triangles.mean(axis=1)
    p_b_hat = selected_triangles_probabilities

    d_f_S_Ss = _numpy_d_f_S_Ss(p_b_hat, b_hat, b)
    d_r_S_Ss = _numpy_d_r_S_Ss(original_nodes, selected_triangles, selected_triangles_probabilities, b_hat)

    return d_f_S_Ss, d_r_S_Ss
# 示例用法
original_nodes = np.random.rand(100, 3)
original_barycenters = np.random.rand(100, 3)
selected_triangles = np.random.rand(50, 3, 3)
selected_triangles_probabilities = np.random.rand(50)

d_f_S_Ss, d_r_S_Ss = triangle_generator_loss(original_nodes, original_barycenters, selected_triangles, selected_triangles_probabilities)
print("d_f_S_Ss:", d_f_S_Ss)
print("d_r_S_Ss:", d_r_S_Ss)