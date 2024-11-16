import numpy as np

def np_d_P_Ps(p_y, x, y, simplification_rate):

    distances = np.linalg.norm(x[:, None] - y, axis=-1)

    min_x = distances.min(axis=1)

    min_y_values = distances.min(axis=0)
    min_y_indices = distances.argmin(axis=0)

    first_term = np.sum(np.take(p_y, min_y_indices) * min_y_values)

    second_term = np.sum(min_x * p_y)

    d_p_ps = first_term + second_term
    return d_p_ps * simplification_rate

p_y = np.array([0.2, 0.5, 0.3])        # 权重向量，对应3个点
x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # 点集 P (3个点，每点2维)
y = np.array([[2.0, 3.0], [4.0, 5.0]])  # 点集 P_s (2个点，每点2维)
simplification_rate = 0.8              # 简化比例
print(np_d_P_Ps(p_y, x, y, simplification_rate))