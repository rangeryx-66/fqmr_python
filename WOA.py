import math
import numpy as np
import trimesh as tr
import pyfqmr
import open3d as o3d
import openmesh as om
min_num=2000
def load_mesh(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    return mesh


def mesh_to_point_cloud(mesh, num_points=10000):
    if len(mesh.vertices) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    else:
        raise ValueError("Mesh has no vertices.")
    return pcd


def compute_hausdorff_distance(pcd1, pcd2):
    distance1 = pcd1.compute_point_cloud_distance(pcd2)
    distance2 = pcd2.compute_point_cloud_distance(pcd1)
    max_distance = max(np.max(distance1), np.max(distance2))
    return max_distance


def compute_mse_distance(pcd1, pcd2):
    distance1 = pcd1.compute_point_cloud_distance(pcd2)
    distance2 = pcd2.compute_point_cloud_distance(pcd1)
    mse = (np.mean(np.square(distance1)) + np.mean(np.square(distance2))) / 2
    return mse


def compute_simplification_ratio(original_mesh, simplified_mesh):
    original_face_count = len(original_mesh.triangles)
    simplified_face_count = len(simplified_mesh.triangles)
    reduction_ratio = 1 - (simplified_face_count / original_face_count)
    return reduction_ratio + 0.1


def compute_loss(original_file, simplified_file, weight_similarity=0.5, weight_reduction=0.5):
    original_mesh = load_mesh(original_file)
    simplified_mesh = load_mesh(simplified_file)

    original_pcd = mesh_to_point_cloud(original_mesh)
    simplified_pcd = mesh_to_point_cloud(simplified_mesh)

    similarity_loss = compute_mse_distance(original_pcd, simplified_pcd)
    reduction_ratio = compute_simplification_ratio(original_mesh, simplified_mesh)

    loss = weight_similarity * similarity_loss + weight_reduction * reduction_ratio
    # loss =weight_reduction * reduction_ratio
    return loss
# 定义一个函数，用于计算三角形三个角的最小角度
def minimum_angle(vertices):
    edges = [np.linalg.norm(vertices[i] - vertices[(i + 1) % 3]) for i in range(3)]
    angles = []
    for i in range(3):
        a, b, c = edges[i], edges[(i + 1) % 3], edges[(i + 2) % 3]
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        angles.append(math.degrees(angle))
    return min(angles)


def split(mesh, min_allowed_angle):
    if isinstance(mesh, tr.Scene):
        # 提取其中的第一个 mesh
        mesh1 = mesh.dump(concatenate=True)
    else:
        mesh1 = mesh

    new_faces = []

    for face in mesh1.faces:

        tri_vertices = mesh1.vertices[face]
        min_angle = minimum_angle(tri_vertices)

        if min_angle < min_allowed_angle:
            # 如果最小角度小于允许的阈值，则将三角形分裂为两个更规则的三角形
            # 选择最长的边，将其分裂
            edges = [np.linalg.norm(tri_vertices[i] - tri_vertices[(i + 1) % 3]) for i in range(3)]
            longest_edge_idx = np.argmax(edges)

            # 找出这条边的两个顶点
            i1, i2 = longest_edge_idx, (longest_edge_idx + 1) % 3
            new_point = (tri_vertices[i1] + tri_vertices[i2]) / 2

            # 新的点加入到顶点列表
            new_point_idx = len(mesh1.vertices)
            mesh1.vertices = np.vstack([mesh1.vertices, new_point])

            # 生成新的三角形
            i3 = (longest_edge_idx + 2) % 3
            new_faces.append([face[i1], face[i3], new_point_idx])
            new_faces.append([face[i2], face[i3], new_point_idx])

        else:
            # 保持原来的三角形
            new_faces.append(face)

    # 创建一个新的Trimesh对象
    new_mesh = tr.Trimesh(vertices=mesh1.vertices, faces=new_faces)
    return new_mesh


def remove_duplicate_vertices(obj):
    if isinstance(mesh, tr.Scene):
        # 提取其中的第一个 mesh
        bunny = mesh.dump(concatenate=True)
    else:
        bunny = mesh
    vertices = bunny.vertices
    faces = bunny.faces
    unique_vertices, unique_indices = np.unique(vertices, axis=0, return_inverse=True)

    new_faces = unique_indices[faces]

    new_obj = tr.Trimesh(vertices=unique_vertices, faces=new_faces)
    return new_obj


def simplify(nums, mesh1):
    if isinstance(mesh1, tr.Scene):
        # 提取其中的第一个 mesh
        bunny = mesh1.dump(concatenate=True)
    else:
        bunny = mesh1

    # 创建 Simplify 对象并设置网格
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(bunny.vertices, bunny.faces)

    # 进行简化
    mesh_simplifier.simplify_mesh(target_count=nums, aggressiveness=7, preserve_border=True, verbose=10)

    # 获取简化后的网格
    vertices, faces, normals = mesh_simplifier.getMesh()

    # 将简化后的网格转换为 Trimesh 对象
    simplified_mesh = tr.Trimesh(vertices=vertices, faces=faces)
    return simplified_mesh

def calculate_similarity(X, mesh):
    mesh_copy = mesh

    sequence = X.astype(int)
    for i in range(len(X)):
        if i % 2 == 0:
            mesh_copy = split(mesh_copy, sequence[i])
        else:
            mesh_copy = simplify(len(mesh_copy.faces)-sequence[i], mesh_copy)

    if 200<=len(mesh_copy.faces)<=300:
        min_num=len(mesh_copy.faces)
        print('minnum:',min_num)
        mesh_copy.export('output_200.obj')
    if 100<=len(mesh_copy.faces)<=200:
        min_num=len(mesh_copy.faces)
        print('minnum:',min_num)
        mesh_copy.export('output_100.obj')
    mesh.export('output_temps1.obj')
    mesh_copy.export('output_temps2.obj')
    return compute_loss('output_temps1.obj','output_temps2.obj')
def tuili(X,mesh,t):
    mesh_copy = mesh

    sequence = X.astype(int)
    for i in range(len(X)):
        if i % 2 == 0:
            mesh_copy = split(mesh_copy, sequence[i])
        else:
            mesh_copy = simplify(len(mesh_copy.faces) - sequence[i], mesh_copy)
    print(len(mesh_copy.faces))
    mesh_copy.export(f'output{t:02d}.obj')

# 定义鲸鱼优化算法类
def is_feasible(X, num):
    X = X.astype(int)
    spilt_sum = sum(X[i] for i in range(0, len(X), 2))
    simplify_sum = sum(X[i] for i in range(1, len(X), 2))
    return simplify_sum - spilt_sum >= num / 2


class WOA_DE:
    def __init__(self, n_agents, max_iter, dim, lb,even_ub, odd_ub, obj_func, mut, crossp, mesh):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.dim = dim
        self.lb = lb
        # self.ub = ub
        self.obj_func = obj_func
        self.mut = mut
        self.crossp = crossp

        self.best_agent = None
        self.best_score = float("inf")
        self.mesh = mesh
        indices = np.arange(self.dim)

        self.even_ub = even_ub
        self.odd_ub = odd_ub
        self.upper_bounds = np.where(indices % 2 == 0, self.even_ub, self.odd_ub)
        self.agents = np.random.uniform(self.lb,self.upper_bounds,(n_agents, dim))

    def apply_bounds(self, agents):

        indices = np.arange(self.dim)
        upper_bounds = np.where(indices % 2 == 0, self.even_ub, self.odd_ub)
        return np.clip(agents, self.lb, upper_bounds)

    def optimize(self):

        for i in range(self.n_agents):

            fitness = self.obj_func(self.agents[i], self.mesh)


            if self.best_agent is None or (fitness < self.best_score and is_feasible(self.agents[i], len(self.mesh.faces))):
                self.best_score = fitness
                self.best_agent = self.agents[i].copy()

        for t in range(self.max_iter):
            # 更新鲸鱼群体的位置
            a = 2 - t * (2 / self.max_iter)  # 线性递减系数

            for i in range(self.n_agents):
                r1 = np.random.rand()
                r2 = np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2

                if np.random.rand() < 0.5:
                    if abs(A) < 1:

                        D = abs(C * self.best_agent - self.agents[i])
                        trial = self.best_agent - A * D
                    else:
                        idxs = [idx for idx in range(self.n_agents) if idx != i]
                        x, y, z = self.agents[np.random.choice(idxs, 3, replace=False)]
                        mutant = np.clip(x + self.mut * (y - z), self.lb, self.upper_bounds)
                        cross_points = np.random.rand(self.dim) < self.crossp
                        if not np.any(cross_points):
                            cross_points[np.random.randint(0, self.dim)] = True
                        trial = np.where(cross_points, mutant, self.agents[i])

                else:

                    b = 1
                    l = np.random.uniform(-1, 1)
                    distance_to_best = abs(self.best_agent - self.agents[i])
                    trial = distance_to_best * np.exp(b * l) * np.cos(2 * np.pi * l) + self.best_agent

                if is_feasible(trial, len(self.mesh.faces)) and self.obj_func(trial, self.mesh) < self.obj_func(self.agents[i], self.mesh):
                    self.agents[i] = trial
                    if self.obj_func(trial, self.mesh) < self.best_score:
                        self.best_score = self.obj_func(trial, self.mesh)
                        self.best_agent = trial


            self.agents = np.clip(self.agents, self.lb, self.upper_bounds)

            tuili(self.best_agent, mesh, t)

        return self.best_agent, self.best_score



# 参数设置
n_agents = 20
max_iter = 50
dim = 8
lb = 0
even_ub = 3
odd_ub=500
mut = 0.8
crossp = 0.7
mesh = tr.load('input.obj', force='mesh')
loss=compute_loss('input.obj','input.obj')
print("loss:",loss)
woa_de = WOA_DE(n_agents, max_iter, dim, lb, even_ub,odd_ub, calculate_similarity, mut, crossp, mesh)
best_agent, best_score = woa_de.optimize()
print("最优解: ", best_agent)
print("最优目标函数值: ", best_score)