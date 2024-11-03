import numpy as np
import math
import trimesh as tr


def getangles(vertices):
    edges = [np.linalg.norm(vertices[i] - vertices[(i + 1) % 3]) for i in range(3)]
    angles = []
    for i in range(3):
        a, b, c = edges[(i + 2) % 3], edges[(i + 1) % 3], edges[i]
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        angles.append(math.degrees(angle))
    angle1=angles[0]
    angle2=angles[1]
    angle3=angles[2]
    angles[0]=angle3
    angles[1]=angle1
    angles[2]=angle2
    return angles


def obtuse_angle_index(vertices):
    angles = getangles(vertices)
    return angles.index(max(angles))


def obtuse_bisector_intersection(vertices):
    obtuse_index = obtuse_angle_index(vertices)

    obtuse_vertex = vertices[obtuse_index]

    opposite_edge_vertex1 = vertices[(obtuse_index + 1) % 3]
    opposite_edge_vertex2 = vertices[(obtuse_index + 2) % 3]

    edge1 = np.linalg.norm(obtuse_vertex - opposite_edge_vertex1)
    edge2 = np.linalg.norm(obtuse_vertex - opposite_edge_vertex2)

    ratio = edge2 / edge1
    intersection_point = (opposite_edge_vertex1 + ratio * opposite_edge_vertex2) / (1 + ratio)

    return intersection_point


def split2(mesh, min_allowed_angle):
    if isinstance(mesh, tr.Scene):

        mesh1 = mesh.dump(concatenate=True)
    else:
        mesh1 = mesh

    new_faces = []

    for face in mesh1.faces:
        tri_vertices = mesh1.vertices[face]
        angles = getangles(tri_vertices)
        if max(angles) > min_allowed_angle:
            index = obtuse_angle_index(tri_vertices)


            i1, i2 = (index + 1) % 3, (index + 2) % 3
            new_point = obtuse_bisector_intersection(tri_vertices)



            new_point_idx = len(mesh1.vertices)
            mesh1.vertices = np.vstack([mesh1.vertices, new_point])


            new_faces.append([face[i1], face[index], new_point_idx])
            new_faces.append([face[i2], face[index], new_point_idx])

        else:

            new_faces.append(face)


    new_mesh = tr.Trimesh(vertices=mesh1.vertices, faces=new_faces)
    return new_mesh

# 示例调用
# veritce=[np.array([0,0,0]),np.array([1.73,0,0]),np.array([0,1,0])]
# print(obtuse_angle_index(veritce))
# mesh = tr.load("input.obj", force="mesh")
# mesh = split(mesh, 100)
# mesh.export("output.obj")