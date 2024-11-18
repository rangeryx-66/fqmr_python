import trimesh as tr
import numpy as np
from scipy.spatial import cKDTree
def computeLoss(original_mesh,mesh,dot_weight=0.5,face_weight=0.5,simplification_rate=4):
    original_vertices=original_mesh.vertices
    vertices=mesh.vertices
    original_dots=[]
    dots=[]
    original_baryCenters=[]
    baryCenters=[]
    for face in original_mesh.faces:
        tri_vertices = original_mesh.vertices[face]
        dot1=tri_vertices[0]
        dot2=tri_vertices[1]
        dot3=tri_vertices[2]
        x=(dot1[0]+dot2[0]+dot3[0])/3
        y=(dot1[1]+dot2[1]+dot3[1])/3
        z=(dot1[2]+dot2[2]+dot3[2])/3
        original_baryCenters.append([x,y,z])
    for face in mesh.faces:
        tri_vertices = mesh.vertices[face]
        dot1=tri_vertices[0]
        dot2=tri_vertices[1]
        dot3=tri_vertices[2]
        x=(dot1[0]+dot2[0]+dot3[0])/3
        y=(dot1[1]+dot2[1]+dot3[1])/3
        z=(dot1[2]+dot2[2]+dot3[2])/3
        baryCenters.append([x,y,z])
    for vertice in original_vertices:
        x=vertice[0]
        y=vertice[1]
        z=vertice[2]
        original_dots.append([x,y,z])
    for vertice in vertices:
        x=vertice[0]
        y=vertice[1]
        z=vertice[2]
        dots.append([x,y,z])
    dot_loss=compute_dots_distance_loss(original_dots,dots)
    face_loss=compute_dots_distance_loss(original_baryCenters,baryCenters)
    face_simp_num=np.abs(len(original_mesh.faces)-len(mesh.faces))
    #return dot_loss,face_loss,face_simp_num
    return dot_weight*dot_loss+face_weight*face_loss-face_simp_num*simplification_rate
def compute_nearest_distances(dots1,dots2):
    tree = cKDTree(dots2)
    distances, indices = tree.query(dots1, k=1)
    return distances
def compute_dots_distance_loss(dots1,dots2):
    distances1=compute_nearest_distances(dots1,dots2)
    distances2=compute_nearest_distances(dots2,dots1)
    sum=0
    for distance in distances1:
        sum+=distance**2
    for distance in distances2:
        sum+=distance**2
    return sum


# mesh = tr.load('input.obj', force='mesh')
# mesh2=tr.load('output.obj', force='mesh')
# print(computeLoss(mesh,mesh2))

