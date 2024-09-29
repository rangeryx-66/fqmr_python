import re
import numpy as np
import openmesh as om
import numpy
import math
import random
from scipy import constants
from fast_qem_class import *


class MeshSimplifier:
    def __init__(self):
        self.triangles = []
        self.vertices = []
        self.refs = []
        self.mtllib = ""
        self.materials = []
        self.collapses = []

    def vertex_error(self, q, x, y, z):
        return q[0] * x * x + 2 * q[1] * x * y + 2 * q[2] * x * z + 2 * q[3] * x + q[4] * y * y + 2 * q[5] * y * z + 2 * \
            q[6] * y + q[7] * z * z + 2 * q[8] * z + q[9]

    def calculate_error(self, id_v1, id_v2, p_result):
        # v1 = self.vertices[id_v1]
        # v2 = self.vertices[id_v2]
        # K1 = 0.
        # K2 = 0.
        # S1 = 0.
        # S2 = 0.
        # D1 = 0.
        # D2 = 0.
        # for k in range(v1.tcount):
        #     r = self.refs[v1.tstart + k]
        #     t = self.triangles[r.tid]
        #     near1 = t.v[(r.tvertex + 1) % 3]
        #     near2 = t.v[(r.tvertex + 2) % 3]
        #     near1V = self.vertices[near1]
        #     near2V = self.vertices[near2]
        #     print(near1V.p.x)
        #     print(near2V.p.y)
        #     D1 = D1 + self.angleBetweenVectors(v1.p, near1V.p, near2V.p)
        #     S1 = S1 + self.crossProductMagnitude(v1.p, near1V.p, near2V.p)
        #
        # K1 = 3 * (2 * 3.14159265358979323846 - D1) / S1
        # for k in range(v2.tcount):
        #     r = self.refs[v2.tstart + k]
        #     t = self.triangles[r.tid]
        #     near1 = t.v[(r.tvertex + 1) % 3]
        #     near2 = t.v[(r.tvertex + 2) % 3]
        #     near1V = self.vertices[near1]
        #     near2V = self.vertices[near2]
        #     D2 = D2 + self.angleBetweenVectors(v2.p, near1V.p, near2V.p)
        #     S2 = S2 + self.crossProductMagnitude(v2.p, near1V.p, near2V.p)
        # K2 = 3 * (2 * 3.14159265358979323846 - D2) / S2
        # K = abs(K2 + K1)
        xishu = 1 #- math.exp(-3 * K)
        q = self.vertices[id_v1].q + self.vertices[id_v2].q
        border = self.vertices[id_v1] and self.vertices[id_v2]
        error = 0
        det = q.det(0, 1, 2, 1, 4, 5, 2, 5, 7)
        if det != 0 and not border:
            p_result.x = -1 / det * (q.det(1, 2, 3, 4, 5, 6, 5, 7, 8))
            p_result.y = 1 / det * (q.det(0, 2, 3, 1, 5, 6, 2, 7, 8))
            p_result.z = -1 / det * (q.det(0, 1, 3, 1, 4, 6, 2, 5, 8))
            error = xishu * self.vertex_error(q, p_result.x, p_result.y, p_result.z)
        else:
            p1 = self.vertices[id_v1].p
            p2 = self.vertices[id_v2].p
            p3 = (p1 + p2) / 2
            error1 = xishu * self.vertex_error(q, p1.x, p1.y, p1.z)
            error2 = xishu * self.vertex_error(q, p2.x, p2.y, p2.z)
            error3 = xishu * self.vertex_error(q, p3.x, p3.y, p3.z)
            error = min(error1, error2, error3)
            if error == error1:
                best_point = p1
            elif error == error2:
                best_point = p2
            elif error == error3:
                best_point = p3
            else:
                best_point = p_result

                # 更新 p_result 的属性
            p_result.x = best_point.x
            p_result.y = best_point.y
            p_result.z = best_point.z

        return error

    def flipped(self, p, i0, i1, v0, v1, deleted):
        for k in range(v0.tcount):
            t = self.triangles[self.refs[v0.tstart + k].tid]
            if t.deleted:
                continue
            s = self.refs[v0.tstart + k].tvertex
            id1 = t.v[(s + 1) % 3]
            id2 = t.v[(s + 2) % 3]
            if id1 == i1 or id2 == i1:
                deleted[k] = 1
                continue
            d1 = self.vertices[id1].p - p
            d1.normalize()
            d2 = self.vertices[id2].p - p
            d2.normalize()
            if abs(d1.dot(d2)) > 0.999:
                return True
            n = d1.cross(d2)
            n.normalize()
            deleted[k] = 0
            if n.dot(t.n) < 0.2:
                return True
        return False

    def update_uvs(self, i0, v, v2, p, deleted):
        for k in range(v.tcount):
            r = self.refs[v.tstart + k]
            t = self.triangles[r.tid]
            r2 = self.refs[v2.tstart + k]
            t2 = self.triangles[r2.tid]
            if t.deleted:
                continue
            if deleted[k]:
                continue
            t.uvs[r.tvertex] = (t.uvs[r.tvertex] + t2.uvs[r2.tvertex]) / 2

    def update_triangles(self, i0, v, deleted, deleted_triangles):
        p = Vec3f(0, 0, 0)
        for k in range(v.tcount):
            r = self.refs[v.tstart + k]
            t = self.triangles[r.tid]
            if t.deleted:
                continue
            if deleted[k]:
                t.deleted = 1
                deleted_triangles[0] = deleted_triangles[0] + 1
                continue
            t.v[r.tvertex] = i0
            t.dirty = 1
            t.err[0] = self.calculate_error(t.v[0], t.v[1], p)
            t.err[1] = self.calculate_error(t.v[1], t.v[2], p)
            t.err[2] = self.calculate_error(t.v[2], t.v[0], p)
            t.err[3] = min(t.err[0], t.err[1], t.err[2])
            self.refs.append(r)
        return

    def update_mesh(self, iteration):
        if (iteration > 0):
            dst = 0
            for i in range(len(self.triangles)):
                if not self.triangles[i].deleted:
                    self.triangles[dst] = self.triangles[i]
                    dst = dst + 1
            self.triangles = [Triangle()]*dst
        for i in range(len(self.vertices)):
            self.vertices[i].tstart = 0
            self.vertices[i].tcount = 0
        for i in range(len(self.triangles)):
            t = self.triangles[i]
            for j in range(3):
                self.vertices[t.v[j]].tcount = self.vertices[t.v[j]].tcount + 1
        tstart = 0
        for i in range(len(self.vertices)):
            v = self.vertices[i]
            v.tstart = tstart
            tstart = tstart + v.tcount
            v.tcount = 0
        self.refs = [Ref(0,0) for _ in range(len(self.triangles)*3)]

        for i in range(len(self.triangles)):
            t = self.triangles[i]
            for j in range(3):
                v = self.vertices[t.v[j]]
                self.refs[v.tstart + v.tcount].tid = i
                self.refs[v.tstart + v.tcount].tvertex = j
                v.tcount = v.tcount + 1
        # for i in range(len(self.refs)):
        #     print(self.refs[i].tid," ",self.refs[i].tvertex)


        if iteration == 0:
            vcount = []
            vids = []
            for i in range(len(self.vertices)):
                self.vertices[i].border = 0
            for i in range(len(self.vertices)):
                v = self.vertices[i]
                vcount.clear()
                vids.clear()
                for j in range(v.tcount):
                    k = self.refs[v.tstart + j].tid
                    t = self.triangles[k]
                    for k in range(3):
                        ofs = 0
                        id = t.v[k]
                        while ofs < len(vcount):
                            if vids[ofs] == id:
                                break
                            ofs = ofs + 1
                        if ofs == len(vcount):
                            vcount.append(1)
                            vids.append(id)
                        else:
                            vcount[ofs] = vcount[ofs] + 1
                for j in range(len(vcount)):
                    if vcount[j] == 1:
                        self.vertices[vids[j]].border = 1

        if iteration == 0:
            for i in range(len(self.vertices)):
                self.vertices[i].q = SymetricMatrix(0.0)
            for i in range(len(self.triangles)):
                t = self.triangles[i]
                p = []
                for j in range(3):
                    p.append(self.vertices[t.v[j]].p)
                n = (p[1] - p[0]).cross(p[2] - p[0])
                n.normalize()
                t.n = n
                for j in range(3):
                    self.vertices[t.v[j]].q = self.vertices[t.v[j]].q + SymetricMatrix(n.x, n.y, n.z, -n.dot(p[0]))

            # for i in range(len(self.triangles)):
            #     print("triangles")
            #     print(self.triangles[i].v[0]," ",self.triangles[i].v[1]," ",self.triangles[i].v[2])
            # for i in range(len(self.vertices)):
            #     print("vertices")
            #     print(self.vertices[i].p.x," ",self.vertices[i].p.y," ",self.vertices[i].p.z)
            for i in range(len(self.triangles)):
                t = self.triangles[i]
                p = Vec3f(0, 0, 0)
                for j in range(3):
                    t.err[j] = self.calculate_error(t.v[j], t.v[(j + 1) % 3], p)
                t.err[3] = min(t.err[0], t.err[1], t.err[2])


        return

    def compact_mesh(self):

        dst = 0
        for i in range(len(self.vertices)):
            self.vertices[i].tcount = 0
        for i in range(len(self.triangles)):
            if not self.triangles[i].deleted:
                t = self.triangles[i]
                self.triangles[dst] = t
                dst = dst + 1
                for j in range(3):
                    self.vertices[t.v[j]].tcount = 1
        self.triangles = self.triangles[:dst]
        dst = 0
        for i in range(len(self.vertices)):
            if self.vertices[i].tcount:
                self.vertices[i].tstart = dst
                self.vertices[dst].p = self.vertices[i].p
                dst = dst + 1
        for i in range(len(self.triangles)):
            t = self.triangles[i]
            for j in range(3):
                t.v[j] = self.vertices[t.v[j]].tstart
        self.vertices = self.vertices[:dst]
        return

    def crossProductMagnitude(self, A, B, C):
        ABx = B.x - A.x
        ABy = B.y - A.y
        ABz = B.z - A.z
        ACx = C.x - A.x
        ACy = C.y - A.y
        ACz = C.z - A.z
        crossX = ABy * ACz - ABz * ACy
        crossY = ABz * ACx - ABx * ACz
        crossZ = ABx * ACy - ABy * ACx
        return math.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ)

    def angleBetweenVectors(self, A, B, C):
        ABx = B.x - A.x
        ABy = B.y - A.y
        ABz = B.z - A.z
        ACx = C.x - A.x
        ACy = C.y - A.y
        ACz = C.z - A.z
        dotProduct = ABx * ACx + ABy * ACy + ABz * ACz
        ABmag = math.sqrt(ABx * ABx + ABy * ABy + ABz * ABz)
        ACmag = math.sqrt(ACx * ACx + ACy * ACy + ACz * ACz)
        angleRAdians = math.acos(dotProduct / (ABmag * ACmag))
        return angleRAdians

    def simplify_mesh(self, target_count, update_rate=5, agressiveness=7, verbose=True, max_iterations=100, alpha=1e-9,
                      K=3,
                      lossless=True, threshold_lossless=1e-4, preserve_border=False):
        face_start = len(self.triangles)
        for i in range(len(self.triangles)):
            self.triangles[i].deleted = 0
        deleted_triangles = [0]
        deleted0 = []
        deleted1 = []
        triangle_count = len(self.triangles)
        self.collapses.clear()
        for iteration in range(max_iterations):
            if triangle_count - deleted_triangles[0] <= target_count:
                break
            if iteration % update_rate == 0 or lossless:
                self.update_mesh(iteration)
            for i in range(len(self.triangles)):
                self.triangles[i].dirty = 0
            threshold = alpha * pow(float(iteration + K), agressiveness)
            if lossless:
                threshold = threshold_lossless
            if verbose and iteration % update_rate == 0:
                print(
                    "iteration {} - triangles {} threshold {}".format(iteration, triangle_count - deleted_triangles[0],
                                                                      threshold))
            for i in range(len(self.triangles)):
                t = self.triangles[i]
                if t.err[3] > threshold:
                    continue
                if t.deleted:
                    continue
                if t.dirty:
                    continue
                for j in range(3):
                    if t.err[j] < threshold:
                        i0 = t.v[j]
                        v0 = self.vertices[i0]
                        i1 = t.v[(j + 1) % 3]
                        v1 = self.vertices[i1]
                        if preserve_border:
                            if v0.border or v1.border:
                                continue
                        else:
                            if v0.border != v1.border:
                                continue
                        p = Vec3f(0, 0, 0)
                        self.calculate_error(i0, i1, p)
                        deleted0 = [0 for _ in range(v0.tcount)]
                        deleted1 = [0 for _ in range(v1.tcount)]
                        if self.flipped(p, i0, i1, v0, v1, deleted0) or self.flipped(p, i1, i0, v1, v0, deleted1):
                            continue
                        if t.attr & Attributes.TEXCOORD.value == Attributes.TEXCOORD.value:
                            if t.attr == 0:
                                self.update_uvs(i0, v0, v1, p, deleted0)
                        v0.p = p
                        v0.q = v0.q + v1.q
                        tstart = len(self.refs)
                        self.update_triangles(i0, v0, deleted0, deleted_triangles)

                        self.update_triangles(i0, v1, deleted1, deleted_triangles)



                        self.collapses.append([i0, i1])
                        tcount = len(self.refs) - tstart
                        if tcount == v0.tcount:
                            if tcount:
                                self.refs[v0.tstart:v0.tstart + tcount] = self.refs[tstart:tstart + tcount]
                        else:
                            v0.tstart = tstart
                        v0.tcount = tcount
                        break
                if lossless and deleted_triangles[0] <= 0:
                    break
                elif not lossless and triangle_count - deleted_triangles[0] <= target_count:
                    break

        self.compact_mesh()
        face_end = len(self.triangles)
        print("from {} to {}".format(face_start, face_end))
        return

    def trimwhitespace(self, str):
        return str.strip()

    def load_obj(self, filename, process_uv=False):
        self.vertices.clear()
        self.triangles.clear()
        vertex_cnt = 0
        material = -1
        uvs = []
        uvmap = []
        material_map = {}
        if filename is None or len(filename) == 0:
            return
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("mtllib"):
                        self.mtllib = line[7:].strip()
                    elif line.startswith("usemtl"):
                        self.usemtl = line[7:].strip()
                        if self.usemtl not in material_map:
                            material_map[self.usemtl] = len(self.materials)
                            self.materials.append(self.usemtl)
                        material = material_map[self.usemtl]
                    elif line.startswith("vt"):
                        uv = list(map(float, line[3:].strip().split()))
                        uvs.append(np.array(uv + [0.0] if len(uv) == 2 else uv))
                    elif line.startswith("v") and not line.startswith("vn"):
                        v = list(map(float, line[2:].strip().split()))
                        vertex=Vertex(Vec3f(v[0], v[1], v[2]))
                        self.vertices.append(vertex)
                    intergers = [0, 0, 0, 0, 0, 0, 0, 0, 0]
                    if line.startswith("f"):
                        tri_ok = False
                        has_uv = False

                        parts = re.split(r'[\s/]+', line.strip()[2:])
                        if len(parts) == 3:
                            tri_ok = True
                            intergers[0:3] = [int(part) for part in parts if part.isdigit()]
                        elif len(parts) == 6:
                            tri_ok = True
                            has_uv = True
                            if "\/\/" in line:
                                has_uv = False
                            intergers[0:6] = [int(part) for part in parts if part.isdigit()]
                        elif len(parts) == 9:
                            intergers[0:9] = [int(part) for part in parts if part.isdigit()]
                            tri_ok = True
                            has_uv = True
                        else:
                            print("wrong format")
                            return
                        if tri_ok:
                            t = Triangle()
                            t.v[0] = intergers[0] - 1 - vertex_cnt
                            t.v[1] = intergers[3] - 1 - vertex_cnt
                            t.v[2] = intergers[6] - 1 - vertex_cnt
                            t.attr = 0
                            if process_uv and has_uv:
                                indices = []
                                indices.append(intergers[1] - 1 - vertex_cnt)
                                indices.append(intergers[4] - 1 - vertex_cnt)
                                indices.append(intergers[7] - 1 - vertex_cnt)
                                uvmap.append(indices)
                                t.attr = t.attr | Attributes.TEXCOORD.value
                            t.material = material
                            self.triangles.append(t)
        except FileNotFoundError:
            print(f"File {filename} not found!")
        if process_uv and len(uvs) > 0:
            for i in range(len(self.triangles)):
                for j in range(3):
                    self.triangles[i].uvs[j] = uvs[uvmap[i][j]]
        return

    def write_obj(self):
        mesh=om.TriMesh()
        vhs=[]
        for vertex in self.vertices:
            vh1 = mesh.add_vertex((vertex.p.x, vertex.p.y, vertex.p.z))
            vhs.append(vh1)
        for triangle in self.triangles:
            fh = mesh.add_face(vhs[triangle.v[0]], vhs[triangle.v[1]], vhs[triangle.v[2]])
        return mesh



mesh=MeshSimplifier()
mesh.load_obj("input.obj",process_uv=False)
for i in range(len(mesh.triangles)):
    print(mesh.triangles[i].v[0]," ",mesh.triangles[i].v[1]," ",mesh.triangles[i].v[2])
mesh.simplify_mesh(1400)
ommesh=mesh.write_obj()
om.write_mesh("output.obj",ommesh)
