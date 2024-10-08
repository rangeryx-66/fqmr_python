import math
import random
from enum import Enum,auto
import re
import numpy as np
import openmesh as om
import numpy
import math
import random
from scipy import constants
class SymetricMatrix:
    def __init__(self, *args):
        if len(args) == 1:
            self.m = [args[0]] * 10
        elif len(args) == 10:
            self.m = list(args)
        elif len(args) == 4:
            self.m = [args[0] * args[0], args[0] * args[1], args[0] * args[2], args[0] * args[3],
                      args[1] * args[1], args[1] * args[2], args[1] * args[3],
                      args[2] * args[2], args[2] * args[3],
                      args[3] * args[3]]
        else:
            raise ValueError("Invalid number of arguments for SymetricMatrix constructor")

    def __getitem__(self, index):
        return self.m[index]

    def det(self, a11, a12, a13, a21, a22, a23, a31, a32, a33):
        det = (self.m[a11] * self.m[a22] * self.m[a33]
               + self.m[a13] * self.m[a21] * self.m[a32]
               + self.m[a12] * self.m[a23] * self.m[a31]
               - self.m[a13] * self.m[a22] * self.m[a31]
               - self.m[a11] * self.m[a23] * self.m[a32]
               - self.m[a12] * self.m[a21] * self.m[a33])
        return det

    def __add__(self, other):
        return SymetricMatrix(*[x + y for x, y in zip(self.m, other.m)])

    def __iadd__(self, other):
        self.m = [x + y for x, y in zip(self.m, other.m)]
        return self


class Vec3f:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other):
        return Vec3f(self.x + other.x, self.y + other.y, self.z + other.z)

    def __iadd__(self, other):
        self.x += other.x
        self.y += other.y
        self.z += other.z
        return self

    def __mul__(self, other):
        if isinstance(other, Vec3f):
            return Vec3f(self.x * other.x, self.y * other.y, self.z * other.z)
        else:
            return Vec3f(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        if isinstance(other, Vec3f):
            return Vec3f(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            return Vec3f(self.x / other, self.y / other, self.z / other)

    def __sub__(self, other):
        return Vec3f(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3f(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self):
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def normalize(self, desired_length=1.0):
        len_self = self.length()
        if len_self > 0:
            self.x /= len_self
            self.y /= len_self
            self.z /= len_self
        return self * desired_length

    def angle(self, other):
        dot_product = self.dot(other)
        len_self = self.length()
        len_other = other.length()
        if len_self == 0 or len_other == 0:
            return 0.0
        cos_angle = max(-1, min(1, dot_product / (len_self * len_other)))
        return math.acos(cos_angle)

    def angle2(self, v, w):
        dot_product = v.dot(self)
        len_v = v.length()
        len_self = self.length()
        if len_v == 0 or len_self == 0:
            return 0.0

        plane = self.cross(w)
        if plane.dot(v) > 0:
            return -math.acos(dot_product / (len_v * len_self))
        return math.acos(dot_product / (len_v * len_self))

    def clamp(self, min_val, max_val):
        self.x = max(min_val, min(max_val, self.x))
        self.y = max(min_val, min(max_val, self.y))
        self.z = max(min_val, min(max_val, self.z))

    def rot_x(self, a):
        yy = math.cos(a) * self.y + math.sin(a) * self.z
        zz = math.cos(a) * self.z - math.sin(a) * self.y
        self.y = yy
        self.z = zz
        return self

    def rot_y(self, a):
        xx = math.cos(-a) * self.x + math.sin(-a) * self.z
        zz = math.cos(-a) * self.z - math.sin(-a) * self.x
        self.x = xx
        self.z = zz
        return self

    def rot_z(self, a):
        yy = math.cos(a) * self.y + math.sin(a) * self.x
        xx = math.cos(a) * self.x - math.sin(a) * self.y
        self.y = yy
        self.x = xx
        return self

    def invert(self):
        self.x = -self.x
        self.y = -self.y
        self.z = -self.z
        return self

    def frac(self):
        return Vec3f(self.x - int(self.x), self.y - int(self.y), self.z - int(self.z))

    def integer(self):
        return Vec3f(float(int(self.x)), float(int(self.y)), float(int(self.z)))

    @staticmethod
    def random():
        return Vec3f(random.random(), random.random(), random.random())

    def random_double_01(self, a):
        rnf = (a * 14.434252 + a * 364.2343 +
               a * 4213.45352 + a * 2341.43255 +
               a * 254341.43535 + a * 223454341.3523534245 +
               23453.423412)
        rni = int(rnf) % 100000
        return float(rni) / (100000.0 - 1.0)

    def random01_fxyz(self):
        self.x = self.random_double_01(self.x)
        self.y = self.random_double_01(self.y)
        self.z = self.random_double_01(self.z)
        return self


class Triangle:
    def __init__(self, v0=0, v1=0, v2=0, err0=0., err1=0., err2=0., err3=0., deleted=0, dirty=0, attr=0, n=Vec3f(0,0,0), uvs0=Vec3f(0,0,0), uvs1=Vec3f(0,0,0), uvs2=Vec3f(0,0,0), material=-114514):
        self.v = [v0, v1, v2]
        self.err = [err0, err1, err2, err3]
        self.deleted = deleted
        self.dirty = dirty
        self.attr = attr
        self.n = n
        self.uvs = [uvs0, uvs1, uvs2]
        self.material = material


class Attributes(Enum):
    NONE = 0
    NORMAL = 2
    TEXCOORD = 4
    COLOR = 8


class Vertex:
    def __init__(self, p, tstart=0, tcount=0, q=SymetricMatrix(0), border=0):
        self.p = p
        self.tstart = tstart
        self.tcount = tcount
        self.q = q
        self.border = border


class Ref:
    def __init__(self, tid, tvertex):
        self.tid = tid
        self.tvertex = tvertex


def barycentric(p, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = v0.dot(v0)
    d01 = v0.dot(v1)
    d11 = v1.dot(v1)
    d20 = v2.dot(v0)
    d21 = v2.dot(v1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - v - w
    return Vec3f(u, v, w)


def interpolate(p, a, b, c, attrs):
    bary = barycentric(p, a, b, c)
    out = Vec3f(0, 0, 0)
    out = out + attrs[0] * bary.x
    out = out + attrs[1] * bary.y
    out = out + attrs[2] * bary.z
    return out





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

    # def crossProductMagnitude(self, A, B, C):
    #     ABx = B.x - A.x
    #     ABy = B.y - A.y
    #     ABz = B.z - A.z
    #     ACx = C.x - A.x
    #     ACy = C.y - A.y
    #     ACz = C.z - A.z
    #     crossX = ABy * ACz - ABz * ACy
    #     crossY = ABz * ACx - ABx * ACz
    #     crossZ = ABx * ACy - ABy * ACx
    #     return math.sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ)
    #
    # def angleBetweenVectors(self, A, B, C):
    #     ABx = B.x - A.x
    #     ABy = B.y - A.y
    #     ABz = B.z - A.z
    #     ACx = C.x - A.x
    #     ACy = C.y - A.y
    #     ACz = C.z - A.z
    #     dotProduct = ABx * ACx + ABy * ACy + ABz * ACz
    #     ABmag = math.sqrt(ABx * ABx + ABy * ABy + ABz * ABz)
    #     ACmag = math.sqrt(ACx * ACx + ACy * ACy + ACz * ACz)
    #     angleRAdians = math.acos(dotProduct / (ABmag * ACmag))
    #     return angleRAdians

    def simplify_mesh(self, target_count, update_rate=5, agressiveness=7, verbose=True, max_iterations=100, alpha=1e-9,
                      K=3,
                      lossless=False, threshold_lossless=1e-4, preserve_border=False):
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

    def write_obj(self,filename='output.obj'):
        with open(filename, 'w') as file:
            cur_material = -1
            has_uv = (len(self.triangles) and (self.triangles[0].attr & Attributes.TEXCOORD.value) == Attributes.TEXCOORD.value)

            if not file:
                print(f"write_obj: can't write data file \"{filename}\".")
                exit(0)

            if self.mtllib:
                file.write(f"mtllib {self.mtllib}\n")

            for i, vertex in enumerate(self.vertices):
                # More compact: remove trailing zeros
                file.write(f"v {vertex.p.x:g} {vertex.p.y:g} {vertex.p.z:g}\n")

            if has_uv:
                for triangle in self.triangles:
                    if not triangle.deleted:
                        for uv in triangle.uvs:
                            file.write(f"vt {uv.x:g} {uv.y:g}\n")

            uv = 1
            for triangle in self.triangles:
                if not triangle.deleted:
                    if triangle.material != cur_material:
                        cur_material = triangle.material
                        file.write(f"usemtl {self.materials[triangle.material]}\n")

                    if has_uv:
                        file.write(
                            f"f {triangle.v[0] + 1}/{uv} {triangle.v[1] + 1}/{uv + 1} {triangle.v[2] + 1}/{uv + 2}\n")
                        uv += 3
                    else:
                        file.write(f"f {triangle.v[0] + 1} {triangle.v[1] + 1} {triangle.v[2] + 1}\n")



mesh=MeshSimplifier()
mesh.load_obj("input.obj",process_uv=False)
for i in range(len(mesh.triangles)):
    print(mesh.triangles[i].v[0]," ",mesh.triangles[i].v[1]," ",mesh.triangles[i].v[2])
mesh.simplify_mesh(1600)
mesh.write_obj("output.obj")
