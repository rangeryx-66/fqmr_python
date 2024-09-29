import math
import random
from enum import Enum,auto

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

