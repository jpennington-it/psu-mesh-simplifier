"""Garland-Heckbert quadric edge-collapse decimation with attribute, seam and skin awareness."""
import heapq
from dataclasses import dataclass
import numpy as np
from .mesh import Mesh, limit_weights
from .topology import boundary_edges


@dataclass
class SimplifyOptions:
    lock_boundaries: bool = True   # lock border + UV/normal seam vertices (split-vertex boundaries)
    boundary_penalty: float = 1e3  # used when lock_boundaries=False: cost multiplier for boundary verts
    uv_weight: float = 1.0         # attribute cost weights (scaled by edge length^2)
    normal_weight: float = 0.5
    bone_weight: float = 2.0       # penalty for collapsing across different dominant bones
    weight_diff_weight: float = 0.5  # penalty for differing skin weights in general
    max_normal_flip_cos: float = 0.3  # reject collapse if any face normal turns past this (cos)
    max_bones: int = 2


def _plane_quadrics(pos, faces):
    Q = np.zeros((len(pos), 4, 4))
    for f in faces:
        p0, p1, p2 = pos[f]
        n = np.cross(p1 - p0, p2 - p0)
        area = np.linalg.norm(n)
        if area < 1e-20:
            continue
        n /= area
        plane = np.append(n, -n @ p0)
        K = np.outer(plane, plane) * (area * 0.5)
        for v in f:
            Q[v] += K
    return Q


def _skin_vec(ids, w):
    d = {}
    for b, x in zip(ids, w):
        d[int(b)] = d.get(int(b), 0) + float(x)
    return d


class _Decimator:
    def __init__(self, mesh: Mesh, opt: SimplifyOptions):
        self.o = opt
        self.P = mesh.positions.copy(); self.UV = mesh.uvs.copy(); self.N = mesh.normals.copy()
        self.BI = mesh.bone_ids.copy(); self.BW = mesh.bone_weights.copy()
        self.faces = [list(map(int, f)) for f in mesh.faces]
        self.alive_f = [True] * len(self.faces)
        self.n_alive = len(self.faces)
        nv = len(self.P)
        self.vf = [set() for _ in range(nv)]
        for i, f in enumerate(self.faces):
            for v in f:
                self.vf[v].add(i)
        self.alive_v = [True] * nv
        self.ver = [0] * nv
        self.locked = np.zeros(nv, bool)
        bedges = boundary_edges(mesh.faces)
        self.bverts = np.zeros(nv, bool)
        for a, b in bedges:
            self.bverts[a] = self.bverts[b] = True
        if opt.lock_boundaries:
            self.locked[:] = self.bverts
        self.Q = _plane_quadrics(self.P, mesh.faces)
        self.heap = []
        for a, b in {(min(f[i], f[(i + 1) % 3]), max(f[i], f[(i + 1) % 3]))
                     for f in self.faces for i in range(3)}:
            self._push(a, b)

    def _nbrs(self, v):
        s = set()
        for fi in self.vf[v]:
            s.update(self.faces[fi])
        s.discard(v)
        return s

    def _cost(self, a, b):
        o = self.o
        if self.locked[a] and self.locked[b]:
            return None
        Q = self.Q[a] + self.Q[b]
        pa, pb = self.P[a], self.P[b]
        if self.locked[a]:
            cands = [0.0]
        elif self.locked[b]:
            cands = [1.0]
        else:
            cands = [0.0, 1.0, 0.5]
            A = Q[:3, :3]
            if abs(np.linalg.det(A)) > 1e-12:
                x = np.linalg.solve(A, -Q[:3, 3])
                d = pb - pa
                dd = d @ d
                if dd > 0:
                    t = float(np.clip((x - pa) @ d / dd, 0, 1))
                    cands.append(t)
        best = None
        for t in cands:
            p = pa + t * (pb - pa)
            vh = np.append(p, 1.0)
            err = max(float(vh @ Q @ vh), 0.0)
            if best is None or err < best[0]:
                best = (err, t)
        err, t = best
        L2 = float((pb - pa) @ (pb - pa))
        duv = self.UV[a] - self.UV[b]
        attr = o.uv_weight * float(duv @ duv) + o.normal_weight * (1 - float(self.N[a] @ self.N[b]))
        sa = _skin_vec(self.BI[a], self.BW[a]); sb = _skin_vec(self.BI[b], self.BW[b])
        wdiff = sum(abs(sa.get(k, 0) - sb.get(k, 0)) for k in set(sa) | set(sb)) * 0.5
        attr += o.weight_diff_weight * wdiff
        if self.BI[a][np.argmax(self.BW[a])] != self.BI[b][np.argmax(self.BW[b])]:
            attr += o.bone_weight
        cost = err + attr * L2
        if not o.lock_boundaries and (self.bverts[a] or self.bverts[b]):
            cost = cost * o.boundary_penalty + o.boundary_penalty * L2
        return cost, t

    def _push(self, a, b):
        r = self._cost(a, b)
        if r is not None:
            heapq.heappush(self.heap, (r[0], a, b, self.ver[a], self.ver[b], r[1]))

    def _try_collapse(self, a, b, t):
        # keep = a, remove = b
        shared = self.vf[a] & self.vf[b]
        if not shared:
            return False
        # link condition (keeps mesh manifold)
        third = set()
        for fi in shared:
            third.update(self.faces[fi])
        third -= {a, b}
        if (self._nbrs(a) & self._nbrs(b)) != third:
            return False
        newp = self.P[a] + t * (self.P[b] - self.P[a])
        # flip / degeneracy check
        for v in (a, b):
            for fi in self.vf[v] - shared:
                f = self.faces[fi]
                p = [self.P[x] for x in f]
                n0 = np.cross(p[1] - p[0], p[2] - p[0])
                q = [newp if x in (a, b) else self.P[x] for x in f]
                n1 = np.cross(q[1] - q[0], q[2] - q[0])
                l0, l1 = np.linalg.norm(n0), np.linalg.norm(n1)
                if l1 < 1e-12 * max(l0, 1e-30) or l1 == 0:
                    return False
                if l0 > 0 and (n0 @ n1) / (l0 * l1) < self.o.max_normal_flip_cos:
                    return False
        # apply
        self.P[a] = newp
        self.UV[a] = self.UV[a] + t * (self.UV[b] - self.UV[a])
        n = self.N[a] + t * (self.N[b] - self.N[a]); ln = np.linalg.norm(n)
        self.N[a] = n / ln if ln > 0 else self.N[a]
        ids = np.concatenate([self.BI[a], self.BI[b]])
        w = np.concatenate([self.BW[a] * (1 - t), self.BW[b] * t])
        bi, bw = limit_weights(ids[None], w[None], self.o.max_bones)
        self.BI[a], self.BW[a] = bi[0], bw[0]
        self.Q[a] = self.Q[a] + self.Q[b]
        self.locked[a] = self.locked[a] or self.locked[b]
        self.bverts[a] = self.bverts[a] or self.bverts[b]
        for fi in shared:
            self.alive_f[fi] = False; self.n_alive -= 1
            for v in self.faces[fi]:
                self.vf[v].discard(fi)
        for fi in list(self.vf[b]):
            f = self.faces[fi]
            f[f.index(b)] = a
            self.vf[a].add(fi)
        self.vf[b] = set(); self.alive_v[b] = False
        self.ver[a] += 1; self.ver[b] += 1
        for nb in self._nbrs(a):  # only edges touching `a` changed cost
            self._push(min(a, nb), max(a, nb))
        return True

    def run(self, target_faces):
        while self.n_alive > target_faces and self.heap:
            cost, a, b, va, vb, t = heapq.heappop(self.heap)
            if not (self.alive_v[a] and self.alive_v[b]) or va != self.ver[a] or vb != self.ver[b]:
                continue
            if self.locked[a] and not self.locked[b]:
                pass  # t==0 => keep a's position
            elif self.locked[b] and not self.locked[a]:
                a, b, t = b, a, 0.0  # keep the locked vertex
            self._try_collapse(a, b, t)
        return self.result()

    def result(self):
        faces = [self.faces[i] for i in range(len(self.faces)) if self.alive_f[i]]
        used = sorted({v for f in faces for v in f})
        remap = {v: i for i, v in enumerate(used)}
        F = np.array([[remap[v] for v in f] for f in faces], np.int64).reshape(-1, 3)
        u = np.array(used, np.int64)
        return Mesh(self.P[u], F, self.UV[u], self.N[u], self.BI[u], self.BW[u])


def simplify(mesh: Mesh, target_faces: int, options: SimplifyOptions = None) -> Mesh:
    """Decimate `mesh` toward `target_faces` triangles. May stop above the target when
    every remaining collapse would break a seam/border lock, flip a face or go non-manifold."""
    return _Decimator(mesh, options or SimplifyOptions()).run(max(int(target_faces), 0))
