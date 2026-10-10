"""Pluggable mesh I/O. Implement MeshAdapter and register() it to support a new format."""
import json, os
from abc import ABC, abstractmethod
import numpy as np
from .mesh import Mesh


class MeshAdapter(ABC):
    name = "base"
    extensions = ()

    @abstractmethod
    def read(self, path) -> Mesh: ...

    @abstractmethod
    def write(self, mesh: Mesh, path) -> None: ...


_REGISTRY = {}


def register(adapter: MeshAdapter):
    _REGISTRY[adapter.name] = adapter
    return adapter


def get_adapter(path=None, fmt=None) -> MeshAdapter:
    if fmt:
        return _REGISTRY[fmt]
    ext = os.path.splitext(path)[1].lower()
    for a in _REGISTRY.values():
        if ext in a.extensions:
            return a
    raise ValueError(f"no adapter for {path!r}; known: {list(_REGISTRY)}")


def read_mesh(path, fmt=None): return get_adapter(path, fmt).read(path)
def write_mesh(mesh, path, fmt=None): get_adapter(path, fmt).write(mesh, path)


class JsonAdapter(MeshAdapter):
    """{"positions":[[x,y,z]..], "uvs":[[u,v]..], "normals":[..], "bone_ids":[[b0,b1]..],
        "bone_weights":[[w0,w1]..], "faces":[[i,j,k]..]}  (faces CCW)"""
    name = "json"; extensions = (".json",)

    def read(self, path):
        d = json.load(open(path))
        return Mesh(d["positions"], d["faces"], d.get("uvs"), d.get("normals"),
                    d.get("bone_ids"), d.get("bone_weights"))

    def write(self, mesh, path):
        json.dump({"positions": mesh.positions.tolist(), "uvs": mesh.uvs.tolist(),
                   "normals": mesh.normals.tolist(), "bone_ids": mesh.bone_ids.tolist(),
                   "bone_weights": mesh.bone_weights.tolist(), "faces": mesh.faces.tolist()},
                  open(path, "w"))


class ObjAdapter(MeshAdapter):
    """Wavefront OBJ (v/vt/vn/f) + optional sidecar `<name>.weights.json`:
    {"bone_ids": [[b0,b1]...], "bone_weights": [[w0,w1]...]} indexed by the split vertex
    order this adapter produces (OBJ corners are split per unique v/vt/vn triple)."""
    name = "obj"; extensions = (".obj",)

    @staticmethod
    def sidecar(path): return os.path.splitext(path)[0] + ".weights.json"

    def read(self, path):
        V, VT, VN, key, P, T, N, F = [], [], [], {}, [], [], [], []
        for line in open(path):
            p = line.split()
            if not p: continue
            if p[0] == "v": V.append([float(x) for x in p[1:4]])
            elif p[0] == "vt": VT.append([float(x) for x in p[1:3]])
            elif p[0] == "vn": VN.append([float(x) for x in p[1:4]])
            elif p[0] == "f":
                idx = []
                for c in p[1:]:
                    k = tuple((int(x) - 1) if x else -1 for x in (c.split("/") + ["", ""])[:3])
                    if k not in key:
                        key[k] = len(P); P.append(V[k[0]])
                        T.append(VT[k[1]] if k[1] >= 0 else [0, 0])
                        N.append(VN[k[2]] if k[2] >= 0 else None)
                    idx.append(key[k])
                for i in range(1, len(idx) - 1):
                    F.append([idx[0], idx[i], idx[i + 1]])
        normals = None if any(n is None for n in N) else N
        bi = bw = None
        sc = self.sidecar(path)
        if os.path.exists(sc):
            d = json.load(open(sc)); bi, bw = d["bone_ids"], d["bone_weights"]
        return Mesh(P, F, T, normals, bi, bw)

    def write(self, mesh, path):
        with open(path, "w") as f:
            for p in mesh.positions: f.write("v %.6f %.6f %.6f\n" % tuple(p))
            for t in mesh.uvs: f.write("vt %.6f %.6f\n" % tuple(t))
            for n in mesh.normals: f.write("vn %.6f %.6f %.6f\n" % tuple(n))
            for t in mesh.faces + 1: f.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % tuple(np.repeat(t, 3)))
        json.dump({"bone_ids": mesh.bone_ids.tolist(), "bone_weights": mesh.bone_weights.tolist()},
                  open(self.sidecar(path), "w"))


class PsuPs2Adapter(MeshAdapter):
    """Placeholder for the real PSU PS2 part format (write-up pending from project lead).
    Expected work: parse header -> per-strip vertex runs (48-byte skinned / 22-byte rigid),
    de-strip into triangles with welded split vertices, and on write re-stripify with
    psumesh.strips.stripify and emit vertices in the original layout."""
    name = "psu"; extensions = (".psu-ps2",)

    def read(self, path): raise NotImplementedError("PSU PS2 format write-up pending")
    def write(self, mesh, path): raise NotImplementedError("PSU PS2 format write-up pending")


for _a in (JsonAdapter(), ObjAdapter(), PsuPs2Adapter()):
    register(_a)
