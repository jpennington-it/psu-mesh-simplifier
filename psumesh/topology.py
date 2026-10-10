import numpy as np
from collections import Counter


def edge_counts(faces):
    c = Counter()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            c[(min(a, b), max(a, b))] += 1
    return c


def boundary_edges(faces):
    """Edges used by exactly one face in the split-vertex mesh. These are true mesh
    borders AND UV/normal seams (split vertices make seams look like borders)."""
    return {e for e, n in edge_counts(faces).items() if n == 1}


def position_key(p, tol=1e-6):
    return tuple(np.round(np.asarray(p) / tol).astype(np.int64))


def seam_edges_geometric(mesh, tol=1e-6):
    """Boundary edges as sets of positions (independent of vertex indexing), for tests."""
    out = set()
    for a, b in boundary_edges(mesh.faces):
        ka, kb = position_key(mesh.positions[a], tol), position_key(mesh.positions[b], tol)
        out.add(frozenset((ka, kb)))
    return out


def face_normals(pos, faces):
    return np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
