"""Generic in-memory mesh (split-vertex: one vertex per unique pos/uv/normal/weights)."""
from dataclasses import dataclass
import numpy as np

MAX_BONES = 2


@dataclass
class Mesh:
    positions: np.ndarray          # (N,3) float
    faces: np.ndarray              # (M,3) int, CCW front faces
    uvs: np.ndarray = None         # (N,2)
    normals: np.ndarray = None     # (N,3)
    bone_ids: np.ndarray = None    # (N,2) int
    bone_weights: np.ndarray = None  # (N,2) float, rows sum to 1

    def __post_init__(self):
        self.positions = np.asarray(self.positions, float).reshape(-1, 3)
        self.faces = np.asarray(self.faces, np.int64).reshape(-1, 3)
        n = len(self.positions)
        self.uvs = np.zeros((n, 2)) if self.uvs is None else np.asarray(self.uvs, float).reshape(n, 2)
        if self.normals is None:
            self.normals = compute_vertex_normals(self.positions, self.faces)
        self.normals = np.asarray(self.normals, float).reshape(n, 3)
        if self.bone_ids is None:
            self.bone_ids = np.zeros((n, MAX_BONES), np.int64)
            self.bone_weights = np.zeros((n, MAX_BONES)); self.bone_weights[:, 0] = 1
        self.bone_ids = np.asarray(self.bone_ids, np.int64).reshape(n, -1)
        self.bone_weights = np.asarray(self.bone_weights, float).reshape(n, -1)
        if self.bone_ids.shape[1] != MAX_BONES:
            self.bone_ids, self.bone_weights = limit_weights(self.bone_ids, self.bone_weights)

    @property
    def n_verts(self): return len(self.positions)

    @property
    def n_faces(self): return len(self.faces)

    def is_rigid(self, eps=1e-6):
        """True if every vertex is fully bound to a single bone (22-byte format eligible)."""
        return bool(np.all(self.bone_weights[:, 1:] <= eps))

    def copy(self):
        return Mesh(self.positions.copy(), self.faces.copy(), self.uvs.copy(), self.normals.copy(),
                    self.bone_ids.copy(), self.bone_weights.copy())


def compute_vertex_normals(pos, faces):
    n = np.zeros_like(pos, dtype=float)
    if len(faces):
        fn = np.cross(pos[faces[:, 1]] - pos[faces[:, 0]], pos[faces[:, 2]] - pos[faces[:, 0]])
        for k in range(3):
            np.add.at(n, faces[:, k], fn)
    l = np.linalg.norm(n, axis=1, keepdims=True); l[l == 0] = 1
    return n / l


def limit_weights(ids, w, k=MAX_BONES):
    """Merge duplicate bones, keep top-k influences, renormalize."""
    ids = np.asarray(ids); w = np.asarray(w, float)
    out_i = np.zeros((len(ids), k), np.int64); out_w = np.zeros((len(ids), k))
    for r in range(len(ids)):
        acc = {}
        for b, x in zip(ids[r], w[r]):
            if x > 0: acc[int(b)] = acc.get(int(b), 0) + float(x)
        top = sorted(acc.items(), key=lambda t: -t[1])[:k] or [(int(ids[r][0]), 1.0)]
        s = sum(x for _, x in top) or 1.0
        for j, (b, x) in enumerate(top):
            out_i[r, j] = b; out_w[r, j] = x / s
    return out_i, out_w
