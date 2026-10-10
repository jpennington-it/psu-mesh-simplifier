"""Synthetic skinned test meshes with UV seams and 2-bone weights."""
import numpy as np
from .mesh import Mesh


def _tube(radius_fn, center_fn, segments, rings, bone_split=0.5, blend=0.3, bones=(0, 1)):
    """Open tube along t in [0,1]; duplicated column at u=0/u=1 creates a UV seam.
    Skin: bone0 below bone_split, bone1 above, linear blend over `blend` width."""
    P, UV, BI, BW = [], [], [], []
    for r in range(rings + 1):
        t = r / rings
        c, R = np.asarray(center_fn(t), float), radius_fn(t)
        w1 = float(np.clip((t - (bone_split - blend / 2)) / blend, 0, 1))
        for s in range(segments + 1):  # s == segments duplicates s == 0 (seam)
            a = 2 * np.pi * (s % segments) / segments
            P.append(c + [R * np.cos(a), 0.0, R * np.sin(a)])
            UV.append([s / segments, t])
            if w1 <= 1e-9: BI.append([bones[0], bones[1]]); BW.append([1.0, 0.0])
            elif w1 >= 1 - 1e-9: BI.append([bones[1], bones[0]]); BW.append([1.0, 0.0])
            else: BI.append([bones[0], bones[1]]); BW.append([1 - w1, w1])
    F, W = [], segments + 1
    for r in range(rings):
        for s in range(segments):
            a, b, c, d = r * W + s, r * W + s + 1, (r + 1) * W + s, (r + 1) * W + s + 1
            F += [[a, c, b], [b, c, d]]  # outward-facing CCW
    return Mesh(P, F, UV, None, BI, BW)


def skinned_cylinder(segments=32, rings=24, radius=1.0, height=4.0):
    return _tube(lambda t: radius, lambda t: [0, t * height, 0], segments, rings)


def boot(segments=40, rings=40):
    """Shin tube that bends forward into a foot, ankle-bone blend at the bend."""
    def center(t):
        if t < 0.6:
            return [0, 4.0 - t / 0.6 * 3.0, 0]
        a = (t - 0.6) / 0.4 * (np.pi / 2)  # quarter bend, radius 2 (> tube radius: no self-overlap)
        return [0, 1.0 - np.sin(a) * 2.0, 2.0 - np.cos(a) * 2.0]
    radius = lambda t: 0.8 + 0.12 * np.sin(t * 9) + 0.25 * max(0.0, t - 0.7) / 0.3
    return _tube(radius, center, segments, rings, bone_split=0.6, blend=0.25, bones=(3, 4))


def rigid_tube(segments=24, rings=16):
    m = skinned_cylinder(segments, rings)
    m.bone_ids[:] = [[7, 7]]; m.bone_weights[:] = [[1.0, 0.0]]
    return m
