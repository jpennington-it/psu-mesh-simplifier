"""PS2 byte estimation and byte-budget fitting."""
from dataclasses import dataclass
from .strips import stripify, join_strips
from .simplify import simplify, SimplifyOptions

PSU_CHAR_BLOCK = 0x34800  # 215,040 bytes


@dataclass
class ByteModel:
    skinned_vertex_bytes: int = 48
    rigid_vertex_bytes: int = 22
    per_strip_bytes: int = 0      # header/GIF tag overhead per strip (set once format is known)
    fixed_bytes: int = 0          # per-part header/material/texture refs
    join_mode: str = "break"      # "break" = separate strips, "degenerate" = one joined strip
    force_skinned: bool = False


def estimate_bytes(mesh, model: ByteModel = None, strips=None):
    m = model or ByteModel()
    strips = strips if strips is not None else stripify(mesh.faces)
    if m.join_mode == "degenerate":
        nverts = len(join_strips(strips)); nstrips = 1 if strips else 0
    else:
        nverts = sum(len(s) for s in strips); nstrips = len(strips)
    vb = m.skinned_vertex_bytes if (m.force_skinned or not mesh.is_rigid()) else m.rigid_vertex_bytes
    total = m.fixed_bytes + nverts * vb + nstrips * m.per_strip_bytes
    return {"bytes": total, "strip_vertices": nverts, "strips": nstrips,
            "vertex_bytes": vb, "triangles": int(mesh.n_faces)}


def fit_to_budget(mesh, budget_bytes, model: ByteModel = None, options: SimplifyOptions = None,
                  min_faces=4):
    """Binary-search the largest triangle target whose stripified size fits the budget.
    Returns (mesh, report). report['fits'] is False if even max simplification doesn't fit."""
    model = model or ByteModel()
    orig = estimate_bytes(mesh, model)
    if orig["bytes"] <= budget_bytes:
        return mesh, {**orig, "fits": True, "original": orig, "iterations": 0}
    lo, hi = min_faces, mesh.n_faces  # search target in [lo, hi)
    best = None; it = 0
    while lo < hi:
        mid = (lo + hi) // 2; it += 1
        cand = simplify(mesh, mid, options)
        est = estimate_bytes(cand, model)
        if est["bytes"] <= budget_bytes:
            best = (cand, est); lo = mid + 1
        else:
            hi = mid
    if best is None:
        cand = simplify(mesh, min_faces, options)
        est = estimate_bytes(cand, model)
        return cand, {**est, "fits": est["bytes"] <= budget_bytes, "original": orig, "iterations": it}
    return best[0], {**best[1], "fits": True, "original": orig, "iterations": it}
