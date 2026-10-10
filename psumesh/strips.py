"""Greedy triangle stripifier (NvTriStrip-flavoured) for index-less PS2 strips.

Strip convention: triangle k of strip s is (s[k], s[k+1], s[k+2]) for even k and
(s[k+1], s[k], s[k+2]) for odd k, so all emitted triangles keep the source CCW winding.
"""
from collections import defaultdict


def _canon(f):
    i = f.index(min(f))
    return tuple(f[i:] + f[:i])


def _emitted(s, k):
    return (s[k], s[k + 1], s[k + 2]) if k % 2 == 0 else (s[k + 1], s[k], s[k + 2])


def stripify(faces):
    """Return a list of strips (lists of vertex indices) that exactly cover `faces`."""
    faces = [tuple(int(x) for x in f) for f in faces]
    edge_faces = defaultdict(list)
    for i, f in enumerate(faces):
        for j in range(3):
            edge_faces[frozenset((f[j], f[(j + 1) % 3]))].append(i)
    canon = [_canon(list(f)) for f in faces]
    used = [False] * len(faces)

    def unused_nbrs(i):
        f = faces[i]; n = 0
        for j in range(3):
            n += sum(1 for g in edge_faces[frozenset((f[j], f[(j + 1) % 3]))] if g != i and not used[g])
        return n

    def grow(start, rot, mark):
        f = faces[start]
        s = [f[rot], f[(rot + 1) % 3], f[(rot + 2) % 3]]
        taken = {start}
        while True:
            k = len(s) - 2  # index of next triangle
            e = frozenset((s[-2], s[-1]))
            nxt = None
            for g in edge_faces[e]:
                if g in taken or used[g]:
                    continue
                w = [x for x in faces[g] if x not in e]
                if len(w) != 1:
                    continue
                cand = s + [w[0]]
                if _canon(list(_emitted(cand, k))) == canon[g]:
                    nxt = (g, w[0]); break
            if nxt is None:
                break
            taken.add(nxt[0]); s.append(nxt[1])
        if mark:
            for g in taken:
                used[g] = True
        return s

    strips = []
    order = sorted(range(len(faces)), key=unused_nbrs)
    for i in order:
        if used[i]:
            continue
        best = max(range(3), key=lambda r: len(grow(i, r, False)))
        strips.append(grow(i, best, True))
    return strips


def join_strips(strips):
    """Join strips into one using degenerate triangles (repeat last + first, plus a parity pad)."""
    out = []
    for s in strips:
        if out:
            out += [out[-1], s[0]]
            if len(out) % 2 == 1:
                out.append(s[0])
        out += s
    return out


def strips_to_faces(strips, drop_degenerate=True):
    tris = []
    for s in strips:
        for k in range(len(s) - 2):
            t = _emitted(s, k)
            if drop_degenerate and len(set(t)) < 3:
                continue
            tris.append(t)
    return tris
