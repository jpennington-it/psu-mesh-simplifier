# psu-mesh-simplifier

> **Unofficial fan tool.** Not affiliated with or endorsed by SEGA. This repository contains
> **no SEGA or Clementine assets** (models, textures, or game data); all test meshes are generated
> synthetically by `psumesh/synth.py`. The PSU PS2 format adapter (`PsuPs2Adapter`) is currently a
> **stub** awaiting format documentation. Real parts must be **welded** on import (see below).

Mesh decimator for the PSU Clementine -> PS2 outfit port. Each character's body parts must fit the
0x34800 (215,040-byte) block, and the PS2 format has no index buffer: every triangle-strip vertex is
stored in full (48 bytes skinned, 22 bytes rigid single-bone). This tool shrinks oversized parts
just enough to fit a byte budget.

Pure Python + numpy. No other runtime deps (pytest for tests).

## What it does
- **QEM edge collapse** (Garland-Heckbert plane quadrics, area-weighted), picking the best of
  endpoints, midpoint, or the optimum clamped to the edge.
- **Attribute-aware cost**: UV distance, normal deviation, skin-weight difference, plus an extra
  penalty for collapsing across different dominant bones. All scaled by edge length squared.
- **Seams/borders**: the mesh is split-vertex, so UV/normal seams show up as boundary edges. Those
  vertices are locked by default and never move. `--no-lock-seams` penalizes them instead.
- **Winding/topology**: collapses that rotate any face normal past `max_normal_flip_cos` (default
  0.3, about 72°), that make degenerate faces, or that fail the link condition (non-manifold) are rejected.
- **Skinning**: weights are interpolated, merged per bone, cut to the top 2, and renormalized.
- **Stripifier**: a greedy, NvTriStrip-style stripifier. It starts from the face with the fewest free
  neighbours, tries all 3 rotations, and keeps winding correct with the even/odd convention. Strips
  can be kept separate (`--join break`) or merged with degenerate triangles (`--join degenerate`).
- **Byte budget**: `ByteModel` (48/22 bytes per vertex, plus optional per-strip and fixed overhead)
  and a binary search over the target triangle count.

## Usage
```
python3 -m venv .venv && .venv/bin/pip install numpy pytest
.venv/bin/python -m psumesh synth boot boot.obj               # synthetic skinned boot (+ boot.weights.json)
.venv/bin/python -m psumesh estimate boot.obj                  # stripify + PS2 byte estimate
.venv/bin/python -m psumesh simplify boot.obj out.json --budget 60000
.venv/bin/python -m psumesh simplify boot.obj out.obj --ratio 0.3 --join degenerate --per-strip-bytes 16
.venv/bin/python -m pytest -q
```
Exit code 2 means the budget couldn't be met because locked seams/borders limit how far it can reduce.

Python API: `simplify(mesh, target_faces, SimplifyOptions(...))`, `fit_to_budget(mesh, bytes, ByteModel(...))`,
`stripify(faces)`, `estimate_bytes(mesh, model)`.

## Formats and the adapter interface
Built in: `.json` (positions/uvs/normals/bone_ids/bone_weights/faces) and `.obj` with a sidecar
`<name>.weights.json` (`bone_ids`, `bone_weights` per split vertex, in the order the OBJ reader
produces, which is the order the OBJ writer emits).

To add the real PSU PS2 part format, fill in `PsuPs2Adapter` in `psumesh/io.py` (it's a stub for now),
or subclass `MeshAdapter`:
```python
from psumesh.io import MeshAdapter, register
from psumesh import Mesh
class MyFormat(MeshAdapter):
    name = "myfmt"; extensions = (".bin",)
    def read(self, path) -> Mesh: ...   # de-strip to triangles (CCW), weld identical vertices, max 2 bones
    def write(self, mesh, path): ...    # psumesh.strips.stripify(mesh.faces) -> emit 48/22-byte verts
register(MyFormat())
```
When reading, weld bit-identical strip vertices back into shared vertices. Otherwise every triangle
looks like a border and gets locked, and nothing simplifies. Once the header/GIF-tag overhead is
known, set `ByteModel.per_strip_bytes` / `fixed_bytes`.

## Limitations
- PSU PS2 adapter is a stub until the format write-up is available; importers must weld strip vertices.
- Locked seams/borders put a floor on reduction. Heavily seamed parts may not reach very tight budgets.
- Byte size isn't strictly monotonic in the triangle count, so the binary search finds a good fit,
  not a guaranteed maximum.
- After decimation the stripifier gets about 1.4 strip verts per triangle, versus about 1.0 on clean grids.
  Strip quality could still be improved (backward extension, swaps).
- Pure Python. A 3k-tri part simplifies in about 1 s; budget mode re-runs it about 12 times.
- There's no texture/material awareness beyond UVs, and no per-vertex bone-palette limits.

## License
MIT, see `LICENSE`. Phantasy Star Universe is a trademark of SEGA; this project is independent.
