import argparse, json, sys
from .io import read_mesh, write_mesh
from .simplify import simplify, SimplifyOptions
from .budget import ByteModel, estimate_bytes, fit_to_budget, PSU_CHAR_BLOCK
from . import synth


def main(argv=None):
    ap = argparse.ArgumentParser(prog="psu-simplify", description="QEM mesh simplifier with PS2 byte budgeting")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--skinned-bytes", type=int, default=48)
        p.add_argument("--rigid-bytes", type=int, default=22)
        p.add_argument("--per-strip-bytes", type=int, default=0)
        p.add_argument("--fixed-bytes", type=int, default=0)
        p.add_argument("--join", choices=["break", "degenerate"], default="break")
        p.add_argument("--force-skinned", action="store_true")
        p.add_argument("--in-format"); p.add_argument("--out-format")

    s = sub.add_parser("simplify", help="decimate a mesh"); common(s)
    s.add_argument("input"); s.add_argument("output")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--faces", type=int, help="target triangle count")
    g.add_argument("--ratio", type=float, help="target fraction of triangles")
    g.add_argument("--budget", type=lambda x: int(x, 0), help="byte budget (e.g. 0x34800 or 60000)")
    s.add_argument("--no-lock-seams", action="store_true", help="penalize instead of lock seams/borders")
    s.add_argument("--uv-weight", type=float, default=1.0)
    s.add_argument("--normal-weight", type=float, default=0.5)
    s.add_argument("--bone-weight", type=float, default=2.0)

    e = sub.add_parser("estimate", help="stripify and report PS2 bytes"); common(e)
    e.add_argument("input")

    gsy = sub.add_parser("synth", help="write a synthetic test mesh")
    gsy.add_argument("kind", choices=["cylinder", "boot", "rigid"]); gsy.add_argument("output")
    a = ap.parse_args(argv)

    if a.cmd == "synth":
        m = {"cylinder": synth.skinned_cylinder, "boot": synth.boot, "rigid": synth.rigid_tube}[a.kind]()
        write_mesh(m, a.output); print(f"wrote {a.output}: {m.n_verts} verts, {m.n_faces} tris"); return 0
    model = ByteModel(a.skinned_bytes, a.rigid_bytes, a.per_strip_bytes, a.fixed_bytes, a.join, a.force_skinned)
    mesh = read_mesh(a.input, a.in_format)
    if a.cmd == "estimate":
        print(json.dumps(estimate_bytes(mesh, model), indent=2)); return 0
    opt = SimplifyOptions(lock_boundaries=not a.no_lock_seams, uv_weight=a.uv_weight,
                          normal_weight=a.normal_weight, bone_weight=a.bone_weight)
    before = estimate_bytes(mesh, model)
    if a.budget is not None:
        out, rep = fit_to_budget(mesh, a.budget, model, opt)
        after = {k: v for k, v in rep.items() if k != "original"}
    else:
        tgt = a.faces if a.faces is not None else int(mesh.n_faces * a.ratio)
        out = simplify(mesh, tgt, opt); after = estimate_bytes(out, model)
    write_mesh(out, a.output, a.out_format)
    print(json.dumps({"before": before, "after": after}, indent=2))
    if a.budget is not None and not after.get("fits"):
        print("WARNING: budget not met (locked seams/borders limit reduction)", file=sys.stderr); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
