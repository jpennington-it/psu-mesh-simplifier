import numpy as np, pytest
from psumesh import simplify, stripify, strips_to_faces, estimate_bytes, ByteModel, fit_to_budget
from psumesh.strips import _canon, join_strips
from psumesh.topology import seam_edges_geometric, face_normals, edge_counts
from psumesh.synth import skinned_cylinder, boot, rigid_tube
from psumesh.io import read_mesh, write_mesh, get_adapter

MESHES = {"cylinder": skinned_cylinder, "boot": boot}


def seam_points(m, tol=1e-6):
    pts = set()
    for e in seam_edges_geometric(m, tol): pts |= set(e)
    return pts


@pytest.fixture(params=list(MESHES))
def pair(request):
    m = MESHES[request.param]()
    return m, simplify(m, m.n_faces // 4)


def test_reduces(pair):
    m, s = pair
    assert s.n_faces < m.n_faces * 0.5


def test_seams_and_borders_intact(pair):
    m, s = pair
    # every original seam/border vertex position still exists on a seam/border of the result
    assert seam_points(m) <= seam_points(s)
    # UV seam: both u=0 and u=1 copies survive
    assert np.any(np.isclose(s.uvs[:, 0], 0)) and np.any(np.isclose(s.uvs[:, 0], 1))


def test_no_flipped_faces(pair):
    m, s = pair
    fn = face_normals(s.positions, s.faces)
    # result vertex normals are interpolated from the (outward) source normals; face normals must agree
    vn = s.normals[s.faces].sum(1)
    assert np.all((fn * vn).sum(1) > 0)
    assert np.all(np.linalg.norm(fn, axis=1) > 1e-10)


def test_manifold(pair):
    _, s = pair
    assert max(edge_counts(s.faces).values()) <= 2


def test_weights_valid(pair):
    _, s = pair
    assert s.bone_weights.shape[1] == 2
    assert np.allclose(s.bone_weights.sum(1), 1)
    assert np.all(s.bone_weights >= 0)


def test_stripify_exact_cover(pair):
    for mesh in pair:
        st = stripify(mesh.faces)
        want = sorted(_canon(list(map(int, f))) for f in mesh.faces)
        assert sorted(_canon(list(t)) for t in strips_to_faces(st)) == want
        assert sorted(_canon(list(t)) for t in strips_to_faces([join_strips(st)])) == want


@pytest.mark.parametrize("join", ["break", "degenerate"])
def test_budget_met(join):
    m = boot(); model = ByteModel(join_mode=join, per_strip_bytes=16)
    orig = estimate_bytes(m, model)["bytes"]
    budget = orig // 3
    out, rep = fit_to_budget(m, budget, model)
    assert rep["fits"] and estimate_bytes(out, model)["bytes"] <= budget
    assert np.allclose(out.bone_weights.sum(1), 1)


def test_rigid_uses_22_bytes():
    r = rigid_tube()
    assert estimate_bytes(r)["vertex_bytes"] == 22
    assert estimate_bytes(skinned_cylinder())["vertex_bytes"] == 48


@pytest.mark.parametrize("ext", [".json", ".obj"])
def test_io_roundtrip(tmp_path, ext):
    m = skinned_cylinder(12, 6); p = tmp_path / ("m" + ext)
    write_mesh(m, str(p)); r = read_mesh(str(p))
    assert r.n_faces == m.n_faces
    assert np.allclose(r.positions[r.faces], m.positions[m.faces], atol=1e-5)
    assert np.allclose(r.bone_weights.sum(1), 1)


def test_psu_adapter_stub():
    with pytest.raises(NotImplementedError):
        get_adapter("x.psu-ps2").read("x.psu-ps2")
