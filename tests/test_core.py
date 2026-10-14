"""Unit tests for coordinate conventions and decision rules (code correctness only;
system behaviour is evaluated on real recordings in evaluation/)."""
import numpy as np

from roomdelta import navigation as N
from roomdelta import visibility as V
from roomdelta.alignment import apply, umeyama
from roomdelta.geometry import VisitMap


def _rot(ax, ang):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def _wall_map(name, z_wall=4.0, H=40, W=60, f=50.0, centers=((0, 0, 0),), blocker=None):
    """Cameras looking down +z at a fronto-parallel wall; optional square blocker at depth blocker[0]."""
    n = len(centers)
    K = np.tile(np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]]), (n, 1, 1))
    w2c = np.tile(np.eye(4), (n, 1, 1))
    for i, c in enumerate(centers):
        w2c[i, :3, 3] = -np.asarray(c, float)
    depth = np.full((n, H, W), z_wall, np.float32)
    if blocker is not None:
        zb, (u0, u1, v0, v1) = blocker
        depth[:, v0:v1, u0:u1] = zb
    return VisitMap(name, np.zeros((n, H, W, 3), np.uint8), np.arange(n, dtype=float), K, w2c, depth,
                    np.ones((n, H, W), np.float32), np.ones((n, H, W), bool), np.eye(3))


def test_umeyama_recovers_known_sim3():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 3))
    R, t, s = _rot([1, 2, 3], 0.7), np.array([0.3, -1.0, 2.0]), 1.7
    s2, R2, t2 = umeyama(X, apply(s, R, t, X))
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R) and np.allclose(t2, t)


def test_transformed_moves_cameras_and_depth_consistently():
    m = _wall_map("a", centers=[(0, 0, 0), (0.5, 0, 0.2)])
    s, R, t = 2.5, _rot([0, 1, 0], 0.4), np.array([1.0, 2.0, -3.0])
    m2 = m.transformed(s, R, t)
    for i in range(m.n):
        P, _ = m.backproject(i)
        P2, _ = m2.backproject(i)
        assert np.allclose(apply(s, R, t, P), P2, atol=1e-6)


def test_visibility_sign_convention():
    # surface point at z=2 in front of the cameras
    P = np.array([[0.0, 0.0, 2.0]])
    pts = V.SurfacePoints(P, np.zeros((1, 3), np.uint8), np.zeros(1, int), np.zeros((1, 2)), np.ones(1, int))
    centers = [(0, 0, 0), (1.0, 0, 0), (-1.0, 0, 0)]
    groups = np.arange(3)
    # target sees the wall far behind -> free-space conflict ("seen through")
    far = _wall_map("far", z_wall=4.0, centers=centers)
    r = V.visibility_test(far, far, pts, groups, eps_rel=0.05, eps_abs=0.0)
    assert r.status[0] == V.CONFLICT
    # target sees a nearer surface covering the point -> occluded, never "removed"
    near = _wall_map("near", z_wall=1.0, centers=centers)
    r = V.visibility_test(near, near, pts, groups, eps_rel=0.05, eps_abs=0.0)
    assert r.status[0] == V.OCCLUDED
    # target sees a surface at the same depth -> compatible
    same = _wall_map("same", z_wall=2.0, centers=[(0, 0, 0)])
    r = V.visibility_test(same, same, pts, np.zeros(1, int), eps_rel=0.05, eps_abs=0.0)
    assert r.status[0] == V.COMPAT


def test_single_view_group_conflict_is_weak():
    P = np.array([[0.0, 0.0, 2.0]])
    pts = V.SurfacePoints(P, np.zeros((1, 3), np.uint8), np.zeros(1, int), np.zeros((1, 2)), np.ones(1, int))
    far = _wall_map("far", z_wall=4.0, centers=[(0, 0, 0), (0.01, 0, 0), (0.02, 0, 0)])
    r = V.visibility_test(far, far, pts, np.zeros(3, int), eps_rel=0.05, eps_abs=0.0)   # all one view group
    assert r.status[0] == 4


def test_astar_respects_obstacles():
    passable = np.ones((20, 20), bool)
    passable[5:15, 10] = False
    p = N.astar(passable, (10, 2), (10, 18))
    assert p is not None and all(passable[tuple(c)] for c in p)
    passable[:, 10] = False
    assert N.astar(passable, (10, 2), (10, 18)) is None


def test_layered_render_and_identity_hypothesis():
    """Rendering a wall with an object in front: removing the object exposes the wall; moving it by
    the identity transform reproduces 'stays' exactly (zero log-likelihood ratio)."""
    from roomdelta.inference import SceneRenderer, SensorModel, behind_label0, render_layers

    m = _wall_map("t", z_wall=4.0, centers=[(0, 0, 0), (0.3, 0, 0)], blocker=(2.0, (20, 40, 10, 30)))
    P, lab = [], []
    for i in range(m.n):
        p, uv = m.backproject(i)
        P.append(p)
        lab.append(np.where(m.depth[i][uv[:, 1].astype(int), uv[:, 0].astype(int)] < 3, 0, -1))
    P, lab = np.concatenate(P), np.concatenate(lab)
    L = render_layers(P, lab, m.K[0], m.w2c[0], m.hw, voxel=0.05, gap_rel=0.1)
    zb, _ = behind_label0(L)
    assert np.isclose(np.nanmin(L.depth[0][15:25, 25:35]), 2.0, atol=0.05)     # object in front
    s = SensorModel(a=0.01, b=0.0, s_align=0.0, outlier=0.05, R=10.0)
    r = SceneRenderer(P, lab, 1, m, s, np.arange(m.n), voxel=0.05)
    l_id, _, _ = r.hypothesis_llr(0, (np.eye(3), np.zeros(3)))
    l_gone, _, _ = r.hypothesis_llr(0, None)
    assert abs(l_id) < 1e-6          # identity motion == stays
    assert l_gone < 0                # the target still sees the object: "gone" is contradicted
