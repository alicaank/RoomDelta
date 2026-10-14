"""Visibility-aware geometric change detection.

For a surface point p observed in visit S, we ask every camera j of visit T:
would p have been visible, and what does T's depth say along that ray?

    delta = D_T,j(u) - z_p
    |delta| <= eps(z)   compatible surface
    delta  >  eps(z)    T sees *through* p  -> free-space conflict
    delta  < -eps(z)    something nearer hides p -> occluded
    off-image / behind / invalid / silhouette -> not observed

Evidence is aggregated per *view group* (cameras clustered by pose) so thirty
near-identical frames count as one witness. Tolerances are self-calibrated
from within-visit multi-view depth consistency plus alignment uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import VisitMap

COMPAT, CONFLICT, OCCLUDED, UNOBS = 0, 1, 2, 3


@dataclass
class SurfacePoints:
    P: np.ndarray        # (N,3) shared-world coordinates
    rgb: np.ndarray      # (N,3) uint8
    frame: np.ndarray    # (N,) source frame index
    uv: np.ndarray       # (N,2) source pixel
    self_support: np.ndarray  # (N,) number of own-visit view groups confirming the surface


@dataclass
class VisibilityResult:
    source: str
    target: str
    pts: SurfacePoints
    counts: np.ndarray           # (N,4) view-group counts: compat, conflict, occluded, unobserved
    status: np.ndarray           # (N,) 0..4 (see STATUS_NAMES)
    witness: dict                # point idx -> list of (target frame, observed depth, projected depth)
    eps_rel: float
    eps_abs: float
    meta: dict = field(default_factory=dict)


def view_groups(m: VisitMap, pos_thr: float, ang_thr_deg: float = 20.0) -> np.ndarray:
    """Greedy clustering of cameras by centre and optical axis."""
    c2w = m.c2w()
    C, z = c2w[:, :3, 3], c2w[:, :3, 2]
    reps, g = [], np.zeros(m.n, int)
    cos_thr = np.cos(np.radians(ang_thr_deg))
    for i in range(m.n):
        for k, r in enumerate(reps):
            if np.linalg.norm(C[i] - C[r]) < pos_thr and z[i] @ z[r] > cos_thr:
                g[i] = k
                break
        else:
            reps.append(i)
            g[i] = len(reps) - 1
    return g


def surface_points(m: VisitMap, voxel: float, stride: int = 2) -> SurfacePoints:
    P, rgb, fr, uv = [], [], [], []
    for i in range(m.n):
        p, q = m.backproject(i, stride=stride)
        P.append(p); uv.append(q); fr.append(np.full(len(p), i))
        rgb.append(m.images[i][q[:, 1].astype(int), q[:, 0].astype(int)])
    P, rgb, fr, uv = map(np.concatenate, (P, rgb, fr, uv))
    # voxel subsample: one representative surface sample per voxel
    key = np.floor(P / voxel).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    sel = np.sort(first)
    return SurfacePoints(P[sel], rgb[sel], fr[sel], uv[sel], np.zeros(len(sel), int))


def project(m: VisitMap, j: int, P: np.ndarray):
    Pc = P @ m.w2c[j][:3, :3].T + m.w2c[j][:3, 3]
    z = Pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvh = Pc @ m.K[j].T
        u, v = uvh[:, 0] / z, uvh[:, 1] / z
    return u, v, z


def depth_residuals(m: VisitMap, j: int, P: np.ndarray, border: int = 2):
    """delta = observed - projected, NaN where not observed."""
    H, W = m.hw
    u, v, z = project(m, j, P)
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
    inb = (z > 1e-6) & (ui >= border) & (ui < W - border) & (vi >= border) & (vi < H - border)
    d = np.full(len(P), np.nan)
    idx = np.where(inb)[0]
    ok = m.valid[j][vi[idx], ui[idx]]
    idx = idx[ok]
    d[idx] = m.depth[j][vi[idx], ui[idx]] - z[idx]
    return d, z


def calibrate_eps(m: VisitMap, pts: SurfacePoints, groups: np.ndarray, sample: int = 20000, seed: int = 0):
    """Within-visit multi-view relative depth consistency (robust sigma of delta/z)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts.P), min(sample, len(pts.P)), replace=False)
    rel = []
    for j in range(m.n):
        d, z = depth_residuals(m, j, pts.P[idx])
        other = groups[pts.frame[idx]] != groups[j]
        r = d / np.maximum(z, 1e-6)
        r = r[other & np.isfinite(r) & (np.abs(r) < 0.25)]
        rel.append(r)
    rel = np.concatenate(rel)
    mad = 1.4826 * np.median(np.abs(rel - np.median(rel)))
    return float(mad), rel


def self_support(m: VisitMap, pts: SurfacePoints, groups: np.ndarray, eps_rel: float, eps_abs: float):
    """Number of other own-visit view groups that observe a compatible surface;
    also flags floaters that the own visit sees through from other groups."""
    G = groups.max() + 1
    compat = np.zeros((len(pts.P), G), bool)
    thru = np.zeros((len(pts.P), G), bool)
    for j in range(m.n):
        d, z = depth_residuals(m, j, pts.P)
        eps = eps_rel * z + eps_abs
        compat[:, groups[j]] |= np.abs(d) <= eps
        thru[:, groups[j]] |= d > eps
    own = groups[pts.frame]
    compat[np.arange(len(own)), own] = False
    thru[np.arange(len(own)), own] = False
    return compat.sum(1), (thru & ~compat).sum(1)


def visibility_test(S: VisitMap, T: VisitMap, pts: SurfacePoints, groups_T: np.ndarray, eps_rel: float, eps_abs: float,
                    min_conflict_groups: int = 2, keep_witness: int = 6) -> VisibilityResult:
    N = len(pts.P)
    G = groups_T.max() + 1
    per = np.zeros((N, G, 3), np.int32)  # compat, conflict, occluded frame counts per group
    wit_f, wit_d, wit_z = [], [], []
    for j in range(T.n):
        d, z = depth_residuals(T, j, pts.P)
        eps = eps_rel * z + eps_abs
        obs = np.isfinite(d)
        c = obs & (np.abs(d) <= eps)
        f = obs & (d > eps)
        o = obs & (d < -eps)
        g = groups_T[j]
        per[:, g, 0] += c; per[:, g, 1] += f; per[:, g, 2] += o
        wit_f.append(np.where(f, j, -1)); wit_d.append(np.where(f, d + z, np.nan)); wit_z.append(z)
    # group-level vote: majority label within the group (ties -> compatible, the conservative choice)
    tot = per.sum(2)
    lab = np.argmax(per[:, :, [0, 1, 2]] + np.array([0.5, 0, 0.25]), axis=2)
    lab[tot == 0] = 3
    counts = np.stack([(lab == k).sum(1) for k in range(4)], 1)
    status = np.full(N, UNOBS)
    ncomp, nconf, nocc = counts[:, 0], counts[:, 1], counts[:, 2]
    status[nocc > 0] = OCCLUDED
    status[nconf > 0] = 4  # weak
    status[(nconf >= min_conflict_groups) & (ncomp == 0)] = CONFLICT
    status[ncomp > 0] = np.where((nconf[ncomp > 0] >= 2 * ncomp[ncomp > 0] + min_conflict_groups), CONFLICT, COMPAT)
    # witnesses for conflict points: the target frames that saw through them
    wf = np.stack(wit_f, 1)
    wd = np.stack(wit_d, 1)
    wz = np.stack(wit_z, 1)
    witness = {}
    for i in np.where(status == CONFLICT)[0]:
        js = np.where(wf[i] >= 0)[0]
        # prefer one witness per view group
        seen, sel = set(), []
        for j in js:
            if groups_T[j] not in seen:
                seen.add(groups_T[j]); sel.append(j)
        witness[int(i)] = [(int(j), float(wd[i, j]), float(wz[i, j])) for j in sel[:keep_witness]]
    return VisibilityResult(S.name, T.name, pts, counts, status, witness, eps_rel, eps_abs,
                            meta=dict(n_view_groups_target=int(G)))


def change_clusters(vis: VisibilityResult, radius: float, min_points: int = 30) -> np.ndarray:
    """DBSCAN over free-space-conflict points; -1 for noise / non-conflict points."""
    from sklearn.cluster import DBSCAN

    lab = np.full(len(vis.pts.P), -1)
    idx = np.where(vis.status == CONFLICT)[0]
    if len(idx) < min_points:
        return lab
    c = DBSCAN(eps=radius, min_samples=8).fit_predict(vis.pts.P[idx])
    ids, cnt = np.unique(c[c >= 0], return_counts=True)
    good = set(ids[cnt >= min_points])
    remap = {old: k for k, old in enumerate(sorted(good))}
    lab[idx] = [remap.get(x, -1) for x in c]
    return lab
