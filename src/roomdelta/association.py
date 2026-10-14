"""Rigid-motion hypotheses and identity helpers for object pairs across visits.

* ``fit_pair``: rigid motion between a before and an after object. Hypotheses come
  from ALIKED+LightGlue matches restricted to the two objects' masks (lifted to 3D,
  RANSAC) and from ICP started at yaw hypotheses about the floor normal (furniture
  moves on the floor). Shape compatibility is scored AFTER the motion, so motion is
  not penalised.
* ``resolve_other_side``: pairs unmatched partial views that are two sides of one
  unchanged object (occlusion + floor-footprint argument).

The change decision itself lives in ``decision.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .geometry import VisitMap
from .matching import Features, Matcher
from .objects import Object3D, label_compat
from .scene import Floor

def kabsch(X, Y, w=None):
    s, R, t = _umeyama(X, Y, w)
    return R, t


def _umeyama(X, Y, w=None):
    from .alignment import umeyama

    return umeyama(X, Y, w, with_scale=False)


def overlap(Pa: np.ndarray, Pb: np.ndarray, tau: float) -> tuple[float, float]:
    """(fraction of A within tau of B, fraction of B within tau of A)."""
    if len(Pa) == 0 or len(Pb) == 0:
        return 0.0, 0.0
    da, _ = cKDTree(Pb).query(Pa, distance_upper_bound=tau)
    db, _ = cKDTree(Pa).query(Pb, distance_upper_bound=tau)
    return float(np.isfinite(da).mean()), float(np.isfinite(db).mean())


def icp_rigid(Pa, Pb, R0, t0, tau, iters=30, yaw_only_axis=None):
    """Point-to-point ICP from (R0,t0); returns R,t mapping Pa -> Pb.

    With ``yaw_only_axis`` the rotation is constrained to that axis: objects
    resting on the floor translate in-plane and turn about the vertical.
    """
    R, t = R0.copy(), t0.copy()
    tree = cKDTree(Pb)
    for _ in range(iters):
        Q = Pa @ R.T + t
        d, j = tree.query(Q, distance_upper_bound=tau)
        ok = np.isfinite(d)
        if ok.sum() < 10:
            break
        X, Y = Pa[ok], Pb[j[ok]]
        if yaw_only_axis is None:
            R, t = kabsch(X, Y)
        else:
            R, t = _yaw_fit(X, Y, yaw_only_axis)
    return R, t


def _yaw_fit(X, Y, up):
    """Best rotation about ``up`` (plus free translation) mapping X -> Y."""
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    B = np.stack([e1, e2, up])          # world -> local
    Xl, Yl = (X - X.mean(0)) @ B.T, (Y - Y.mean(0)) @ B.T
    H = Xl[:, :2].T @ Yl[:, :2]
    th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
    c, s = np.cos(th), np.sin(th)
    Rl = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = B.T @ Rl @ B
    t = Y.mean(0) - R @ X.mean(0)
    return R, t


def rot_about(up, ang):
    K = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


@dataclass
class PairFit:
    a: str
    b: str
    R: np.ndarray
    t: np.ndarray
    method: str
    feat_inliers: int
    fit_ab: float          # overlap after transform
    fit_ba: float
    identity_fit: tuple
    yaw_alternatives: list = field(default_factory=list)   # (yaw_deg, score) of competing ICP minima

    @property
    def shape_score(self):
        return 0.5 * (self.fit_ab + self.fit_ba)


def feature_hypothesis(oa: Object3D, ob: Object3D, A: VisitMap, B: VisitMap, fa: Features, fb: Features,
                       matcher: Matcher, tau: float, top: int = 3, seed: int = 0):
    import cv2

    def best_frames(o):
        area = {}
        for ob_ in o.obs:
            area[ob_.frame] = max(area.get(ob_.frame, 0), ob_.mask.sum())
        fr = sorted(area, key=area.get, reverse=True)[:top]
        masks = {}
        for ob_ in o.obs:
            if ob_.frame in fr:
                masks[ob_.frame] = masks.get(ob_.frame, np.zeros_like(ob_.mask)) | ob_.mask
        return {f: cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for f, m in masks.items()}

    from .alignment import _lift

    ma, mb = best_frames(oa), best_frames(ob)
    XA, XB = [], []
    for i, mka in ma.items():
        for j, mkb in mb.items():
            m, _ = matcher.match(fa.raw[i], fb.raw[j])
            if len(m) == 0:
                continue
            pa, pb = fa.kpts[i][m[:, 0]], fb.kpts[j][m[:, 1]]
            ina = mka[pa[:, 1].astype(int).clip(0, mka.shape[0] - 1), pa[:, 0].astype(int).clip(0, mka.shape[1] - 1)]
            inb = mkb[pb[:, 1].astype(int).clip(0, mkb.shape[0] - 1), pb[:, 0].astype(int).clip(0, mkb.shape[1] - 1)]
            k = ina & inb
            if k.sum() < 3:
                continue
            Xa, oka = _lift(A, i, pa[k])
            Xb, okb = _lift(B, j, pb[k])
            XA.append(Xa[oka & okb]); XB.append(Xb[oka & okb])
    if not XA:
        return None, 0
    XA, XB = np.concatenate(XA), np.concatenate(XB)
    if len(XA) < 4:
        return None, 0
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(500):
        idx = rng.choice(len(XA), 3, replace=False)
        R, t = kabsch(XA[idx], XB[idx])
        n = int((np.linalg.norm(XA @ R.T + t - XB, axis=1) < tau).sum())
        if n > best_n:
            best, best_n = (R, t), n
    if best_n < 4:
        return None, best_n
    inl = np.linalg.norm(XA @ best[0].T + best[1] - XB, axis=1) < tau
    R, t = kabsch(XA[inl], XB[inl])
    return (R, t), int(inl.sum())


def fit_pair(oa: Object3D, ob: Object3D, floor: Floor, tau: float, feat=None, n_yaw: int = 12) -> PairFit:
    Pa, Pb = oa.points, ob.points
    up = floor.normal
    ident = overlap(Pa, Pb, tau)
    hyps = [("identity", np.eye(3), np.zeros(3))]
    if feat is not None:
        hyps.append(("features", feat[0], feat[1]))
    ca, cb = Pa.mean(0), Pb.mean(0)
    for k in range(n_yaw):
        R0 = rot_about(up, 2 * np.pi * k / n_yaw)
        hyps.append((f"icp_yaw{int(360 * k / n_yaw)}", R0, cb - R0 @ ca))
    results = []
    for name, R0, t0 in hyps:
        if name == "identity":
            R, t = R0, t0
            R, t = icp_rigid(Pa, Pb, R, t, tau, iters=10, yaw_only_axis=up)
        else:
            # coarse-to-fine: large capture radius first, then tight
            R, t = icp_rigid(Pa, Pb, R0, t0, 4 * tau, iters=15, yaw_only_axis=up)
            R, t = icp_rigid(Pa, Pb, R, t, tau, iters=15, yaw_only_axis=up)
        fab, fba = overlap(Pa @ R.T + t, Pb, tau)
        results.append((0.5 * (fab + fba), name, R, t, fab, fba))
    results.sort(key=lambda r: -r[0])
    sc, name, R, t, fab, fba = results[0]
    yaw_alts = []
    for r in results[1:]:
        dyaw = np.degrees(np.arccos(np.clip((np.trace(r[2] @ R.T) - 1) / 2, -1, 1)))
        if dyaw > 30 and r[0] > 0.85 * sc:
            yaw_alts.append((round(float(dyaw), 1), round(float(r[0]), 3)))
    return PairFit(oa.object_id, ob.object_id, R, t, name, 0 if feat is None else int(feat[2]), fab, fba, ident, yaw_alts)


@dataclass
class Association:
    matches: list[dict]
    unmatched_a: list[str]
    unmatched_b: list[str]
    fits: dict
    singles_a: dict = field(default_factory=dict)   # decision state + posterior margin of unpaired before objects
    singles_b: dict = field(default_factory=dict)


def _occluders(o: Object3D, other: VisitMap, objs_other: list[Object3D], eps_rel: float, eps_abs: float, radius: float):
    """For the samples of ``o`` that the other visit sees *behind* something nearer, which of the
    other visit's objects is that nearer surface? Returns ({object_id: n}, n_occluded_samples)."""
    from .visibility import depth_residuals, project

    trees = {x.object_id: cKDTree(x.points) for x in objs_other}
    counts, total = {}, 0
    for j in range(other.n):
        d, z = depth_residuals(other, j, o.points)
        occ = np.isfinite(d) & (d < -(eps_rel * z + eps_abs))
        if not occ.any():
            continue
        u, v, _ = project(other, j, o.points[occ])
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        zz = other.depth[j][vi, ui].astype(np.float64)
        c2w = np.linalg.inv(other.w2c[j])
        rays = (np.linalg.inv(other.K[j]) @ np.stack([ui, vi, np.ones_like(ui)]).astype(np.float64)).T
        P = (rays * zz[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
        total += len(P)
        for k, t in trees.items():
            n = int(np.isfinite(t.query(P, distance_upper_bound=radius)[0]).sum())
            if n:
                counts[k] = counts.get(k, 0) + n
    return counts, total


def _footprint_overlap(a: Object3D, b: Object3D, floor: Floor) -> float:
    la, ha = np.percentile(floor.to_2d(a.points), [5, 95], axis=0)
    lb, hb = np.percentile(floor.to_2d(b.points), [5, 95], axis=0)
    inter = np.prod(np.clip(np.minimum(ha, hb) - np.maximum(la, lb), 0, None))
    return float(inter / max(1e-12, min(np.prod(ha - la), np.prod(hb - lb))))


def resolve_other_side(asc: Association, objs_a: list[Object3D], objs_b: list[Object3D], ev_a: dict, ev_b: dict,
                       A: VisitMap, B: VisitMap, floor: Floor, eps_rel: float, eps_abs: float, radius: float,
                       min_occluded: float = 0.5, min_share: float = 0.6, max_conflict: float = 0.15) -> Association:
    """Pair unmatched objects that are two sides of one unchanged object.

    Partial views from opposite sides (front of an armchair before, its back after) share
    almost no surface, so they neither lock as unchanged nor fit rigidly. The visibility
    test still identifies them: the unmatched object's surface is *occluded* in the other
    visit, and the occluding surface belongs to one object of compatible class whose floor
    footprint overlaps it. Two solid objects cannot share floor area, so this is the same
    object seen from the other side, not a newly observed or vanished one.
    """
    oa = {o.object_id: o for o in objs_a}
    ob = {o.object_id: o for o in objs_b}
    um_a, um_b = set(asc.unmatched_a), set(asc.unmatched_b)
    for src, dst, ev_src, pool_src, pool_dst, other in (("B", "A", ev_b, um_b, um_a, A), ("A", "B", ev_a, um_a, um_b, B)):
        objs_src = ob if src == "B" else oa
        objs_dst = oa if src == "B" else ob
        for k in sorted(pool_src):
            if k not in pool_src:
                continue
            e = ev_src.get(k, {})
            if e.get("occluded", 0) < min_occluded or e.get("free_space_conflict", 0) > max_conflict:
                continue
            counts, total = _occluders(objs_src[k], other, list(objs_dst.values()), eps_rel, eps_abs, radius)
            if not counts:
                continue
            best = max(counts, key=counts.get)
            share = counts[best] / max(1, total)
            if share < min_share or label_compat(objs_src[k].label, objs_dst[best].label) < 0.8:
                continue
            ovl = _footprint_overlap(objs_src[k], objs_dst[best], floor)
            if ovl <= 0:
                continue
            if best not in pool_dst:
                # the occluder is already an unchanged object: k is one more partial view of it
                side = "b" if src == "A" else "a"
                for m in asc.matches:
                    if m[side] == best and m.get("kind") in ("unchanged_lock", "other_side"):
                        m.setdefault("extra_" + ("a" if src == "A" else "b"), []).append(k)
                        m["other_side_share"] = round(share, 3)
                        pool_src.discard(k)
                        break
                continue
            a_id, b_id = (best, k) if src == "B" else (k, best)
            asc.matches.append(dict(a=a_id, b=b_id, kind="other_side", occluder_share=round(share, 3),
                                    footprint_overlap=round(ovl, 3)))
            pool_src.discard(k)
            pool_dst.discard(best)
    asc.unmatched_a = [k for k in asc.unmatched_a if k in um_a]
    asc.unmatched_b = [k for k in asc.unmatched_b if k in um_b]
    return asc
