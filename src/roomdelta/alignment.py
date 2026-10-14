"""Cross-visit alignment from the unchanged environment.

1. ALIKED+LightGlue matches between every before/after keyframe pair, verified
   epipolarly, lifted to 3D with each visit's own depth.
2. Robust Sim(3) (x_A = s R x_B + t): group-aware RANSAC then IRLS with a
   Cauchy kernel. Each frame pair and each spatial voxel has a capped total
   weight so a textured moved object (or one heavily-filmed corner) cannot
   dominate the fit.
3. Optional robust point-to-plane ICP, accepted only if it improves residuals
   on held-out frame pairs.
4. Bootstrap over frame pairs -> per-point alignment uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import VisitMap
from .matching import Matcher, verify_epipolar


def umeyama(X: np.ndarray, Y: np.ndarray, w: np.ndarray | None = None, with_scale: bool = True):
    """Weighted similarity s,R,t minimising sum w ||Y - (s R X + t)||^2."""
    if w is None:
        w = np.ones(len(X))
    w = w / w.sum()
    mx, my = w @ X, w @ Y
    Xc, Yc = X - mx, Y - my
    C = (Yc * w[:, None]).T @ Xc
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_x = (w * (Xc**2).sum(1)).sum()
    s = (D * np.diag(S)).sum() / var_x if with_scale else 1.0
    t = my - s * R @ mx
    return s, R, t


def apply(s, R, t, X):
    return s * X @ R.T + t


@dataclass
class Correspondences:
    xa: np.ndarray       # (K,3) before-world
    xb: np.ndarray       # (K,3) after-world (own frame)
    ia: np.ndarray       # before frame index
    ib: np.ndarray       # after frame index
    ua: np.ndarray       # before pixel
    ub: np.ndarray       # after pixel
    pair: np.ndarray     # pair id
    score: np.ndarray


@dataclass
class Alignment:
    s: float
    R: np.ndarray
    t: np.ndarray
    sigma_median: float            # bootstrap per-point displacement, scene units of A
    sigma_p95: float
    scene_scale: float             # median before depth (unit reference)
    ok: bool
    stats: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(s=self.s, R=self.R.tolist(), t=self.t.tolist(), sigma_median=self.sigma_median,
                    sigma_p95=self.sigma_p95, scene_scale=self.scene_scale, ok=self.ok, stats=self.stats)


def _lift(m: VisitMap, i: int, uv: np.ndarray):
    H, W = m.hw
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    ok = m.valid[i][v, u]
    z = m.depth[i][v, u].astype(np.float64)
    Kinv = np.linalg.inv(m.K[i])
    rays = (Kinv @ np.stack([u, v, np.ones_like(u)]).astype(np.float64)).T
    c2w = np.linalg.inv(m.w2c[i])
    P = (rays * z[:, None]) @ c2w[:3, :3].T + c2w[:3, 3]
    return P, ok


def cross_visit_correspondences(A: VisitMap, B: VisitMap, matcher: Matcher, feats_a=None, feats_b=None,
                                min_inliers: int = 25) -> Correspondences:
    fa = feats_a or matcher.extract(A.images)
    fb = feats_b or matcher.extract(B.images)
    out = {k: [] for k in Correspondences.__dataclass_fields__}
    pid = 0
    for i in range(A.n):
        for j in range(B.n):
            m, sc = matcher.match(fa.raw[i], fb.raw[j])
            if len(m) < min_inliers:
                continue
            pa, pb = fa.kpts[i][m[:, 0]], fb.kpts[j][m[:, 1]]
            inl = verify_epipolar(pa, pb)
            if inl.sum() < min_inliers:
                continue
            pa, pb, sc = pa[inl], pb[inl], sc[inl]
            XA, oka = _lift(A, i, pa)
            XB, okb = _lift(B, j, pb)
            ok = oka & okb
            if ok.sum() < min_inliers // 2:
                continue
            n = int(ok.sum())
            out["xa"].append(XA[ok]); out["xb"].append(XB[ok])
            out["ia"].append(np.full(n, i)); out["ib"].append(np.full(n, j))
            out["ua"].append(pa[ok]); out["ub"].append(pb[ok])
            out["pair"].append(np.full(n, pid)); out["score"].append(sc[ok])
            pid += 1
    if pid == 0:
        raise RuntimeError("no verified cross-visit frame pairs: recordings do not overlap")
    return Correspondences(**{k: np.concatenate(v) for k, v in out.items()})


def group_weights(c: Correspondences, scale: float, pair_cap: float = 40.0, voxel_cap: float = 30.0,
                  voxel_frac: float = 0.1) -> np.ndarray:
    """Down-weight dense clusters: each frame pair and each spatial voxel saturates."""
    w = np.clip(c.score, 0.2, 1.0).astype(np.float64)
    _, inv, cnt = np.unique(c.pair, return_inverse=True, return_counts=True)
    w *= np.minimum(1.0, pair_cap / cnt[inv])
    vox = np.floor(c.xa / (voxel_frac * scale)).astype(np.int64)
    _, inv, cnt = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    w *= np.minimum(1.0, voxel_cap / cnt[inv.ravel()])
    return w


def ransac_sim3(c: Correspondences, w: np.ndarray, thr: float, iters: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    K = len(c.xa)
    p = w / w.sum()
    best, best_score = None, -1
    for _ in range(iters):
        idx = rng.choice(K, 4, replace=False, p=p)
        if len(np.unique(c.pair[idx])) < 2:
            continue
        X, Y = c.xb[idx], c.xa[idx]
        if np.linalg.matrix_rank(X - X.mean(0), tol=1e-3 * thr) < 2:
            continue
        s, R, t = umeyama(X, Y)
        if not (1e-3 < s < 1e3):
            continue
        r = np.linalg.norm(c.xa - apply(s, R, t, c.xb), axis=1)
        score = w[r < thr].sum()
        if score > best_score:
            best, best_score = (s, R, t), score
    return best


def irls_sim3(c: Correspondences, w: np.ndarray, init, cauchy: float, iters: int = 20, idx=None):
    s, R, t = init
    sel = np.arange(len(c.xa)) if idx is None else idx
    for _ in range(iters):
        r = np.linalg.norm(c.xa[sel] - apply(s, R, t, c.xb[sel]), axis=1)
        rw = w[sel] / (1.0 + (r / cauchy) ** 2)
        s, R, t = umeyama(c.xb[sel], c.xa[sel], rw)
    return s, R, t


def _held_out_residual(c, sel, s, R, t, cap):
    r = np.linalg.norm(c.xa[sel] - apply(s, R, t, c.xb[sel]), axis=1)
    r = r[r < cap]
    return float(np.median(r)) if len(r) else np.inf, int(len(r))


def align_visits(A: VisitMap, B: VisitMap, corr: Correspondences, n_boot: int = 60, icp: bool = True,
                 seed: int = 0) -> Alignment:
    import open3d as o3d

    rng = np.random.default_rng(seed)
    scale = float(np.median(A.depth[A.valid]))
    thr = 0.04 * scale
    w = group_weights(corr, scale)

    # Hold out ~20% of frame pairs for validating the refinement step.
    pairs = np.unique(corr.pair)
    held = rng.choice(pairs, max(1, len(pairs) // 5), replace=False)
    is_held = np.isin(corr.pair, held)
    fit_idx, held_idx = np.where(~is_held)[0], np.where(is_held)[0]

    sub = Correspondences(**{k: getattr(corr, k)[fit_idx] for k in Correspondences.__dataclass_fields__})
    init = ransac_sim3(sub, w[fit_idx], thr)
    if init is None:
        raise RuntimeError("Sim(3) RANSAC failed")
    s, R, t = irls_sim3(corr, w, init, cauchy=thr / 2, idx=fit_idx)
    held_before, n_held = _held_out_residual(corr, held_idx, s, R, t, 3 * thr)

    # ---- dense robust point-to-plane ICP on likely-static surfaces ----
    icp_info = dict(attempted=icp, accepted=False)
    if icp:
        def cloud(m: VisitMap, stride=4, xform=None):
            P = np.concatenate([m.backproject(i, stride=stride)[0] for i in range(m.n)])
            if xform is not None:
                P = apply(*xform, P)
            pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
            pc = pc.voxel_down_sample(0.01 * scale)
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05 * scale, max_nn=30))
            return pc

        pa, pb = cloud(A), cloud(B, xform=(s, R, t))
        loss = o3d.pipelines.registration.TukeyLoss(k=0.02 * scale)
        est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
        res = o3d.pipelines.registration.registration_icp(pb, pa, 0.05 * scale, np.eye(4), est,
                                                          o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
        T = res.transformation
        R2, t2 = T[:3, :3], T[:3, 3]
        s_new, R_new, t_new = s, R2 @ R, R2 @ t + t2
        held_after, _ = _held_out_residual(corr, held_idx, s_new, R_new, t_new, 3 * thr)
        icp_info.update(fitness=float(res.fitness), inlier_rmse=float(res.inlier_rmse),
                        rot_deg=float(np.degrees(np.arccos(np.clip((np.trace(R2) - 1) / 2, -1, 1)))),
                        trans=float(np.linalg.norm(t2)), held_out_median_before=held_before, held_out_median_after=held_after)
        if held_after <= held_before * 1.02:
            s, R, t = s_new, R_new, t_new
            icp_info["accepted"] = True

    # ---- bootstrap over frame pairs ----
    Pa = np.concatenate([A.backproject(i, stride=16)[0] for i in range(A.n)])
    Pb_own = ((Pa - t) @ R) / s  # A surface expressed in B's own frame under the current estimate
    disp = []
    order = np.argsort(corr.pair, kind="stable")
    starts = np.searchsorted(corr.pair[order], pairs)
    ends = np.searchsorted(corr.pair[order], pairs, side="right")
    for b in range(n_boot):
        bp = rng.integers(0, len(pairs), len(pairs))
        idx = np.concatenate([order[starts[k]:ends[k]] for k in bp])
        sb, Rb, tb = irls_sim3(corr, w, (s, R, t), cauchy=thr / 2, iters=8, idx=idx)
        disp.append(np.linalg.norm(apply(sb, Rb, tb, Pb_own) - Pa, axis=1))
    disp = np.sqrt(np.mean(np.square(disp), axis=0))

    # ---- diagnostics: per-region residuals on correspondences ----
    r_all = np.linalg.norm(corr.xa - apply(s, R, t, corr.xb), axis=1)
    inl = r_all < thr
    from sklearn.cluster import KMeans

    regions = KMeans(n_clusters=min(5, max(1, inl.sum() // 50)), n_init=4, random_state=0).fit_predict(corr.xa[inl])
    region_med = [float(np.median(r_all[inl][regions == k])) for k in np.unique(regions)]
    region_n = [int((regions == k).sum()) for k in np.unique(regions)]
    n_pairs_inl = len(np.unique(corr.pair[inl]))

    sigma_med, sigma_p95 = float(np.median(disp)), float(np.percentile(disp, 95))
    reasons = []
    if inl.sum() < 150:
        reasons.append(f"only {int(inl.sum())} inlier correspondences")
    if n_pairs_inl < 6:
        reasons.append(f"only {n_pairs_inl} frame pairs support the fit")
    if sigma_p95 > 0.05 * scale:
        reasons.append(f"bootstrap spread p95={sigma_p95:.3f} > 5% of scene scale")
    if max(region_med) > 3 * max(1e-9, np.median(r_all[inl])) and max(region_med) > 0.02 * scale:
        reasons.append("residuals inconsistent across room regions")
    stats = dict(n_corr=int(len(corr.xa)), n_pairs=int(len(pairs)), n_inliers=int(inl.sum()), n_inlier_pairs=n_pairs_inl,
                 inlier_thr=thr, median_inlier_residual=float(np.median(r_all[inl])),
                 region_median_residual=region_med, region_support=region_n, held_out_pairs=int(len(held)),
                 held_out_corr=n_held, icp=icp_info, failure_reasons=reasons)
    return Alignment(float(s), R, t, sigma_med, sigma_p95, scale, ok=not reasons, stats=stats)
