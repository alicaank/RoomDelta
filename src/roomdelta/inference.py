"""Generative change inference: explain the other visit's observations.

Every hypothesis about an object predicts what the other visit's cameras should
have measured. The source visit's surfaces are rendered into each target camera
as a *layered* depth image (several depth layers per pixel, each tagged with the
object it belongs to), so that

    present(o)  -> first layer of the full render
    absent(o)   -> first layer that does not belong to o (or "unknown" if o was
                   the last thing the source saw along that ray)

and the observed target depth is scored under both with a heteroscedastic
sensor model calibrated on the visit itself. Consequences that previously
needed separate rules fall out of the model:

* seen-through old surface -> strong evidence for absent;
* compatible surface still there -> strong evidence for present ("still there");
* occluded by something nearer -> both hypotheses predict the occluder: no evidence;
* flat object on a table -> absent predicts the table a few mm behind: little
  evidence either way, without a thickness rule;
* each independent view group contributes a *bounded* log-likelihood ratio,
  so the number of agreeing viewpoints, combined with an explicit prior on
  change, sets the decision instead of a fixed "two groups" rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import VisitMap

UNKNOWN = np.inf


# --------------------------------------------------------------------------- sensor model
@dataclass
class SensorModel:
    """delta = observed - predicted depth.

    sigma(z, q) = z * (a + b * (1 - q_rank)) + s_align, fitted by robust regression of
    within-visit multi-view disagreement on depth and confidence rank; an outlier
    component covers gross depth errors; R bounds depths for uniform densities.
    """
    a: float
    b: float
    s_align: float
    outlier: float
    R: float
    meta: dict = field(default_factory=dict)

    def sigma(self, z, q_rank):
        return z * (self.a + self.b * (1.0 - q_rank)) + self.s_align

    def loglik_surface(self, delta, sigma):
        """Observation of a predicted surface: Gaussian inlier + uniform outlier."""
        g = np.exp(-0.5 * (delta / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)
        return np.log((1 - self.outlier) * g + self.outlier / self.R)

    def loglik_unknown(self, obs, z_front, sigma):
        """Nothing was predicted behind z_front: any depth beyond it is equally likely;
        a depth well in front of it needs an outlier/occluder explanation."""
        behind = obs > z_front - 2 * sigma
        return np.where(behind, np.log(1.0 / self.R), np.log(self.outlier / self.R))


def _rank(q):
    """Per-frame confidence rank in [0,1] (DA3 confidence is a ranking signal)."""
    r = np.empty_like(q, dtype=np.float64)
    flat = q.ravel()
    order = np.argsort(flat)
    rr = np.empty(len(flat))
    rr[order] = np.linspace(0, 1, len(flat))
    r.ravel()[:] = rr
    return r


def calibrate_sensor(m: VisitMap, P: np.ndarray, frame: np.ndarray, uv: np.ndarray, groups: np.ndarray,
                     s_align: float, sample: int = 30000, seed: int = 0) -> SensorModel:
    """Fit sigma(z, q) from how other view groups of the SAME visit re-observe each surface sample."""
    from .visibility import depth_residuals

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), min(sample, len(P)), replace=False)
    ranks = np.stack([_rank(m.quality[i]) for i in range(m.n)])
    q = ranks[frame[idx], uv[idx, 1].astype(int), uv[idx, 0].astype(int)]
    rel, qs = [], []
    for j in range(m.n):
        d, z = depth_residuals(m, j, P[idx])
        other = groups[frame[idx]] != groups[j]
        ok = other & np.isfinite(d) & (np.abs(d) < 0.25 * z)
        rel.append(d[ok] / z[ok]); qs.append(q[ok])
    rel, qs = np.concatenate(rel), np.concatenate(qs)
    # robust sigma per confidence decile, then a line in (1 - q)
    bins = np.clip((qs * 10).astype(int), 0, 9)
    xs, ys = [], []
    for k in range(10):
        r = rel[bins == k]
        if len(r) > 200:
            xs.append(1 - (k + 0.5) / 10)
            ys.append(1.4826 * np.median(np.abs(r - np.median(r))))
    xs, ys = np.array(xs), np.array(ys)
    b, a = np.polyfit(xs, ys, 1) if len(xs) >= 3 else (0.0, float(np.median(ys)))
    a, b = max(a, 1e-3), max(b, 0.0)
    sig = a + b * (1 - qs)
    outlier = float(np.clip(np.mean(np.abs(rel) > 5 * sig), 0.01, 0.2))
    R = float(np.percentile(m.depth[m.valid], 99) * 1.5)
    return SensorModel(a, b, s_align, outlier, R, meta=dict(bins_x=xs.tolist(), bins_sigma=ys.tolist(), n=int(len(rel))))


# --------------------------------------------------------------------------- layered rendering
@dataclass
class Layers:
    depth: np.ndarray    # (K,H,W), inf where empty
    label: np.ndarray    # (K,H,W) object index, -1 background, -2 empty


def render_layers(P: np.ndarray, lab: np.ndarray, K: np.ndarray, w2c: np.ndarray, hw, voxel: float,
                  n_layers: int = 4, gap_rel: float = 0.06) -> Layers:
    """Splat surface samples into a layered depth image.

    Each sample covers a disc of its projected voxel size. Per pixel, layer 0 is the
    nearest sample; layer k is the nearest sample more than ``gap_rel * z`` behind
    layer k-1 (i.e. the next distinct surface). Each layer carries its sample's label.
    """
    H, W = hw
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    keep = z > 1e-3
    Pc, z, lab = Pc[keep], z[keep], lab[keep]
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    rad = np.clip(np.ceil(0.6 * K[0, 0] * voxel / z), 1, 4).astype(int)
    pix, zz, ll = [], [], []
    for r in range(1, 5):
        sel = rad == r
        if not sel.any():
            continue
        us, vs, zs, ls = u[sel], v[sel], z[sel], lab[sel]
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy > r * r:
                    continue
                ui = np.round(us).astype(int) + dx
                vi = np.round(vs).astype(int) + dy
                ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                pix.append(vi[ok] * W + ui[ok]); zz.append(zs[ok]); ll.append(ls[ok])
    pix, zz, ll = np.concatenate(pix), np.concatenate(zz), np.concatenate(ll)
    depth = np.full((n_layers, H * W), UNKNOWN)
    label = np.full((n_layers, H * W), -2, np.int64)
    prev = np.full(H * W, -np.inf)
    alive = np.ones(len(pix), bool)
    for k in range(n_layers):
        cand = alive & (zz > prev[pix] * (1 + gap_rel)) if k else alive
        if not cand.any():
            break
        best = np.full(H * W, UNKNOWN)
        np.minimum.at(best, pix[cand], zz[cand])
        hit = cand & (zz == best[pix])
        lb = np.full(H * W, -2, np.int64)
        lb[pix[hit]] = ll[hit]
        depth[k], label[k] = best, lb
        prev = np.where(np.isfinite(best), best, prev)
        alive = cand & ~hit
    return Layers(depth.reshape(n_layers, H, W), label.reshape(n_layers, H, W))


def behind_label0(layers: Layers):
    """For every pixel: depth/label of the first layer whose label differs from layer 0's
    (what the source would show if the layer-0 object were absent). UNKNOWN if nothing."""
    K = layers.depth.shape[0]
    l0 = layers.label[0]
    d = np.full(l0.shape, UNKNOWN)
    lab = np.full(l0.shape, -2)
    done = ~np.isfinite(layers.depth[0])
    for k in range(1, K):
        ok = ~done & np.isfinite(layers.depth[k]) & (layers.label[k] != l0)
        d[ok] = layers.depth[k][ok]
        lab[ok] = layers.label[k][ok]
        done |= ok | ~np.isfinite(layers.depth[k])
    return d, lab


# --------------------------------------------------------------------------- object evidence
def label_surface_points(P: np.ndarray, objects: list, radius: float) -> np.ndarray:
    from scipy.spatial import cKDTree

    lab = np.full(len(P), -1, np.int64)
    best = np.full(len(P), np.inf)
    for k, o in enumerate(objects):
        d, _ = cKDTree(o.points).query(P, distance_upper_bound=radius)
        better = d < best
        lab[better] = k
        best[better] = d[better]
    return lab


def splat_depth(P: np.ndarray, K: np.ndarray, w2c: np.ndarray, hw, voxel: float) -> np.ndarray:
    """Nearest-depth splat of a point set (UNKNOWN where empty)."""
    H, W = hw
    Pc = P @ w2c[:3, :3].T + w2c[:3, 3]
    z = Pc[:, 2]
    k = z > 1e-3
    Pc, z = Pc[k], z[k]
    out = np.full(H * W, UNKNOWN)
    if len(z) == 0:
        return out.reshape(H, W)
    u = K[0, 0] * Pc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Pc[:, 1] / z + K[1, 2]
    r = np.clip(np.ceil(0.6 * K[0, 0] * voxel / z), 1, 4).astype(int)
    for rr in range(1, 5):
        s = r == rr
        if not s.any():
            continue
        for dy in range(-rr, rr + 1):
            for dx in range(-rr, rr + 1):
                if dx * dx + dy * dy > rr * rr:
                    continue
                ui = np.round(u[s]).astype(int) + dx
                vi = np.round(v[s]).astype(int) + dy
                ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                np.minimum.at(out, vi[ok] * W + ui[ok], z[s][ok])
    return out.reshape(H, W)


class SceneRenderer:
    """Caches the layered render of one visit's labelled surfaces in every camera of the other
    visit, and scores object-level hypotheses against the other visit's depth maps."""

    def __init__(self, P_src, lab_src, n_obj, T: VisitMap, sensor: SensorModel, groups_T, voxel: float,
                 cap: float = np.log(20.0), stride: int = 2):
        self.T, self.sensor, self.groups, self.voxel, self.cap = T, sensor, groups_T, voxel, cap
        self.n_obj = n_obj
        self.P_src, self.lab_src = P_src, lab_src
        self.nuisance_k = 2.5
        H, W = T.hw
        self.grid = np.zeros((H, W), bool)
        self.grid[::stride, ::stride] = True
        self.frames = []
        for j in range(T.n):
            L = render_layers(P_src, lab_src, T.K[j], T.w2c[j], T.hw, voxel, gap_rel=2 * sensor.a)
            zb, _ = behind_label0(L)
            self.frames.append(dict(l0=L.label[0], z0=L.depth[0], zb=zb, q=_rank(T.quality[j])))

    def _ll(self, D, pred, front, q):
        s = self.sensor
        known = np.isfinite(pred)
        p = np.where(known, pred, 1.0)
        out = s.loglik_surface(D - p, s.sigma(p, q))
        return np.where(known, out, s.loglik_unknown(D, front, s.sigma(front, q)))

    def hypothesis_llr(self, obj: int, motion=None, exclude_groups=()):
        """log p(target obs | obj removed, and re-placed by rigid ``motion`` = (R, t) if given)
           - log p(target obs | everything stays). The re-placed samples are the very samples
        that were removed, so motion = identity reproduces "stays" exactly. Bounded per view group."""
        moved_points = None
        if motion is not None:
            R, t = motion
            moved_points = self.P_src[self.lab_src == obj] @ np.asarray(R).T + np.asarray(t)
        G = int(self.groups.max()) + 1
        acc, cnt = np.zeros(G), np.zeros(G)
        for j, f in enumerate(self.frames):
            l0, z0, zb, q = f["l0"], f["z0"], f["zb"], f["q"]
            zT = splat_depth(moved_points, self.T.K[j], self.T.w2c[j], self.T.hw, self.voxel) if moved_points is not None \
                else np.full(l0.shape, UNKNOWN)
            is_obj = l0 == obj
            use = self.grid & self.T.valid[j] & (is_obj | np.isfinite(zT))
            if not use.any():
                continue
            D = self.T.depth[j][use].astype(np.float64)
            qq = q[use]
            stay = np.where(np.isfinite(z0[use]), z0[use], UNKNOWN)
            # nuisance: the object's predicted surface may be offset by a regional misregistration of
            # the two reconstructions; profile out a relative depth offset bounded by the noise scale
            obj_px = is_obj[use] & np.isfinite(D)
            if obj_px.sum() >= 5:
                r = (D[obj_px] - stay[obj_px]) / stay[obj_px]
                lim = self.nuisance_k * (self.sensor.a + self.sensor.b * 0.5)
                off = float(np.clip(np.median(r[np.abs(r) < 0.15]) if np.any(np.abs(r) < 0.15) else 0.0, -lim, lim))
                stay = np.where(obj_px, stay * (1 + off), stay)
            # without the object: what was behind it (where it was the first surface), else unchanged
            without = np.where(is_obj[use], zb[use], z0[use])
            hyp = np.minimum(without, zT[use])
            front_stay = np.where(np.isfinite(stay), stay, np.where(np.isfinite(zT[use]), zT[use], 0.0))
            front_hyp = np.where(is_obj[use], z0[use], front_stay)
            ll_stay = self._ll(D, stay, front_stay, qq)
            ll_hyp = self._ll(D, hyp, front_hyp, qq)
            nearest = np.fmin(stay, hyp)
            sig = self.sensor.sigma(np.where(np.isfinite(nearest), nearest, D), qq)
            occl = np.isfinite(nearest) & (D < nearest - 3 * sig)      # unexpected occluder: no evidence
            # rays the source visit never observed carry no information about any hypothesis:
            # otherwise a hypothesis could gain likelihood merely by filling unobserved space
            llr = np.where(occl | ~np.isfinite(stay), 0.0, ll_hyp - ll_stay)
            if not np.any(llr != 0):
                continue
            g = self.groups[j]
            if g in exclude_groups:
                continue
            acc[g] += llr.sum()
            cnt[g] += 1
        per_group = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        return float(np.nansum(np.clip(per_group, -self.cap, self.cap))), int((cnt > 0).sum()), per_group


def calibrate_cross_sensor(T: VisitMap, P: np.ndarray, s_align: float,
                           sample: int = 40000, seed: int = 0) -> SensorModel:
    """Noise of comparing S's surfaces with T's depth maps, fitted on the comparison itself.

    Two independent learned reconstructions disagree non-rigidly by more than either
    disagrees with itself, so within-visit consistency underestimates cross-visit noise.
    Most of a room is static: the central mode of the cross-visit residuals (|delta|/z
    within a broad window) is the static-surface noise; true changes and occlusions fall in
    the tails and are covered by the outlier term. Fitted per confidence decile of T."""
    from .visibility import project

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), min(sample, len(P)), replace=False)
    rel, qs = [], []
    ranks = [_rank(T.quality[j]) for j in range(T.n)]
    for j in range(T.n):
        u, v, z = project(T, j, P[idx])
        H, W = T.hw
        ui, vi = np.round(u).astype(int), np.round(v).astype(int)
        ok = (z > 0) & (ui >= 2) & (ui < W - 2) & (vi >= 2) & (vi < H - 2)
        ui, vi, z = ui[ok], vi[ok], z[ok]
        good = T.valid[j][vi, ui]
        d = T.depth[j][vi, ui][good] - z[good]
        r = d / z[good]
        win = np.abs(r) < 0.15
        rel.append(r[win]); qs.append(ranks[j][vi, ui][good][win])
    rel, qs = np.concatenate(rel), np.concatenate(qs)
    xs, ys = [], []
    for k in range(10):
        r = rel[np.clip((qs * 10).astype(int), 0, 9) == k]
        if len(r) > 300:
            # central mode: iterate a robust scale on the inner part of the distribution
            sgm = 1.4826 * np.median(np.abs(r - np.median(r)))
            for _ in range(5):
                inner = r[np.abs(r - np.median(r)) < 3 * sgm]
                sgm = 1.4826 * np.median(np.abs(inner - np.median(inner)))
            xs.append(1 - (k + 0.5) / 10); ys.append(sgm)
    xs, ys = np.array(xs), np.array(ys)
    b, a = np.polyfit(xs, ys, 1) if len(xs) >= 3 else (0.0, float(np.median(ys)))
    a, b = max(a, 1e-3), max(b, 0.0)
    sig = a + b * (1 - qs)
    outlier = float(np.clip(np.mean(np.abs(rel) > 4 * sig), 0.02, 0.3))
    R = float(np.percentile(T.depth[T.valid], 99) * 1.5)
    return SensorModel(a, b, s_align, outlier, R, meta=dict(kind="cross-visit", bins_x=xs.tolist(), bins_sigma=ys.tolist(),
                                                            n=int(len(rel))))
