"""MAP scene-change decision from generative evidence.

One objective replaces the rule cascade. Every before object a and after object b
gets exactly one explanation; the total log-posterior (relative to "nothing
changed") is maximised by a global assignment:

    pair (a, b) unchanged : l_id(I)
    pair (a, b) moved by T: eta * [LLR_B(a moved by T) + LLR_A(b came from T^-1)] + l_id(T)
                            + log pi_move + log p(motion)
    a gone                : eta * LLR_B(a absent) + log pi_change
    b added               : eta * LLR_A(b absent before) + log pi_change
    a / b unexplained     : log p_miss if the other visit looked at it (detector miss), else 0

LLR_X(.) are rendered-hypothesis log-likelihood ratios against visit X's depth maps
(inference.SceneRenderer). eta < 1 tempers them because depth errors of one learned
reconstruction are correlated across its frames (generalised-Bayes learning rate).
Decisions whose margin over the best alternative is below log(odds_min) are
reported as uncertain; near-tied identity assignments of same-class objects are
reported as ambiguous.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .association import Association, PairFit, feature_hypothesis, fit_pair
from .objects import label_compat
from .inference import SceneRenderer


@dataclass
class DecisionConfig:
    eta: float = 0.25                 # likelihood tempering (correlated per-frame depth errors)
    log_pi_change: float = np.log(0.05)
    log_pi_move: float = np.log(0.05)
    sigma_motion_m: float = 1.5       # weak prior on how far furniture moves in a room
    p_miss: float = 0.3               # detector misses an object that the other visit looked at
    odds_min: float = 19.0            # report a change only at >= 95% posterior odds
    shape_slope: float = 6.0          # identity log-likelihood per unit of post-motion shape overlap
    shape_ref: float = 0.6            # overlap at which identity is neutral
    feat_per_inlier: float = 0.08


@dataclass
class Decision:
    matches: list = field(default_factory=list)       # dict(a, b, kind: unchanged|moved, R, t, margin, ambiguous_with, terms)
    singles_a: dict = field(default_factory=dict)     # a -> dict(state: gone|present|unobserved, margin, terms)
    singles_b: dict = field(default_factory=dict)     # b -> dict(state: added|present|unobserved, margin, terms)
    fits: dict = field(default_factory=dict)


def identity_loglik(shape: float, n_inl: int, lc: float, c: DecisionConfig) -> float:
    l = float(np.clip(c.shape_slope * (shape - c.shape_ref), -4.0, 3.0))
    return l + c.feat_per_inlier * min(n_inl, 40) + float(np.log(max(lc, 1e-6)))


def decide(objs_a, objs_b, rA: SceneRenderer, rB: SceneRenderer, floor, tau: float, mpu: float, max_dist: float,
           min_motion: float, A=None, B=None, fa=None, fb=None, matcher=None, cfg: DecisionConfig = DecisionConfig(),
           emb_a=None, emb_b=None, app_llr=None) -> Decision:
    na, nb = len(objs_a), len(objs_b)
    dec = Decision()
    c = cfg
    # ---------------- single-object hypotheses
    gone, seen_a = np.zeros(na), np.zeros(na, int)
    for i in range(na):
        l, n, _ = rA.hypothesis_llr(i, None)
        gone[i], seen_a[i] = l, n
    added, seen_b = np.zeros(nb), np.zeros(nb, int)
    for j in range(nb):
        l, n, _ = rB.hypothesis_llr(j, None)
        added[j], seen_b[j] = l, n
    miss_a = np.where(seen_a > 0, np.log(c.p_miss), 0.0)
    miss_b = np.where(seen_b > 0, np.log(c.p_miss), 0.0)
    s_gone = c.eta * gone + c.log_pi_change
    s_added = c.eta * added + c.log_pi_change
    single_a = np.maximum(miss_a, s_gone)
    single_b = np.maximum(miss_b, s_added)

    # ---------------- pair hypotheses
    W = np.full((na, nb), -np.inf)
    info = {}
    for i, oa in enumerate(objs_a):
        for j, ob in enumerate(objs_b):
            lc = label_compat(oa.label, ob.label)
            if lc < 0.6 or np.linalg.norm(oa.centroid - ob.centroid) > max_dist:
                continue
            feat = None
            n_inl = 0
            if matcher is not None:
                fh, n_inl = feature_hypothesis(oa, ob, A, B, fa, fb, matcher, tau)
                feat = None if fh is None else (fh[0], fh[1], n_inl)
            pf: PairFit = fit_pair(oa, ob, floor, tau, feat)
            dec.fits[(oa.object_id, ob.object_id)] = pf
            shape_I = float(np.mean(pf.identity_fit))
            l_app = 0.0
            if app_llr is not None and oa.object_id in emb_a and ob.object_id in emb_b:
                l_app = app_llr(float(emb_a[oa.object_id] @ emb_b[ob.object_id]))
            s_unch = identity_loglik(shape_I, 0, lc, c) + l_app
            disp = float(np.linalg.norm(pf.R @ oa.centroid + pf.t - oa.centroid))
            rot = float(np.degrees(np.arccos(np.clip((np.trace(pf.R) - 1) / 2, -1, 1))))
            s_move, terms_m = -np.inf, {}
            if disp > min_motion or rot > 15:
                lA, gA_, _ = rA.hypothesis_llr(i, (pf.R, pf.t))
                lB, gB_, _ = rB.hypothesis_llr(j, (pf.R.T, -pf.R.T @ pf.t))
                l_id = identity_loglik(pf.shape_score, n_inl, lc, c) + l_app
                prior = c.log_pi_move - 0.5 * (disp * mpu / c.sigma_motion_m) ** 2
                s_move = c.eta * (lA + lB) + l_id + prior
                terms_m = dict(llr_before_obj_in_after=round(lA, 2), llr_after_obj_in_before=round(lB, 2), groups=(gA_, gB_),
                               identity=round(l_id, 2), appearance=round(l_app, 2), prior=round(prior, 2))
            kind = "moved" if s_move > s_unch else "unchanged"
            score = max(s_move, s_unch)
            W[i, j] = score - single_a[i] - single_b[j]
            info[(i, j)] = dict(kind=kind, score=score, s_unchanged=round(s_unch, 2), s_moved=round(float(s_move), 2),
                                disp=disp, rot=rot, terms=terms_m, pf=pf)

    # ---------------- global assignment (maximise total benefit; unpaired is always allowed)
    big = np.full((na + nb, nb + na), 0.0)
    big[:na, :nb] = np.where(np.isfinite(W), -W, 1e6)
    big[:na, nb:] = np.where(np.eye(na) > 0, 0.0, 1e6)
    big[na:, :nb] = np.where(np.eye(nb) > 0, 0.0, 1e6)
    r, cidx = linear_sum_assignment(big)
    paired_a, paired_b = set(), set()
    for i, j in zip(r, cidx):
        if i < na and j < nb and np.isfinite(W[i, j]) and W[i, j] > 0:
            inf = info[(i, j)]
            margin = float(W[i, j])
            m = dict(a=objs_a[i].object_id, b=objs_b[j].object_id, kind=inf["kind"], margin=round(margin, 2),
                     disp=inf["disp"], rot=inf["rot"], terms=inf["terms"], s_unchanged=inf["s_unchanged"],
                     s_moved=inf["s_moved"], ambiguous_with=[], i=i, j=j)
            if inf["kind"] == "moved":
                m["R"], m["t"] = inf["pf"].R, inf["pf"].t
            dec.matches.append(m)
            paired_a.add(i); paired_b.add(j)

    # identity ambiguity among same-class moves: is the swapped assignment nearly as good?
    moved = [m for m in dec.matches if m["kind"] == "moved"]
    for m1 in moved:
        for m2 in moved:
            if m1 is m2:
                continue
            i1, j1, i2, j2 = m1["i"], m1["j"], m2["i"], m2["j"]
            if not (np.isfinite(W[i1, j2]) and np.isfinite(W[i2, j1])):
                continue
            if objs_a[i1].label != objs_a[i2].label:      # only interchangeable (same-class) objects
                continue
            delta = (W[i1, j1] + W[i2, j2]) - (W[i1, j2] + W[i2, j1])
            if delta < np.log(c.odds_min):
                m1["ambiguous_with"].append(m2["a"])
                m1.setdefault("swap_margin", round(float(delta), 2))

    # co-motion (MDL): parts that touch and are displaced the same way are one move, so only one
    # of them pays the move prior; the others get it back. Transitive through touching parts.
    from scipy.spatial import cKDTree

    def disp_vec(m):
        o = objs_a[m["i"]]
        return m["R"] @ o.centroid + m["t"] - o.centroid

    payer = {}
    order = sorted(moved, key=lambda m: -m["margin"])
    for k, m in enumerate(order):
        for m0 in order[:k]:
            if m0["margin"] < np.log(c.odds_min) and m0["a"] not in payer.values():
                continue
            touch = np.isfinite(cKDTree(objs_a[m0["i"]].points).query(objs_a[m["i"]].points, distance_upper_bound=3 * tau)[0]).any()
            if touch and np.linalg.norm(disp_vec(m) - disp_vec(m0)) < 2 * tau:
                payer[m["a"]] = payer.get(m0["a"], m0["a"])
                m["margin"] = round(m["margin"] - c.log_pi_move, 2)
                m["attached_to"] = payer[m["a"]]
                break

    # when two same-class moves cannot be told apart, report the assignment with the smaller
    # total motion (Occam); the ambiguity flag stays on both
    for m1 in moved:
        for m2 in moved:
            if m1 is m2 or m2["a"] not in m1["ambiguous_with"]:
                continue
            i1, j1, i2, j2 = m1["i"], m1["j"], m2["i"], m2["j"]
            alt1, alt2 = info.get((i1, j2)), info.get((i2, j1))
            if alt1 is None or alt2 is None or alt1["kind"] != "moved" or alt2["kind"] != "moved":
                continue
            if alt1["disp"] + alt2["disp"] < m1["disp"] + m2["disp"]:
                for m, (ii, jj), alt in ((m1, (i1, j2), alt1), (m2, (i2, j1), alt2)):
                    m.update(b=objs_b[jj].object_id, j=jj, disp=alt["disp"], rot=alt["rot"], terms=alt["terms"],
                             margin=round(float(W[ii, jj]), 2), R=alt["pf"].R, t=alt["pf"].t, occam_reassigned=True)

    for i, oa in enumerate(objs_a):
        if i in paired_a:
            continue
        alt = sorted([miss_a[i], s_gone[i]])
        state = "gone" if s_gone[i] > miss_a[i] else ("present" if seen_a[i] > 0 else "unobserved")
        dec.singles_a[oa.object_id] = dict(state=state, margin=round(float(alt[1] - alt[0]), 2), llr_gone=round(float(gone[i]), 2),
                                           groups_observed=int(seen_a[i]))
    for j, ob in enumerate(objs_b):
        if j in paired_b:
            continue
        alt = sorted([miss_b[j], s_added[j]])
        state = "added" if s_added[j] > miss_b[j] else ("present" if seen_b[j] > 0 else "unobserved")
        dec.singles_b[ob.object_id] = dict(state=state, margin=round(float(alt[1] - alt[0]), 2), llr_added=round(float(added[j]), 2),
                                           groups_observed=int(seen_b[j]))
    # explaining away: a leftover "added" fragment lying on a moved object's predicted new place,
    # or a leftover "gone" fragment on its old place, is part of that move, not a separate change
    for m in dec.matches:
        if m["kind"] != "moved":
            continue
        Pa = objs_a[m["i"]].points
        new_place = cKDTree(Pa @ m["R"].T + m["t"])
        old_place = cKDTree(Pa)
        for k, st in list(dec.singles_b.items()):
            if st["state"] == "added":
                ob = next(o for o in objs_b if o.object_id == k)
                if np.isfinite(new_place.query(ob.points, distance_upper_bound=2 * tau)[0]).mean() >= 0.5:
                    m.setdefault("extra_b", []).append(k)
                    st.update(state="explained", by=m["a"])
        for k, st in list(dec.singles_a.items()):
            if st["state"] == "gone":
                oa = next(o for o in objs_a if o.object_id == k)
                if np.isfinite(old_place.query(oa.points, distance_upper_bound=2 * tau)[0]).mean() >= 0.5:
                    m.setdefault("extra_a", []).append(k)
                    st.update(state="explained", by=m["a"])
    dec.singles_a = {k: v for k, v in dec.singles_a.items() if v["state"] != "explained"}
    dec.singles_b = {k: v for k, v in dec.singles_b.items() if v["state"] != "explained"}

    for m in dec.matches:
        m["llr_gone_a"] = round(float(gone[m["i"]]), 2)
        m["llr_added_b"] = round(float(added[m["j"]]), 2)
    return dec


def self_consistency(objs, own: SceneRenderer, groups_own: np.ndarray, cfg: DecisionConfig = DecisionConfig()):
    """Is each object explained by its OWN visit? Views of the same visit from other view groups
    should not see through it. Objects they do see through (tempered, at the reporting odds)
    are reconstruction ghosts (e.g. a few frames with a bad pose) and must not be used as
    evidence of change. Returns {object_id: tempered self-LLR of 'absent'}."""
    out = {}
    for i, o in enumerate(objs):
        mine = {int(groups_own[f]) for f in o.frames}
        l, n, _ = own.hypothesis_llr(i, None, exclude_groups=mine)
        out[o.object_id] = dict(llr_absent_own=round(l, 2), groups=n, ghost=bool(cfg.eta * l > np.log(cfg.odds_min)))
    return out


def to_association(dec: Decision, odds_min: float) -> Association:
    """Adapter for the event builder: rigid_fit for moves, unchanged_lock for identity pairs."""
    matches = []
    for m in dec.matches:
        if m["kind"] == "moved":
            matches.append(dict(a=m["a"], b=m["b"], kind="rigid_fit", cost=-m["margin"], ambiguous_with=m["ambiguous_with"],
                                margin=m["margin"], uncertain=m["margin"] < np.log(odds_min), terms=m["terms"],
                                extra_a=m.get("extra_a", []), extra_b=m.get("extra_b", [])))
        else:
            matches.append(dict(a=m["a"], b=m["b"], kind="unchanged_lock", margin=m["margin"]))
    fits = {}
    for m in dec.matches:
        if m["kind"] == "moved":
            pf = dec.fits[(m["a"], m["b"])]
            pf.R, pf.t = m["R"], m["t"]
            fits[(m["a"], m["b"])] = pf
    return Association(matches, list(dec.singles_a), list(dec.singles_b), fits, dec.singles_a, dec.singles_b)
