"""Structured change events from geometry + identity evidence, and the text report.

The language layer only *describes* events established here; it never
decides whether something happened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .association import Association
from .objects import Object3D
from .scene import Floor

MOVED, ADDED, REMOVED, UNCHANGED, UNCERTAIN, NEWLY_OBSERVED = (
    "moved", "added_to_observed_area", "removed_from_previous_location", "unchanged", "uncertain", "newly_observed")


@dataclass
class ScaleInfo:
    meters_per_unit: float | None
    status: str          # "scene_units" | "approximate" | "learned_metric" | "calibrated"
    source: str
    rel_uncertainty: float | None = None   # relative 1-sigma of meters_per_unit
    cross_check: dict = field(default_factory=dict)

    def fmt(self, d_units: float, unc_units: float | None = None) -> str:
        if self.meters_per_unit is None:
            s = f"{d_units:.2f} scene units"
            return s if unc_units is None else s + f" (±{unc_units:.2f})"
        cm = d_units * self.meters_per_unit * 100
        approx = "≈" if self.status in ("approximate", "learned_metric") else ""
        s = f"{approx}{cm:.0f} cm"
        if unc_units is not None:
            # geometric uncertainty and scale uncertainty, combined
            u = unc_units * self.meters_per_unit * 100
            if self.rel_uncertainty:
                u = float(np.hypot(u, cm * self.rel_uncertainty))
            s += f" (±{u:.0f} cm)"
        return s


@dataclass
class ChangeEvent:
    event_id: str
    event_type: str
    semantic_label: str
    before_object_ids: list
    after_object_ids: list
    before_location: list | None
    after_location: list | None
    estimated_rigid_transform: dict | None
    displacement_units: float | None
    rotation_deg: float | None
    scale_status: str
    alignment_quality: dict
    identity_quality: str
    geometric_evidence: dict
    supporting_frames: dict
    limitations: list = field(default_factory=list)
    description: str = ""
    witness_rays: list = field(default_factory=list)   # [(origin xyz, end xyz, visit observing)]

    def as_dict(self):
        return asdict(self)


def _joint_label(a: Object3D, b: Object3D) -> str:
    """One label for a matched pair: detector votes from both visits, weighted by support."""
    votes = {}
    for o in (a, b):
        for k, v in o.label_scores.items():
            votes[k] = votes.get(k, 0) + v * len(o.frames)
    return max(votes, key=votes.get) if votes else a.label


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _witness_rays(obj: Object3D, vis, observer, radius: float, k: int = 24, seed: int = 0):
    """Rays from observer cameras through the object's old surface to what they actually saw."""
    d, j = cKDTree(vis.pts.P).query(obj.points, distance_upper_bound=radius)
    idx = np.unique(j[np.isfinite(d)])
    idx = [i for i in idx if int(i) in vis.witness]
    if not idx:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, min(k, len(idx)), replace=False)
    C = observer.centers()
    rays = []
    for i in idx:
        p = vis.pts.P[i]
        for (jf, d_obs, z_proj) in vis.witness[int(i)][:2]:
            o = C[jf]
            end = o + (p - o) * (d_obs / max(z_proj, 1e-6))
            rays.append((o.tolist(), end.tolist(), p.tolist(), observer.name, int(jf)))
    return rays


def _anchor_name(a: Object3D, floor: Floor, scale: "ScaleInfo") -> str:
    if a.label != "table" or scale.meters_per_unit is None:
        return a.label
    top_m = np.percentile(floor.height(a.points), 95) * scale.meters_per_unit
    return "coffee table" if top_m < 0.6 else "dining table"


def _place(obj_pts: np.ndarray, anchors: list[Object3D], floor: Floor, scale: "ScaleInfo") -> str:
    """Short spatial description relative to large unchanged furniture."""
    if not anchors:
        return ""
    xy = floor.to_2d(obj_pts).mean(0)
    base = np.percentile(floor.height(obj_pts), 5)
    best, best_d = None, np.inf
    for a in anchors:
        axy = floor.to_2d(a.points)
        lo, hi = np.percentile(axy, [5, 95], axis=0)
        top = np.percentile(floor.height(a.points), 95)
        inside = np.all(xy >= lo) and np.all(xy <= hi)
        if inside and base > 0.5 * top and abs(base - top) < 0.25 * max(top, 1e-6) + 0.03:
            return f"on the {_anchor_name(a, floor, scale)}"
        d = np.linalg.norm(np.clip(xy, lo, hi) - xy)
        if d < best_d:
            best, best_d = a, d
    return f"next to the {_anchor_name(best, floor, scale)}" if best is not None else ""


def build_events(objs_a: list[Object3D], objs_b: list[Object3D], asc: Association, ev_a: dict, ev_b: dict,
                 vis_a, vis_b, A, B, alignment, floor: Floor, scale: ScaleInfo, tau: float, min_motion: float,
                 odds_min: float = 19.0) -> list[ChangeEvent]:
    """Turn the decision (``asc``: matches + per-object states with posterior margins) into events."""
    oa = {o.object_id: o for o in objs_a}
    ob = {o.object_id: o for o in objs_b}
    aq = dict(ok=alignment.ok, sigma_p95_units=alignment.sigma_p95, median_residual_units=alignment.stats["median_inlier_residual"],
              failure_reasons=alignment.stats["failure_reasons"])
    anchors_a = [o for o in objs_a if o.label in ("table", "sofa") and len(o.points) > 1500]
    radius = 0.75 * tau
    events: list[ChangeEvent] = []

    def frames(o: Object3D):
        return sorted({x.frame for x in o.obs})

    def evd(o, ev):
        e = ev.get(o.object_id, {})
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()}

    # ------------------------------------------------------------- matched objects
    moved = []
    for m in asc.matches:
        a, b = oa[m["a"]], ob[m["b"]]
        ids_a, ids_b = [m["a"]] + m.get("extra_a", []), [m["b"]] + m.get("extra_b", [])
        # only rigid-fit matches carry a motion; fits tried and rejected during association do not
        pf = asc.fits.get((m["a"], m["b"])) if m.get("kind") == "rigid_fit" else None
        common = dict(semantic_label=_joint_label(a, b),
                      before_object_ids=ids_a, after_object_ids=ids_b,
                      before_location=a.centroid.tolist(), after_location=b.centroid.tolist(),
                      scale_status=scale.status, alignment_quality=aq,
                      geometric_evidence=dict(before_surface=evd(a, ev_a), after_surface=evd(b, ev_b)),
                      supporting_frames=dict(before=frames(a), after=frames(b)))
        if pf is None:  # locked unchanged, or two sides of one object
            if m.get("kind") == "other_side":
                iq = (f"same object seen from opposite sides: {m['occluder_share']:.0%} of the unmatched surface is hidden "
                      f"behind it, footprints overlap")
                common["limitations"] = ["seen from opposite sides in the two videos; matched because one view is hidden right behind the other"]
            else:
                iq = "high (same place, same class, compatible geometry)"
            if m.get("extra_a") or m.get("extra_b"):
                iq += "; plus partial views from the opposite side (occlusion test)"
            events.append(ChangeEvent("", UNCHANGED, estimated_rigid_transform=None, displacement_units=0.0, rotation_deg=0.0,
                                      identity_quality=iq, **common))
            continue
        c = a.centroid
        d = float(np.linalg.norm(pf.R @ c + pf.t - c))
        rot = _rot_deg(pf.R)
        lim = []
        if pf.yaw_alternatives:
            lim.append("the shape is nearly symmetric, so its rotation is not reported")
            rot_rep = None
        else:
            rot_rep = rot
        ident = "ambiguous" if m.get("ambiguous_with") else (
            "high" if pf.feat_inliers >= 15 else "medium (shape fit after rigid motion; few appearance matches)")
        if m.get("ambiguous_with"):
            lim.append(f"it looks the same as {', '.join(m['ambiguous_with'])}, so which one went where is uncertain")
        rot_significant = rot_rep is not None and rot_rep > 15
        if d < min_motion and not rot_significant:
            events.append(ChangeEvent("", UNCHANGED, estimated_rigid_transform=None, displacement_units=d, rotation_deg=rot_rep,
                                      identity_quality=ident, limitations=lim, **common))
            continue
        if m.get("terms"):
            common["geometric_evidence"]["generative"] = dict(m["terms"], posterior_margin_nats=m.get("margin"))
        if m.get("extra_a") or m.get("extra_b"):
            lim.append(f"{len(m.get('extra_a', [])) + len(m.get('extra_b', []))} extra partial detection(s) belong to this move")
        ev = ChangeEvent("", MOVED, estimated_rigid_transform=dict(R=pf.R.tolist(), t=pf.t.tolist(), method=pf.method,
                                                                    feature_inliers=pf.feat_inliers,
                                                                    shape_fit=[round(pf.fit_ab, 3), round(pf.fit_ba, 3)]),
                         displacement_units=d, rotation_deg=rot_rep, identity_quality=ident, limitations=lim, **common)
        ev.witness_rays = _witness_rays(a, vis_a, B, radius)
        if m.get("uncertain"):
            ev.event_type = UNCERTAIN
            ev.limitations.append(f"a move is the most likely explanation, but not by a clear margin ({m['margin']:.1f} nats, below 95% odds)")
            events.append(ev)
            continue
        moved.append((ev, a, b, pf))

    # group objects that moved together (flowers + vase + bowl) into one event:
    # adjacent before the move, and each part's own reference point displaced
    # the same way (rotation is often ambiguous for round things, so compare
    # where each part went rather than full transforms). Transitive via union-find.
    parent = list(range(len(moved)))

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    for i in range(len(moved)):
        for j in range(i + 1, len(moved)):
            _, a1, _, p1 = moved[i]
            _, a2, _, p2 = moved[j]
            near = np.isfinite(cKDTree(a1.points).query(a2.points, distance_upper_bound=3 * tau)[0]).any()
            if not near:
                continue
            d1 = p1.R @ a1.centroid + p1.t - a1.centroid
            d2 = p2.R @ a2.centroid + p2.t - a2.centroid
            if np.linalg.norm(d1 - d2) < 2 * tau:
                parent[find(j)] = find(i)
    clusters_mv = {}
    for k in range(len(moved)):
        clusters_mv.setdefault(find(k), []).append(k)
    for group in clusters_mv.values():
        ev = moved[group[0]][0]
        if len(group) > 1:
            parts = [moved[k] for k in group]
            main = max(parts, key=lambda p: (p[3].feat_inliers, len(p[1].points)))[0]
            labels = []
            for p in parts:
                if p[1].label not in labels:
                    labels.append(p[1].label)
            main.semantic_label = " and ".join(labels)
            main.before_object_ids = [p[1].object_id for p in parts]
            main.after_object_ids = [p[2].object_id for p in parts]
            main.before_location = np.concatenate([p[1].points for p in parts]).mean(0).tolist()
            main.after_location = np.concatenate([p[2].points for p in parts]).mean(0).tolist()
            main.supporting_frames = dict(before=sorted({f for p in parts for f in frames(p[1])}),
                                          after=sorted({f for p in parts for f in frames(p[2])}))
            main.witness_rays = [r for p in parts for r in p[0].witness_rays][:48]
            main.limitations.append(f"{len(parts)} detected parts moved together and are reported as one change")
            events.append(main)
        else:
            events.append(ev)

    # ------------------------------------------------------------- unmatched before
    for k in asc.unmatched_a:
        a = oa[k]
        common = dict(semantic_label=a.label, before_object_ids=[k], after_object_ids=[], before_location=a.centroid.tolist(),
                      after_location=None, estimated_rigid_transform=None, displacement_units=None, rotation_deg=None,
                      scale_status=scale.status, alignment_quality=aq,
                      geometric_evidence=dict(before_surface=evd(a, ev_a)), supporting_frames=dict(before=frames(a), after=[]))
        st = asc.singles_a.get(k, dict(state="present", margin=0.0))
        common["geometric_evidence"]["generative"] = st
        gone_ok = st["state"] == "gone" and st["margin"] >= np.log(odds_min)
        weak_gone = st["state"] == "gone" and not gone_ok
        kind = REMOVED if gone_ok else (UNCHANGED if st["state"] == "present" else UNCERTAIN)
        if kind == REMOVED:
            e = ChangeEvent("", REMOVED, identity_quality="no supported match in the after recording",
                            limitations=["it may still be somewhere the second video did not film"], **common)
            e.witness_rays = _witness_rays(a, vis_a, B, radius)
        elif kind == UNCHANGED:
            e = ChangeEvent("", UNCHANGED, identity_quality="not re-detected; surface geometry compatible",
                            limitations=["not detected in the second video, but its spot looks the same"], **common)
        else:
            e = ChangeEvent("", UNCERTAIN, identity_quality="no supported match",
                            limitations=[f"removal is the most likely explanation, but not by a clear margin ({st['margin']:.1f} nats, below 95% odds)"]
                            if weak_gone else ["its old spot was hidden or not filmed in the second video, so removal cannot be confirmed"],
                            **common)
        events.append(e)

    # ------------------------------------------------------------- unmatched after
    for k in asc.unmatched_b:
        b = ob[k]
        common = dict(semantic_label=b.label, before_object_ids=[], after_object_ids=[k], before_location=None,
                      after_location=b.centroid.tolist(), estimated_rigid_transform=None, displacement_units=None, rotation_deg=None,
                      scale_status=scale.status, alignment_quality=aq,
                      geometric_evidence=dict(after_surface=evd(b, ev_b)), supporting_frames=dict(before=[], after=frames(b)))
        st = asc.singles_b.get(k, dict(state="present", margin=0.0))
        common["geometric_evidence"]["generative"] = st
        add_ok = st["state"] == "added" and st["margin"] >= np.log(odds_min)
        kind = ADDED if add_ok else UNCERTAIN if st["state"] == "added" else \
            (UNCHANGED if st["state"] == "present" else NEWLY_OBSERVED)
        if kind == UNCERTAIN:
            e = ChangeEvent("", UNCERTAIN, identity_quality="no supported match",
                            limitations=[f"addition is the most likely explanation, but not by a clear margin ({st['margin']:.1f} nats, below 95% odds)"],
                            **common)
            events.append(e)
            continue
        if kind == ADDED:
            e = ChangeEvent("", ADDED, identity_quality="no supported match in the before recording",
                            limitations=["the first video saw this spot empty"], **common)
            e.witness_rays = _witness_rays(b, vis_b, A, radius)
        elif kind == UNCHANGED:
            e = ChangeEvent("", UNCHANGED, identity_quality="detected only afterwards; surface geometry compatible with before",
                            limitations=["only detected in the second video; the geometry at its spot did not change"], **common)
        else:
            e = ChangeEvent("", NEWLY_OBSERVED, identity_quality="no supported match",
                            limitations=["the first video never had a clear view of this spot"], **common)
        events.append(e)

    if not alignment.ok:
        for e in events:
            if e.event_type in (MOVED, ADDED, REMOVED):
                e.limitations.append("the two videos could not be aligned reliably, so this is reported as uncertain")
                e.event_type = UNCERTAIN

    order = {MOVED: 0, ADDED: 1, REMOVED: 2, UNCERTAIN: 3, NEWLY_OBSERVED: 4, UNCHANGED: 5}
    events.sort(key=lambda e: (order[e.event_type], e.semantic_label))
    for i, e in enumerate(events):
        e.event_id = f"E{i:02d}"
        e.description = describe(e, oa, ob, anchors_a, floor, scale, tau)
    return events


def _route(where_a: str, where_b: str) -> str:
    if not where_a and not where_b:
        return ""
    if where_a == where_b:
        return f", staying {where_a}" if where_a.startswith("next") else f" {where_a.replace('on the', 'along the')}"
    return f" from {where_a or 'its old spot'} to {where_b or 'a new spot'}".replace("from next to", "from beside").replace("to next to", "to beside")


def describe(e: ChangeEvent, oa, ob, anchors, floor: Floor, scale: ScaleInfo, tau: float) -> str:
    the = f"The {e.semantic_label}"
    pa = np.concatenate([oa[k].points for k in e.before_object_ids]) if e.before_object_ids else None
    pb = np.concatenate([ob[k].points for k in e.after_object_ids]) if e.after_object_ids else None
    if e.event_type == MOVED:
        s = f"{the} moved {scale.fmt(e.displacement_units, tau)}"
        if e.rotation_deg is not None and e.rotation_deg > 15:
            s += f" and turned about {e.rotation_deg:.0f}°"
        s += _route(_place(pa, anchors, floor, scale), _place(pb, anchors, floor, scale)) + "."
        bs = e.geometric_evidence.get("before_surface", {})
        if bs.get("conflict_of_observed", 0) > 0.2:
            s += f" The second video sees straight through {bs['conflict_of_observed'] * 100:.0f}% of where it used to be."
        if e.identity_quality == "ambiguous":
            s += " An identical object also moved, so the two may be swapped."
        return s
    if e.event_type == REMOVED:
        where = _place(pa, anchors, floor, scale)
        return f"{the} is gone from its spot{' ' + where if where else ''}: the second video sees the empty space behind it, " \
               f"and it was not found anywhere else that was filmed."
    if e.event_type == ADDED:
        return f"A {e.semantic_label} now stands where the first video saw empty space."
    if e.event_type == UNCERTAIN:
        if e.limitations:
            return f"Unclear for the {e.semantic_label}: {e.limitations[0]}."
        return f"Unclear for the {e.semantic_label}: it was not found in the second video, but its old spot was hidden or not filmed."
    if e.event_type == NEWLY_OBSERVED:
        return f"{the} only appears in the second video, and the first video never had a clear view of its spot. " \
               f"It may have been there all along."
    return f"{the} did not change."
