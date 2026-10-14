"""Markdown report and per-event evidence images (before/after crops with masks)."""
from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import numpy as np

from .colors import AFTER, BEFORE
from .events import UNCHANGED
from .visibility import project


def _best_obs(obj, n=1):
    by_frame = {}
    for o in obj.obs:
        if o.frame not in by_frame or o.mask.sum() > by_frame[o.frame].mask.sum():
            by_frame[o.frame] = o
    return sorted(by_frame.values(), key=lambda o: -o.mask.sum())[:n]


def _panel(img, masks, title, color, w=420):
    im = img.copy()
    over = im.copy()
    for m in masks:
        over[m] = (0.5 * over[m] + 0.5 * np.array(color)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(over, cnts, -1, color, 2)
    s = w / im.shape[1]
    over = cv2.resize(over, (w, int(im.shape[0] * s)))
    cv2.rectangle(over, (0, 0), (w, 22), (0, 0, 0), -1)
    cv2.putText(over, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return over


def evidence_image(e, objs_a, objs_b, A, B):
    """Before: object mask. After: the same view direction, with the old surface projected (red) and the new mask (green)."""
    oa = {o.object_id: o for o in objs_a}
    ob = {o.object_id: o for o in objs_b}
    panels = []
    Pa = np.concatenate([oa[k].points for k in e.before_object_ids]) if e.before_object_ids else None
    if e.before_object_ids:
        best = _best_obs(oa[e.before_object_ids[0]])[0]
        masks = [o.mask for k in e.before_object_ids for o in oa[k].obs if o.frame == best.frame]
        panels.append(_panel(A.images[best.frame], masks, f"before f{best.frame}: {e.semantic_label}", BEFORE))
    if e.after_object_ids:
        best = _best_obs(ob[e.after_object_ids[0]])[0]
        masks = [o.mask for k in e.after_object_ids for o in ob[k].obs if o.frame == best.frame]
        p = _panel(B.images[best.frame], masks, f"after f{best.frame}", AFTER)
        panels.append(p)
    if Pa is not None and e.event_type != UNCHANGED:
        # the after frame that best sees the OLD location: shows what is there now
        best_j, best_n = None, 0
        for j in range(B.n):
            u_, v_, z = project(B, j, Pa)
            uv = np.stack([u_, v_], 1)
            H, W = B.hw
            inb = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            if inb.sum() > best_n:
                best_j, best_n = j, int(inb.sum())
        if best_j is not None and best_n > 20:
            u_, v_, z = project(B, best_j, Pa)
            uv = np.stack([u_, v_], 1)
            im = B.images[best_j].copy()
            H, W = B.hw
            for (u, v) in uv[(uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)][::2].astype(int):
                cv2.circle(im, (u, v), 1, BEFORE, -1)
            panels.append(_panel(im, [], f"after f{best_j}: old surface projected (orange)", BEFORE))
    if not panels:
        return None
    h = max(p.shape[0] for p in panels)
    panels = [np.pad(p, ((0, h - p.shape[0]), (0, 0), (0, 0))) for p in panels]
    return np.hstack(panels)


def write_report(out: Path, events, al, scale, vis, A, B, cfg, nav=None):
    d = pickle.load(open(out / "objects.pkl", "rb"))
    objs = d["objects"]
    ev_dir = out / "evidence"
    ev_dir.mkdir(exist_ok=True)
    lines = ["# What changed", ""]
    stat = al.stats
    lines += [f"Compared `{Path(str(A.meta.get('source'))).name}` with `{Path(str(B.meta.get('source'))).name}` "
              + (f"({A.n} frames each)." if A.n == B.n else f"({A.n} and {B.n} frames)."), "",
              f"- Alignment of the two videos: {'good' if al.ok else 'FAILED, so every change below is uncertain'}, "
              f"typical mismatch {scale.fmt(stat['median_inlier_residual'])} over {stat['n_inlier_pairs']} matched frame pairs.",
              "- Scale: " + ({"learned_metric": "estimated by a metric depth model", "approximate": "estimated from the camera height",
                                "calibrated": "from your measurement", "scene_units": "unknown, distances are in scene units"}[scale.status])
              + (f", ±{scale.rel_uncertainty:.0%}" if scale.rel_uncertainty else "")
              + (f" (this puts the camera {scale.cross_check['implied_camera_height_m']:.2f} m above the floor)"
                 if scale.cross_check.get("implied_camera_height_m") else "") + ".",
              ""]
    lines += ["## Changes", ""]
    shown = [e for e in events if e.event_type != UNCHANGED]
    if not shown:
        lines.append("No changes found.")
    for e in shown:
        lines.append(f"### {e.event_id}: {e.semantic_label} {e.event_type.replace('_', ' ')}")
        lines.append(e.description)
        if e.limitations:
            lines.append("")
            lines += [f"- Note: {x[0].upper() + x[1:]}." for x in e.limitations]
        img = evidence_image(e, objs["A"], objs["B"], A, B)
        if img is not None:
            cv2.imwrite(str(ev_dir / f"{e.event_id}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            lines.append(f"\n![{e.event_id}](evidence/{e.event_id}.jpg)")
        lines.append("")
    if nav is not None:
        r = nav["robot_radius_m"] + nav["clearance_m"]
        lines += ["## Is the old walking route still clear?", "",
                  f"The route is the path walked in the first video. The robot is modelled as a disc of {r:.2f} m radius "
                  f"({nav['robot_radius_m']:.2f} m plus {nav['clearance_m']:.2f} m clearance). This checks the footprint only, "
                  f"not full-body collisions.", ""]
        if nav["blocking_events"]:
            for b in nav["blocking_events"]:
                lines.append(f"- The {b['label']} ({b['event_id']}) now stands on the route.")
        else:
            lines.append("- None of the changes blocks the route.")
        lines.append(f"- {nav['unassessed_fraction']:.0%} of the route's footprint was not filmed in the second video. "
                     f"That part is unknown, not blocked.")
        if nav.get("radius_sensitivity"):
            lines += ["", "How the answer depends on the robot's size:", "",
                      "| robot radius (m) | changes on the route |", "|---|---|"]
            for x in nav["radius_sensitivity"]:
                lines.append(f"| {x['robot_radius_m']:.2f} | {', '.join(x['intersecting']) or 'none'} |")
        lines.append("")
    unchanged = [e for e in events if e.event_type == UNCHANGED]
    lines += ["## Unchanged", "", f"{len(unchanged)} objects: " + ", ".join(sorted({e.semantic_label for e in unchanged})) + ".", ""]
    (out / "report.md").write_text("\n".join(lines))
