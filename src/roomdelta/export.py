"""Export the shared 3D representation in standard formats.

Frame: floor plane z = 0, z up, metres (scene units when no metric scale exists).
  scene_before.ply   before visit, coloured points
  scene_after.ply    after visit, aligned into the same frame, coloured points
  changes.ply        points of every changed object: orange = before location, blue = after location
  changes_3d.json    per change: type, label, before/after 3D boxes, displacement, rotation, description
These open in any point-cloud tool (MeshLab, CloudCompare, Blender, Open3D).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .colors import AFTER as AFTER_RGB, BEFORE as BEFORE_RGB
from .geometry import VisitMap


def floor_frame(floor, mpu: float):
    e1, e2, up = floor.basis()

    def to_frame(P):
        P = np.atleast_2d(P)
        return np.stack([P @ e1, P @ e2, floor.height(P)], 1) * mpu

    return to_frame


def _visit_cloud(m: VisitMap, stride: int = 2):
    P, C = [], []
    for i in range(m.n):
        p, uv = m.backproject(i, stride=stride)
        P.append(p); C.append(m.images[i][uv[:, 1].astype(int), uv[:, 0].astype(int)])
    return np.concatenate(P), np.concatenate(C)


def _write_ply(path: Path, P: np.ndarray, C: np.ndarray, voxel_m: float):
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(np.asarray(C, np.float64) / 255.0)
    if voxel_m > 0:
        pc = pc.voxel_down_sample(voxel_m)
    o3d.io.write_point_cloud(str(path), pc, write_ascii=False, compressed=False)
    return len(pc.points)


def _box(Q):
    lo, hi = np.percentile(Q, [5, 95], axis=0)
    return dict(center=((lo + hi) / 2).round(3).tolist(), size=(hi - lo).round(3).tolist())


def export_scene(out: Path, A: VisitMap, B: VisitMap, objs: dict, events: list, floor, scale, voxel_m: float = 0.01) -> dict:
    out = Path(out)
    mpu = scale.meters_per_unit or 1.0
    T = floor_frame(floor, mpu)
    n = {}
    for tag, m in (("before", A), ("after", B)):
        P, C = _visit_cloud(m)
        n[tag] = _write_ply(out / f"scene_{tag}.ply", T(P), C, voxel_m)
    O = {o.object_id: o for t in "AB" for o in objs[t]}
    P_ch, C_ch, changes = [], [], []
    for e in events:
        if e.event_type == "unchanged":
            continue
        rec = dict(event_id=e.event_id, type=e.event_type, label=e.semantic_label, description=e.description,
                   identity=e.identity_quality, limitations=e.limitations)
        for side, ids, rgb in (("before", e.before_object_ids, BEFORE_RGB), ("after", e.after_object_ids, AFTER_RGB)):
            if ids:
                Q = T(np.concatenate([O[k].points for k in ids]))
                rec[f"{side}_box_m" if scale.meters_per_unit else f"{side}_box"] = _box(Q)
                P_ch.append(Q); C_ch.append(np.tile(rgb, (len(Q), 1)))
        if e.displacement_units is not None:
            rec["displacement_m" if scale.meters_per_unit else "displacement_units"] = round(float(e.displacement_units * mpu), 3)
        if e.rotation_deg is not None:
            rec["rotation_deg"] = round(float(e.rotation_deg), 1)
        changes.append(rec)
    if P_ch:
        n["changes"] = _write_ply(out / "changes.ply", np.concatenate(P_ch), np.concatenate(C_ch), voxel_m / 2)
    meta = dict(frame="floor plane z=0, z up, x/y along the floor", units="m" if scale.meters_per_unit else "scene units",
                scale_status=scale.status, scale_source=scale.source, points=n, changes=changes)
    json.dump(meta, open(out / "changes_3d.json", "w"), indent=1)
    return meta
