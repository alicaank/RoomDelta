"""Temporally consistent object masks by video tracking (SAM 2.1 video predictor).

Per-frame detection gives masks only where the detector fires, and identity across
frames has to be inferred from 3D overlap. Tracking each object through the visit's
keyframes (which are in time order) gives (a) masks in frames where the detector
missed, (b) identity across frames from the tracker's memory. Tracking can drift onto
a neighbour, so every tracked mask is accepted only if its lifted 3D points fall inside
the object's (slightly enlarged) 3D extent: geometry validates the tracker.

Objects whose tracks cover the same pixels in several frames are one object (merged);
this replaces 3D-overlap / reprojection heuristics for duplicate detections.
"""
from __future__ import annotations

import numpy as np
import torch

from .geometry import VisitMap
from .objects import MaskObs, Object3D, label_compat, lift_mask, points_from_masks


class VideoTracker:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large", device: str = "cuda"):
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.device = device
        self.model = Sam2VideoModel.from_pretrained(model_id).to(device, dtype=torch.bfloat16).eval()
        self.proc = Sam2VideoProcessor.from_pretrained(model_id)

    @torch.inference_mode()
    def track(self, frames: list[np.ndarray], start: int, mask: np.ndarray) -> dict[int, np.ndarray]:
        """Propagate one mask from frame ``start`` forward and backward. Returns {frame: bool mask}."""
        H, W = frames[0].shape[:2]
        out = {}
        for reverse in (False, True):
            sess = self.proc.init_video_session(video=frames, inference_device=self.device, dtype=torch.bfloat16)
            self.proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=start, obj_ids=1,
                                                      input_masks=[mask.astype(np.float32)])
            for o in self.model.propagate_in_video_iterator(sess, start_frame_idx=start, reverse=reverse):
                m = self.proc.post_process_masks([o.pred_masks], original_sizes=[[H, W]], binarize=True)[0]
                mk = m[0, 0].cpu().numpy().astype(bool) if m.ndim == 4 else m[0].cpu().numpy().astype(bool)
                out[int(o.frame_idx)] = mk
        return out

    def free(self):
        del self.model
        torch.cuda.empty_cache()


def _box(P, grow: float, pad: float):
    lo, hi = np.percentile(P, [5, 95], axis=0)
    ext = hi - lo
    return lo - grow * ext - pad, hi + grow * ext + pad


def _rebuild(m: VisitMap, o: Object3D, obs: list[MaskObs], voxel: float) -> Object3D:
    """Recompute the object's surface samples from all its masks."""
    pts = points_from_masks(m, obs, voxel)
    if pts is None:
        return o
    return Object3D(o.object_id, o.visit, o.label, o.label_scores, pts[0], pts[1], obs, o.sources | {"tracked"})


def track_objects(m: VisitMap, objs: list[Object3D], tracker: VideoTracker, voxel: float, min_area: int = 80,
                  inside_frac: float = 0.6, merge_iou: float = 0.5, merge_frames: int = 2):
    """Extend every object with geometry-validated tracked masks, then merge objects that share tracks."""
    frames = [im for im in m.images]
    order = np.argsort(m.timestamps)
    assert np.all(order == np.arange(m.n)), "keyframes must be in time order"
    tracks, stats = {}, {}
    for o in objs:
        best = max(o.obs, key=lambda x: x.mask.sum())
        tr = tracker.track(frames, best.frame, best.mask)
        lo, hi = _box(o.points, 0.25, 2 * voxel)
        acc = {}
        for f, mk in tr.items():
            if mk.sum() < min_area:
                continue
            P, _ = lift_mask(m, f, mk)
            if len(P) < 10:
                continue
            if np.mean(np.all((P >= lo) & (P <= hi), axis=1)) >= inside_frac:
                acc[f] = mk
        tracks[o.object_id] = acc
        stats[o.object_id] = dict(tracked=len(tr), accepted=len(acc), detected=len({x.frame for x in o.obs}))

    # merge objects whose accepted tracks overlap in several frames (same physical object)
    ids = [o.object_id for o in objs]
    parent = {k: k for k in ids}

    def find(k):
        while parent[k] != k:
            k = parent[k]
        return k

    for a in range(len(objs)):
        for b in range(a + 1, len(objs)):
            oa, ob = objs[a], objs[b]
            if label_compat(oa.label, ob.label) < 0.6:
                continue
            ta, tb = tracks[oa.object_id], tracks[ob.object_id]
            common = [f for f in ta if f in tb]
            n = sum(np.logical_and(ta[f], tb[f]).sum() / max(1, np.logical_or(ta[f], tb[f]).sum()) > merge_iou for f in common)
            if n >= merge_frames:
                parent[find(ob.object_id)] = find(oa.object_id)

    by_root = {}
    for o in objs:
        by_root.setdefault(find(o.object_id), []).append(o)
    out = []
    for root, group in by_root.items():
        main = max(group, key=lambda o: len(o.points))
        obs = [x for o in group for x in o.obs]
        have = {x.frame for x in obs}
        for o in group:
            for f, mk in tracks[o.object_id].items():
                if f not in have:
                    obs.append(MaskObs(f, mk, main.label, 0.5, "tracked"))
                    have.add(f)
        scores = {}
        for o in group:
            for k, v in o.label_scores.items():
                scores[k] = scores.get(k, 0) + v * len(o.obs)
        tot = sum(scores.values()) or 1
        merged = Object3D(main.object_id, main.visit, max(scores, key=scores.get) if scores else main.label,
                          {k: v / tot for k, v in scores.items()}, main.points, main.colors, obs,
                          set().union(*[o.sources for o in group]))
        out.append(_rebuild(m, merged, obs, voxel))
        if len(group) > 1:
            stats[main.object_id]["merged"] = [o.object_id for o in group if o is not main]
    prefix = objs[0].object_id[:-2] if objs else ""
    out.sort(key=lambda o: (o.label, -len(o.points)))
    for k, o in enumerate(out):
        o.object_id = f"{prefix}{k:02d}"
    return out, stats
