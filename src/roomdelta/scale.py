"""Metric scale for an RGB-only reconstruction.

DA3METRIC-LARGE predicts monocular metric depth (metres = focal * output / 300,
focal in pixels of the image it saw). Running it on the same processed
keyframes and dividing by the multi-view reconstruction's depth gives one
metres-per-unit ratio per frame; the median over frames of both visits is the
scale and the spread across frames is its uncertainty. This is a *learned*
scale, reported as approximate and cross-checked against the handheld camera
height it implies.
"""
from __future__ import annotations

import gc

import numpy as np

from .geometry import VisitMap


def metric_scale(maps: list[VisitMap], model_id: str = "depth-anything/DA3METRIC-LARGE", device: str = "cuda",
                 min_quality_pct: float = 30.0) -> dict:
    """``maps`` must share one coordinate frame (e.g. before map and aligned after map)."""
    import torch
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(model_id).to(device).eval()
    ratios, per_frame = [], []
    for m in maps:
        H, W = m.hw
        with torch.inference_mode():
            pred = model.inference([im for im in m.images], process_res=max(H, W), process_res_method="upper_bound_resize")
        assert pred.depth.shape[1:] == (H, W)
        for i in range(m.n):
            f = 0.5 * (m.K[i, 0, 0] + m.K[i, 1, 1])
            metric = f * pred.depth[i] / 300.0
            v = m.valid[i] & np.isfinite(metric) & (metric > 0)
            if hasattr(pred, "conf") and pred.conf is not None:
                v &= pred.conf[i] >= np.percentile(pred.conf[i], min_quality_pct)
            if v.sum() < 500:
                continue
            r = float(np.median(metric[v] / m.depth[i][v]))
            ratios.append(r)
            per_frame.append(dict(visit=m.name, frame=i, ratio=round(r, 4), median_metric_depth_m=round(float(np.median(metric[v])), 3)))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    ratios = np.asarray(ratios)
    s = float(np.median(ratios))
    lo, hi = np.percentile(ratios, [16, 84])
    return dict(meters_per_unit=s, rel_spread=float((hi - lo) / 2 / s), p16=float(lo), p84=float(hi), n_frames=int(len(ratios)),
                model_id=model_id, per_frame=per_frame)
