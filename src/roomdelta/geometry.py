"""Reconstruction backends behind one adapter, and the per-visit map structure.

Every backend returns a :class:`VisitMap`: per-frame image, intrinsics,
OpenCV world-to-camera extrinsics, z-depth, a depth-quality score and a valid
mask, all expressed in the *processed* image coordinates (see
``CaptureResult.preprocessing_transform``). Visits are reconstructed
independently and are never fused before change detection.
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .capture import CaptureResult


@dataclass
class VisitMap:
    name: str
    images: np.ndarray        # N,H,W,3 uint8
    timestamps: np.ndarray    # N
    K: np.ndarray             # N,3,3
    w2c: np.ndarray           # N,4,4  (OpenCV: x right, y down, z forward)
    depth: np.ndarray         # N,H,W  z-depth in this visit's scene units
    quality: np.ndarray       # N,H,W  backend confidence (NOT a probability)
    valid: np.ndarray         # N,H,W  bool
    preprocessing_transform: np.ndarray  # 3x3 original video px -> processed px
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.images)

    @property
    def hw(self) -> tuple[int, int]:
        return self.depth.shape[1:3]

    def c2w(self) -> np.ndarray:
        return np.linalg.inv(self.w2c)

    def centers(self) -> np.ndarray:
        return self.c2w()[:, :3, 3]

    def transformed(self, s: float, R: np.ndarray, t: np.ndarray, name: str | None = None) -> "VisitMap":
        """Apply x' = s R x + t consistently to cameras AND depth."""
        S = np.eye(4)
        S[:3, :3] = s * R
        S[:3, 3] = t
        c2w = self.c2w()
        new_c2w = np.einsum("ij,njk->nik", S, c2w)
        # remove the scale from the rotation part; scale goes into depth
        new_c2w[:, :3, :3] /= s
        return VisitMap(
            name=name or self.name, images=self.images, timestamps=self.timestamps, K=self.K.copy(),
            w2c=np.linalg.inv(new_c2w), depth=self.depth * s, quality=self.quality, valid=self.valid,
            preprocessing_transform=self.preprocessing_transform, meta=dict(self.meta, sim3=dict(s=float(s), R=R.tolist(), t=list(map(float, t)))),
        )

    def backproject(self, i: int, mask: np.ndarray | None = None, stride: int = 1):
        """World points for frame i (optionally masked). Returns (P[N,3], uv[N,2])."""
        H, W = self.hw
        v, u = np.mgrid[0:H:stride, 0:W:stride]
        m = self.valid[i][::stride, ::stride]
        if mask is not None:
            m = m & mask[::stride, ::stride]
        u, v = u[m].astype(np.float64), v[m].astype(np.float64)
        z = self.depth[i][v.astype(int), u.astype(int)].astype(np.float64)
        Kinv = np.linalg.inv(self.K[i])
        rays = (Kinv @ np.stack([u, v, np.ones_like(u)])).T
        Pc = rays * z[:, None]
        c2w = np.linalg.inv(self.w2c[i])
        Pw = Pc @ c2w[:3, :3].T + c2w[:3, 3]
        return Pw, np.stack([u, v], 1)

    def save(self, path: str | Path):
        np.savez_compressed(path, name=self.name, images=self.images, timestamps=self.timestamps, K=self.K, w2c=self.w2c,
                            depth=self.depth, quality=self.quality, valid=self.valid,
                            preprocessing_transform=self.preprocessing_transform, meta=np.array([self.meta], dtype=object))

    @staticmethod
    def load(path: str | Path) -> "VisitMap":
        d = np.load(path, allow_pickle=True)
        return VisitMap(str(d["name"]), d["images"], d["timestamps"], d["K"], d["w2c"], d["depth"], d["quality"],
                        d["valid"], d["preprocessing_transform"], d["meta"][0])


def depth_edge_mask(depth: np.ndarray, rel_jump: float = 0.05, radius: int = 2) -> np.ndarray:
    """True near depth discontinuities (silhouettes), where z is unreliable."""
    import cv2

    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    dmax = cv2.dilate(depth, k)
    dmin = cv2.erode(depth, k)
    return (dmax - dmin) > rel_jump * np.maximum(dmin, 1e-6)


class DA3Backend:
    """Depth Anything 3 adapter (any DA3 checkpoint)."""

    def __init__(self, model_id: str = "depth-anything/DA3-BASE", device: str = "cuda"):
        self.model_id = model_id
        self.device = device

    def reconstruct(self, cap: CaptureResult, name: str, quality_percentile: float = 20.0) -> VisitMap:
        import torch
        from depth_anything_3.api import DepthAnything3

        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        model = DepthAnything3.from_pretrained(self.model_id).to(self.device).eval()
        imgs = [k.image for k in cap.keyframes]
        pw, ph = cap.proc_size
        with torch.inference_mode():
            pred = model.inference(imgs, process_res=max(pw, ph), process_res_method="upper_bound_resize")
        peak = torch.cuda.max_memory_allocated() / 2**30
        del model
        gc.collect()
        torch.cuda.empty_cache()

        N = len(imgs)
        assert pred.depth.shape[1:] == (ph, pw), (pred.depth.shape, (ph, pw))
        w2c = np.tile(np.eye(4), (N, 1, 1))
        w2c[:, :3, :] = pred.extrinsics
        depth = pred.depth.astype(np.float32)
        conf = pred.conf.astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        # Per-frame relative threshold: confidence is a ranking signal, not a probability.
        for i in range(N):
            thr = np.percentile(conf[i][valid[i]], quality_percentile)
            valid[i] &= conf[i] >= thr
            valid[i] &= ~depth_edge_mask(depth[i])
        meta = dict(backend="DA3", model_id=self.model_id, runtime_s=round(time.time() - t0, 2), peak_gpu_gb=round(peak, 2),
                    quality_percentile=quality_percentile, source=cap.path, capture=cap.report,
                    orig_size=cap.orig_size, proc_size=cap.proc_size)
        return VisitMap(name=name, images=np.stack(imgs), timestamps=np.array([k.timestamp for k in cap.keyframes]),
                        K=pred.intrinsics.astype(np.float64), w2c=w2c, depth=depth, quality=conf, valid=valid,
                        preprocessing_transform=cap.preprocessing_transform, meta=meta)


def _to_visit(pred, cap: CaptureResult, name: str, quality_percentile: float, meta: dict) -> VisitMap:
    N = len(cap.keyframes)
    w2c = np.tile(np.eye(4), (N, 1, 1))
    w2c[:, :3, :] = np.asarray(pred.extrinsics)[:, :3, :]
    depth = pred.depth.astype(np.float32)
    conf = pred.conf.astype(np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    for i in range(N):
        thr = np.percentile(conf[i][valid[i]], quality_percentile)
        valid[i] &= conf[i] >= thr
        valid[i] &= ~depth_edge_mask(depth[i])
    return VisitMap(name=name, images=np.stack([k.image for k in cap.keyframes]),
                    timestamps=np.array([k.timestamp for k in cap.keyframes]), K=np.asarray(pred.intrinsics, np.float64),
                    w2c=w2c, depth=depth, quality=conf, valid=valid, preprocessing_transform=cap.preprocessing_transform,
                    meta=dict(meta, source=cap.path, capture=cap.report, orig_size=cap.orig_size, proc_size=cap.proc_size,
                              quality_percentile=quality_percentile))


def reconstruct_shared_pose(backend: "DA3Backend", capA: CaptureResult, capB: CaptureResult,
                            quality_percentile: float = 20.0) -> tuple[VisitMap, VisitMap]:
    """Shared camera frame, separate content.

    1. One joint pass over both visits' keyframes estimates all camera poses and intrinsics in
       one frame (only the cameras are kept: changed objects are a small part of the views).
    2. Each visit's depth is re-inferred from its OWN images only, conditioned on those poses,
       so both depth sets live in one geometric frame and scale while every surface of a visit
       still comes from that visit alone (visits are never fused).
    """
    import torch
    from depth_anything_3.api import DepthAnything3

    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    model = DepthAnything3.from_pretrained(backend.model_id).to(backend.device).eval()
    pw, ph = capA.proc_size
    assert capB.proc_size == capA.proc_size
    imgs = [k.image for k in capA.keyframes] + [k.image for k in capB.keyframes]
    with torch.inference_mode():
        joint = model.inference(imgs, process_res=max(pw, ph), process_res_method="upper_bound_resize")
    na = len(capA.keyframes)
    ext = np.tile(np.eye(4, dtype=np.float32), (len(imgs), 1, 1))
    ext[:, :3, :] = joint.extrinsics
    Kj = joint.intrinsics.astype(np.float32)
    maps = []
    for cap, sl, name in ((capA, slice(0, na), "before"), (capB, slice(na, None), "after")):
        with torch.inference_mode():
            pred = model.inference([k.image for k in cap.keyframes], extrinsics=ext[sl], intrinsics=Kj[sl],
                                   process_res=max(pw, ph), process_res_method="upper_bound_resize")
        maps.append(_to_visit(pred, cap, name, quality_percentile,
                              dict(backend="DA3", model_id=backend.model_id, mode="shared_pose")))
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model
    gc.collect()
    torch.cuda.empty_cache()
    for m in maps:
        m.meta.update(runtime_s=round(time.time() - t0, 2), peak_gpu_gb=round(peak, 2))
    return maps[0], maps[1]


def get_backend(name: str, **kw):
    if name.lower().startswith("da3") or "depth-anything" in name:
        mid = name if "/" in name else {"da3-base": "depth-anything/DA3-BASE", "da3-large": "depth-anything/DA3-LARGE-1.1",
                                         "da3-giant": "depth-anything/DA3-GIANT-1.1"}[name.lower()]
        return DA3Backend(mid, **kw)
    raise ValueError(f"unknown geometry backend {name}")
