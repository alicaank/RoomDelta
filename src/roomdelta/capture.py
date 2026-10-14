"""Video decoding, quality checks and keyframe selection.

Keyframes are chosen for sharpness, overlap and viewpoint diversity rather than
pure temporal spacing: candidates are decoded at ``candidate_fps``; blurry ones
are rejected relative to the recording's own sharpness distribution; the rest
are selected greedily so that consecutive keyframes still overlap (tracked
feature survival) but differ in viewpoint (median optical-flow displacement).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Keyframe:
    index: int            # index in the decoded video stream
    timestamp: float      # seconds
    image: np.ndarray     # HxWx3 uint8 RGB, already resized to the processing size
    sharpness: float


@dataclass
class CaptureResult:
    path: str
    fps: float
    duration: float
    orig_size: tuple[int, int]          # (W, H) of the decoded video
    proc_size: tuple[int, int]          # (W, H) of keyframe images
    preprocessing_transform: np.ndarray  # 3x3 map from original pixels to processed pixels
    keyframes: list[Keyframe]
    report: dict = field(default_factory=dict)


def _proc_size(w: int, h: int, long_side: int, patch: int = 14) -> tuple[int, int]:
    s = long_side / max(w, h)
    pw = max(patch, int(round(w * s / patch)) * patch)
    ph = max(patch, int(round(h * s / patch)) * patch)
    return pw, ph


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _flow_stats(g0: np.ndarray, g1: np.ndarray) -> tuple[float, float]:
    """Return (median displacement in px, fraction of tracked features surviving)."""
    p0 = cv2.goodFeaturesToTrack(g0, maxCorners=400, qualityLevel=0.01, minDistance=8)
    if p0 is None or len(p0) < 10:
        return 0.0, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, winSize=(21, 21), maxLevel=4)
    ok = st.ravel() == 1
    if ok.sum() < 10:
        return float("inf"), ok.mean()
    d = np.linalg.norm((p1 - p0).reshape(-1, 2)[ok], axis=1)
    return float(np.median(d)), float(ok.mean())


def decode_candidates(path: str | Path, candidate_fps: float, phase: int = 0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / candidate_fps)))
    out = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == phase % step:
            out.append((i, i / fps, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        i += 1
    cap.release()
    return out, fps, max(n, i) / fps


def select_keyframes(
    path: str | Path,
    max_keyframes: int = 32,
    candidate_fps: float = 6.0,
    long_side: int = 504,
    blur_quantile: float = 0.25,
    min_flow_px: float = 6.0,
    min_survival: float = 0.5,
    phase: int = 0,
) -> CaptureResult:
    """``phase`` shifts candidate sampling (used to build disjoint keyframe sets from one video)."""
    cands, fps, duration = decode_candidates(path, candidate_fps, phase)
    if len(cands) < 4:
        raise RuntimeError(f"{path}: only {len(cands)} frames decoded")
    H, W = cands[0][2].shape[:2]
    pw, ph = _proc_size(W, H, long_side)
    T = np.diag([pw / W, ph / H, 1.0])

    small = []
    sharp = []
    for _, _, rgb in cands:
        g = cv2.cvtColor(cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        small.append(g)
        sharp.append(sharpness(g))
    sharp = np.asarray(sharp)
    # Blur is judged relative to this recording; motion blur shows up as a
    # lower tail of the Laplacian-variance distribution.
    blur_thr = np.quantile(sharp, blur_quantile) * 0.8
    usable = sharp >= blur_thr

    # Greedy forward pass: accept a candidate once it has moved enough relative
    # to the last keyframe; if overlap is about to break, take the sharpest
    # candidate seen since the last keyframe.
    chosen = [int(np.argmax(np.where(usable[: max(1, len(cands) // 10)], sharp[: max(1, len(cands) // 10)], -1)))]
    pending = []
    for j in range(chosen[0] + 1, len(cands)):
        if not usable[j]:
            continue
        disp, surv = _flow_stats(small[chosen[-1]], small[j])
        if surv < min_survival and pending:
            best = max(pending, key=lambda k: sharp[k])
            chosen.append(best)
            pending = [k for k in pending if k > best]
            disp, surv = _flow_stats(small[chosen[-1]], small[j])
        if disp >= min_flow_px:
            # pick the sharpest among recent candidates within the viewpoint step
            window = [k for k in pending[-2:] + [j]]
            best = max(window, key=lambda k: sharp[k])
            chosen.append(best)
            pending = [k for k in pending if k > best]
            if best != j:
                pending.append(j)
        else:
            pending.append(j)
    chosen = sorted(set(chosen))

    # Thin to the budget by repeatedly dropping the keyframe whose neighbours
    # are closest in viewpoint (keeps diversity and coverage).
    while len(chosen) > max_keyframes:
        gaps = []
        for a in range(1, len(chosen) - 1):
            d, _ = _flow_stats(small[chosen[a - 1]], small[chosen[a + 1]])
            gaps.append((d, -sharp[chosen[a]], a))
        _, _, drop = min(gaps)
        chosen.pop(drop)

    kfs = []
    for k in chosen:
        i, t, rgb = cands[k]
        img = cv2.resize(rgb, (pw, ph), interpolation=cv2.INTER_AREA)
        kfs.append(Keyframe(index=i, timestamp=t, image=img, sharpness=float(sharp[k])))

    report = {
        "num_candidates": len(cands),
        "num_blurry_rejected": int((~usable).sum()),
        "sharpness_median": float(np.median(sharp)),
        "num_keyframes": len(kfs),
        "keyframe_times": [round(k.timestamp, 3) for k in kfs],
    }
    return CaptureResult(
        path=str(path), fps=fps, duration=duration, orig_size=(W, H), proc_size=(pw, ph),
        preprocessing_transform=T, keyframes=kfs, report=report,
    )
