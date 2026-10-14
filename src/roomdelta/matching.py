"""ALIKED + LightGlue features and matching, shared by alignment and object association."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch


@dataclass
class Features:
    kpts: list[np.ndarray]         # per frame, (M,2) pixel coords in processed image
    raw: list[dict]                # LightGlue feature dicts (on device)


class Matcher:
    def __init__(self, device: str = "cuda", max_kpts: int = 2048):
        from lightglue import ALIKED, LightGlue

        self.device = device
        self.extractor = ALIKED(max_num_keypoints=max_kpts, detection_threshold=0.01).eval().to(device)
        self.matcher = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval().to(device)
        self._cache = {}

    @torch.inference_mode()
    def extract(self, images: np.ndarray) -> Features:
        kp, raw = [], []
        for im in images:
            t = torch.from_numpy(im).permute(2, 0, 1).float().div(255).to(self.device)
            f = self.extractor.extract(t)
            raw.append(f)
            kp.append(f["keypoints"][0].cpu().numpy())
        return Features(kp, raw)

    @torch.inference_mode()
    def match(self, fa: dict, fb: dict, min_score: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
        """Matches between two frames' features (cached: the same frame pairs are matched by several stages)."""
        key = (id(fa), id(fb), min_score)
        if key not in self._cache:
            self._cache[key] = self._match(fa, fb, min_score)
        return self._cache[key]

    def _match(self, fa: dict, fb: dict, min_score: float):
        out = self.matcher({"image0": fa, "image1": fb})
        m = out["matches"][0].cpu().numpy()
        s = out["scores"][0].cpu().numpy()
        keep = s >= min_score
        return m[keep], s[keep]

    def free(self):
        self._cache.clear()
        del self.extractor, self.matcher
        torch.cuda.empty_cache()


def verify_epipolar(pa: np.ndarray, pb: np.ndarray, thr_px: float = 1.5) -> np.ndarray:
    """Boolean inlier mask from a fundamental-matrix MAGSAC fit."""
    if len(pa) < 12:
        return np.zeros(len(pa), bool)
    try:
        F, inl = cv2.findFundamentalMat(pa.astype(np.float64), pb.astype(np.float64), cv2.USAC_MAGSAC, thr_px, 0.999, 10000)
    except cv2.error:  # degenerate configurations can trip assertions inside USAC
        return np.zeros(len(pa), bool)
    if F is None or inl is None:
        return np.zeros(len(pa), bool)
    return inl.ravel().astype(bool)
