"""Object-level appearance identity.

Each object gets one embedding: the mean of L2-normalised DINOv2 patch features
inside its masks over all its frames. Cosine similarity of before/after embeddings
is turned into an identity log-likelihood ratio that is calibrated on the scene
itself (same-place pairs = positives, different objects = negatives).
(A point-level variant was tried and rejected: patch similarity on uniform floors
and rugs is too noisy.)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F



class DenseFeatures:
    def __init__(self, model_id: str = "facebook/dinov2-base", device: str = "cuda", upscale: float = 2.0):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(model_id).to(device).eval().half()
        self.device = device
        self.upscale = upscale

    @torch.inference_mode()
    def __call__(self, images: np.ndarray) -> torch.Tensor:
        """Returns (N, C, h, w) L2-normalised patch features (fp16, on device)."""
        H, W = images.shape[1:3]
        h, w = int(round(H * self.upscale / 14)), int(round(W * self.upscale / 14))
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        out = []
        for im in images:
            x = torch.from_numpy(im).to(self.device).permute(2, 0, 1)[None].float() / 255
            x = F.interpolate(x, (h * 14, w * 14), mode="bilinear", align_corners=False)
            x = ((x - mean) / std).half()
            tok = self.model(pixel_values=x).last_hidden_state[:, 1:]
            f = tok.reshape(1, h, w, -1).permute(0, 3, 1, 2)
            out.append(F.normalize(f.float(), dim=1).half())
        return torch.cat(out)

    def free(self):
        del self.model
        torch.cuda.empty_cache()


def object_embeddings(objs, feats: torch.Tensor, hw) -> dict:
    """Mean L2-normalised patch feature inside each object's masks (all its frames)."""
    H, W = hw
    h, w = feats.shape[2:]
    out = {}
    for o in objs:
        acc, n = None, 0
        for x in o.obs:
            m = torch.from_numpy(cv2_resize(x.mask.astype(np.float32), (w, h))).to(feats.device) > 0.5
            if m.sum() == 0:
                continue
            f = feats[x.frame][:, m].float().mean(1)
            acc = f if acc is None else acc + f
            n += 1
        if acc is not None:
            out[o.object_id] = F.normalize(acc / n, dim=0).cpu().numpy()
    return out


def cv2_resize(a, size):
    import cv2

    return cv2.resize(a, size, interpolation=cv2.INTER_AREA)


def appearance_model(emb_a: dict, emb_b: dict, objs_a, objs_b, tau: float):
    """Self-calibrated identity log-likelihood ratio from embedding cosine similarity.

    Positives: before/after objects at the same place (most of a room does not change).
    Negatives: same-visit pairs of different objects (certainly different physical things).
    Returns f(cos) -> clipped log N(cos; pos) - log N(cos; neg), plus the fitted stats."""
    from scipy.spatial import cKDTree

    pos, neg = [], []
    for oa in objs_a:
        if oa.object_id not in emb_a:
            continue
        ta = cKDTree(oa.points)
        for ob in objs_b:
            if ob.object_id not in emb_b:
                continue
            d = ta.query(ob.points, distance_upper_bound=tau)[0]
            if np.isfinite(d).mean() > 0.5:
                pos.append(float(emb_a[oa.object_id] @ emb_b[ob.object_id]))
    for embs in (emb_a, emb_b):
        keys = list(embs)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                neg.append(float(embs[keys[i]] @ embs[keys[j]]))
    pos, neg = np.array(pos), np.array(neg)
    if len(pos) < 3 or len(neg) < 3:
        return (lambda c: 0.0), dict(n_pos=len(pos), n_neg=len(neg))
    mp, sp = float(np.median(pos)), max(1.4826 * float(np.median(np.abs(pos - np.median(pos)))), 0.02)
    mn, sn = float(np.median(neg)), max(1.4826 * float(np.median(np.abs(neg - np.median(neg)))), 0.02)

    def llr(c):
        lp = -0.5 * ((c - mp) / sp) ** 2 - np.log(sp)
        ln = -0.5 * ((c - mn) / sn) ** 2 - np.log(sn)
        return float(np.clip(lp - ln, -4.0, 4.0))

    return llr, dict(n_pos=int(len(pos)), n_neg=int(len(neg)), pos=(mp, sp), neg=(mn, sn))
