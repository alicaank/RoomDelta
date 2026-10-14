"""Per-visit object inventory from two complementary proposal sources.

* Semantic: an open-vocabulary detector (text prompts) -> box prompts -> masks.
* Geometric: clusters of visibility-test conflicts are projected into the
  frames they came from and used as box/point prompts, so an object whose
  category is missing from the prompt list is still segmented.

Per-frame masks are lifted to 3D with the visit's depth and merged across
frames by 3D voxel overlap (not by 2D connectivity), giving one object per
physical thing with its supporting frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

from .geometry import VisitMap

# Chunks are queried separately: Grounding DINO merges phrases that co-occur in
# one query ("chair armchair"), so confusable categories live in different chunks.
PROMPT_CHUNKS = [["chair", "table", "suitcase", "plate", "vase", "pillow", "lamp", "box"],
                 ["armchair", "sofa", "backpack", "bowl", "flowers", "plant", "remote control", "book"],
                 ["bag", "bottle", "shoe", "basket", "cup"]]


# label pairs a detector confuses for the same physical object
COMPATIBLE = [{"chair", "armchair"}, {"sofa", "armchair"}, {"vase", "flowers"}, {"flowers", "plant"},
              {"vase", "plant"}, {"bag", "backpack"}, {"bag", "suitcase"}, {"plate", "bowl"}]


def label_compat(a: str, b: str) -> float:
    """1 for the same label, 0.8 for commonly confused labels, 0.6 if one is unknown, else 0."""
    if a == b:
        return 1.0
    if "unidentified object" in (a, b):
        return 0.6
    return 0.8 if {a, b} in COMPATIBLE else 0.0


@dataclass
class MaskObs:
    frame: int
    mask: np.ndarray       # HxW bool (processed image)
    label: str
    score: float
    source: str            # "semantic" | "geometric"
    vox: set = field(default_factory=set)


@dataclass
class Object3D:
    object_id: str
    visit: str
    label: str
    label_scores: dict
    points: np.ndarray         # (N,3) shared-world, voxel-downsampled
    colors: np.ndarray         # (N,3) uint8
    obs: list[MaskObs]
    sources: set

    @property
    def frames(self):
        return sorted({o.frame for o in self.obs})

    @property
    def centroid(self):
        return self.points.mean(0)


class Segmenter:
    """Grounding DINO (text -> boxes) + SAM 2.1 (box/points -> masks).

    """

    def __init__(self, det_id="IDEA-Research/grounding-dino-base", sam_id="facebook/sam2.1-hiera-large", device="cuda"):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor

        self.device = device
        self.dp = AutoProcessor.from_pretrained(det_id)
        self.dm = AutoModelForZeroShotObjectDetection.from_pretrained(det_id).to(device).eval()
        self.sp = Sam2Processor.from_pretrained(sam_id)
        self.sm = Sam2Model.from_pretrained(sam_id).to(device).eval()

    @torch.inference_mode()
    def detect(self, image: np.ndarray, chunks: list[list[str]], box_thr=0.3, text_thr=0.25):
        from PIL import Image

        im = Image.fromarray(image)
        boxes, labels, scores = [], [], []
        for chunk in chunks:
            text = ". ".join(chunk) + "."
            inp = self.dp(images=im, text=text, return_tensors="pt").to(self.device)
            out = self.dm(**inp)
            r = self.dp.post_process_grounded_object_detection(out, inp.input_ids, threshold=box_thr, text_threshold=text_thr,
                                                               target_sizes=[im.size[::-1]])[0]
            for b, l, s in zip(r["boxes"].cpu().numpy(), r["text_labels"], r["scores"].cpu().numpy()):
                l = _canonical_label(l, chunk)
                if l is None:
                    continue
                boxes.append(b); labels.append(l); scores.append(float(s))
        return np.array(boxes).reshape(-1, 4), labels, np.array(scores)

    @torch.inference_mode()
    def segment(self, image: np.ndarray, boxes=None, points=None):
        """boxes: (K,4) xyxy; points: list of (M,2) positive points (one list per prompt)."""
        from PIL import Image

        im = Image.fromarray(image)
        kw = {}
        if boxes is not None:
            kw["input_boxes"] = [[list(map(float, b)) for b in boxes]]
        if points is not None:
            kw["input_points"] = [[[list(map(float, p)) for p in pts] for pts in points]]
            kw["input_labels"] = [[[1] * len(pts) for pts in points]]
        inp = self.sp(images=im, return_tensors="pt", **kw).to(self.device)
        out = self.sm(**inp, multimask_output=False)
        masks = self.sp.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"])[0][:, 0].numpy()
        return masks.astype(bool), out.iou_scores[0, :, 0].float().cpu().numpy()

    def free(self):
        del self.dm, self.sm
        torch.cuda.empty_cache()


def _canonical_label(text_label: str, prompts: list[str]):
    """Map a detector phrase to one prompt by whole-phrase match; ambiguous merges are dropped."""
    t = f" {text_label.strip()} "
    hits = [p for p in prompts if f" {p} " in t]
    if len(hits) == 1:
        return hits[0]
    return None


def dedupe_masks(obs: list[MaskObs], contain_thr=0.8):
    """Within a frame, a mask mostly contained in a stronger one is the same thing seen twice
    (duplicate detections, part masks, or a geometric prompt landing on a detected object).
    The kept mask inherits the label of the semantic one and records both sources."""
    keep: list[MaskObs] = []
    order = sorted(obs, key=lambda o: (o.source != "semantic", -o.mask.sum()))
    for o in order:
        dup = None
        for k in keep:
            if k.frame != o.frame:
                continue
            inter = np.logical_and(o.mask, k.mask).sum()
            if inter / max(1, min(o.mask.sum(), k.mask.sum())) >= contain_thr:
                dup = k
                break
        if dup is None:
            keep.append(o)
        else:
            if o.source not in dup.source.split("+"):
                dup.source = dup.source + "+" + o.source
            dup.score = max(dup.score, o.score)
    return keep


def lift_mask(m: VisitMap, i: int, mask: np.ndarray, erode: int = 2):
    if erode:
        mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)).astype(bool)
    P, uv = m.backproject(i, mask=mask)
    return P, uv


def semantic_proposals(m: VisitMap, seg: Segmenter, chunks=PROMPT_CHUNKS, max_area_frac=0.6) -> list[MaskObs]:
    H, W = m.hw
    obs = []
    for i in range(m.n):
        boxes, labels, scores = seg.detect(m.images[i], chunks)
        if len(boxes) == 0:
            continue
        masks, iou = seg.segment(m.images[i], boxes=boxes)
        for mk, l, s, q in zip(masks, labels, scores, iou):
            if mk.sum() < 80 or mk.mean() > max_area_frac:
                continue
            obs.append(MaskObs(i, mk, l, float(s) * float(q), "semantic"))
    return obs


def geometric_proposals(m: VisitMap, seg: Segmenter, frames: np.ndarray, uv: np.ndarray,
                        cluster: np.ndarray, min_pts: int = 40) -> list[MaskObs]:
    """SAM prompts from 3D change clusters, in the frames those points came from."""
    obs = []
    for c in np.unique(cluster[cluster >= 0]):
        sel = cluster == c
        for i in np.unique(frames[sel]):
            q = uv[sel & (frames == i)]
            if len(q) < min_pts:
                continue
            lo, hi = np.percentile(q, [3, 97], axis=0)
            box = np.array([lo[0], lo[1], hi[0], hi[1]])
            mk, iou = seg.segment(m.images[i], boxes=box[None])
            # keep the mask only if it actually explains the change pixels
            inside = mk[0][q[:, 1].astype(int), q[:, 0].astype(int)].mean()
            if inside < 0.5 or mk[0].sum() < 80:
                continue
            obs.append(MaskObs(int(i), mk[0], "unidentified object", float(iou[0]) * inside, f"geometric:{int(c)}"))
    return obs


def build_objects(m: VisitMap, obs: list[MaskObs], voxel: float, prefix: str, overlap_thr: float = 0.3,
                  min_frames: int = 2, min_voxels: int = 20) -> list[Object3D]:
    """Merge per-frame masks into 3D objects by voxel overlap (union-find)."""
    obs = dedupe_masks(obs)
    lifted = {}
    for o in obs:
        lifted[id(o)] = lift_mask(m, o.frame, o.mask)
        o.vox = set(map(tuple, np.floor(lifted[id(o)][0] / voxel).astype(np.int64)))
    obs = [o for o in obs if len(o.vox) >= 5]
    n = len(obs)
    parent = list(range(n))
    frames_of = {k: {obs[k].frame} for k in range(n)}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # pairs ordered by overlap so strong evidence merges first
    cand = []
    for a in range(n):
        for b in range(a + 1, n):
            if obs[a].frame == obs[b].frame:
                continue
            inter = len(obs[a].vox & obs[b].vox)
            if inter == 0:
                continue
            ov = inter / min(len(obs[a].vox), len(obs[b].vox))
            if ov >= overlap_thr and label_compat(obs[a].label, obs[b].label) >= 0.6:
                cand.append((ov, a, b))
    for ov, a, b in sorted(cand, reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        # do not chain two masks of the same frame into one object (two chairs side by side)
        if frames_of[ra] & frames_of[rb]:
            continue
        parent[rb] = ra
        frames_of[ra] |= frames_of.pop(rb)

    groups = {}
    for k in range(n):
        groups.setdefault(find(k), []).append(obs[k])
    objects = []
    for g in groups.values():
        frames = {o.frame for o in g}
        geometric_only = all("semantic" not in o.source for o in g)
        if len(frames) < min_frames and not geometric_only:
            continue
        pts = points_from_masks(m, g, voxel, lifted, min_voxels)
        if pts is None:
            continue
        P, C = pts
        scores = {}
        for o in g:
            if o.label != "unidentified object":
                scores[o.label] = scores.get(o.label, 0) + o.score
        label = max(scores, key=scores.get) if scores else "unidentified object"
        tot = sum(scores.values()) or 1
        srcs = {x.split(":")[0] for o in g for x in o.source.split("+")}
        objects.append(Object3D("", m.name, label, {k: v / tot for k, v in scores.items()}, P, C, g, srcs))
    objects.sort(key=lambda o: (o.label, -len(o.points)))
    for k, o in enumerate(objects):
        o.object_id = f"{prefix}{k:02d}"
    return objects


def points_from_masks(m: VisitMap, obs: list[MaskObs], voxel: float, lifted: dict | None = None, min_voxels: int = 20):
    """Surface samples of one object from all its masks.

    Each mask is lifted to 3D with the visit's depth. When an object has three or more
    masks, only voxels supported by at least two of them are kept, which removes the
    background that bleeds in at mask edges. Returns (points, colours) or None.
    """
    lifted = lifted or {}
    lifts = [lifted.get(id(o)) or lift_mask(m, o.frame, o.mask) for o in obs]
    keys = [np.floor(P / voxel).astype(np.int64) for P, _ in lifts]
    cnt = {}
    for k in keys:
        for v in set(map(tuple, k)):
            cnt[v] = cnt.get(v, 0) + 1
    need = 2 if len(obs) >= 3 else 1
    if sum(c >= need for c in cnt.values()) < min_voxels:
        return None
    pts, cols = [], []
    for o, (P, uv), k in zip(obs, lifts, keys):
        sel = np.fromiter((cnt[tuple(v)] >= need for v in k), bool, len(k))
        pts.append(P[sel])
        cols.append(m.images[o.frame][uv[sel, 1].astype(int), uv[sel, 0].astype(int)])
    P, C = np.concatenate(pts), np.concatenate(cols)
    _, first = np.unique(np.floor(P / (voxel / 2)).astype(np.int64), axis=0, return_index=True)
    return P[first], C[first]


def reprojection_containment(m: VisitMap, src: Object3D, dst: Object3D, eps_rel: float = 0.08) -> float:
    """Fraction of src's points that, seen from dst's frames, fall inside dst's masks.

    Only points that are actually visible in that frame (depth agrees within
    eps) count, so a chair hidden behind another chair is not "contained".
    """
    hit = tot = 0
    for o in dst.obs:
        i = o.frame
        Pc = src.points @ m.w2c[i][:3, :3].T + m.w2c[i][:3, 3]
        z = Pc[:, 2]
        uv = (Pc @ m.K[i].T)[:, :2] / np.maximum(z[:, None], 1e-6)
        H, W = m.hw
        u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        inb = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[inb], v[inb], z[inb]
        vis = np.abs(m.depth[i][v, u] - z) < eps_rel * z
        tot += int(vis.sum())
        hit += int((vis & o.mask[v, u]).sum())
    return hit / tot if tot >= 30 else 0.0


def merge_duplicates(objects: list[Object3D], voxel: float, m: VisitMap | None = None, thr: float = 0.5,
                     contain_thr: float = 0.6) -> list[Object3D]:
    """Merge objects of compatible class that are the same physical thing.

    The per-frame union-find refuses to merge two masks from one frame, which
    keeps adjacent chairs apart, but partial views of one object from opposite
    sides can share few voxels. Two tests, either sufficient:
      * 3D voxel overlap (relative to the smaller object) >= thr;
      * cross-view mask consistency: one object's visible points land inside
        the other's masks in the other's frames (both directions checked).
    """
    def merged(oa, ob):
        scores = dict(oa.label_scores)
        for k, v in ob.label_scores.items():
            scores[k] = scores.get(k, 0) + v
        P = np.concatenate([oa.points, ob.points]); C = np.concatenate([oa.colors, ob.colors])
        _, first = np.unique(np.floor(P / (voxel / 2)).astype(np.int64), axis=0, return_index=True)
        tot = sum(scores.values()) or 1
        label = max(scores, key=scores.get) if scores else "unidentified object"
        return Object3D(oa.object_id, oa.visit, label, {k: v / tot for k, v in scores.items()},
                        P[first], C[first], oa.obs + ob.obs, oa.sources | ob.sources)

    changed = True
    while changed:
        changed = False
        vox = [set(map(tuple, np.floor(o.points / voxel).astype(np.int64))) for o in objects]
        for a in range(len(objects)):
            for b in range(a + 1, len(objects)):
                oa, ob = objects[a], objects[b]
                if label_compat(oa.label, ob.label) < 0.6:
                    continue
                inter = len(vox[a] & vox[b])
                same = inter / max(1, min(len(vox[a]), len(vox[b]))) >= thr
                if not same and m is not None:
                    # cheap gate: bounding boxes must be close
                    lo_a, hi_a = np.percentile(oa.points, [5, 95], axis=0)
                    lo_b, hi_b = np.percentile(ob.points, [5, 95], axis=0)
                    gap = np.maximum(0, np.maximum(lo_a - hi_b, lo_b - hi_a))
                    if np.linalg.norm(gap) < 2 * voxel and not ({o.frame for o in oa.obs} & {o.frame for o in ob.obs}):
                        c1, c2 = reprojection_containment(m, oa, ob), reprojection_containment(m, ob, oa)
                        same = 0.5 * (c1 + c2) >= contain_thr and min(c1, c2) >= 0.4
                if not same:
                    continue
                objects[a] = merged(oa, ob)
                objects.pop(b)
                changed = True
                break
            if changed:
                break
    prefix = objects[0].object_id[:-2] if objects else ""
    objects.sort(key=lambda o: (o.label, -len(o.points)))
    for k, o in enumerate(objects):
        o.object_id = f"{prefix}{k:02d}"
    return objects


def evidence_fractions(obj: Object3D, vis, radius: float) -> dict:
    """Visibility-status mix of the surface samples that lie on this object."""
    from scipy.spatial import cKDTree

    d, j = cKDTree(vis.pts.P).query(obj.points, distance_upper_bound=radius)
    st = vis.status[j[np.isfinite(d)]]
    n = max(1, len(st))
    f = {name: float((st == k).sum() / n) for k, name in
         {0: "compatible", 1: "free_space_conflict", 2: "occluded", 3: "not_observed", 4: "weak_conflict"}.items()}
    f["n_samples"] = int(len(st))
    observed = f["compatible"] + f["free_space_conflict"] + f["weak_conflict"]
    f["conflict_of_observed"] = f["free_space_conflict"] / observed if observed > 0 else 0.0
    return f
