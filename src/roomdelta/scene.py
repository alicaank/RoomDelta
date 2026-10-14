"""Scene-level frame: floor plane and gravity direction from the before visit."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import VisitMap


@dataclass
class Floor:
    normal: np.ndarray   # unit "up" vector in shared world
    d: float             # plane: normal . x + d = 0
    inlier_frac: float

    def height(self, P: np.ndarray) -> np.ndarray:
        return P @ self.normal + self.d

    def basis(self):
        """Orthonormal (e1, e2, up) with e1/e2 spanning the floor."""
        up = self.normal
        a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = np.cross(up, a); e1 /= np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        return e1, e2, up

    def to_2d(self, P: np.ndarray) -> np.ndarray:
        e1, e2, _ = self.basis()
        return np.stack([P @ e1, P @ e2], 1)


def estimate_floor(m: VisitMap, scale: float, max_tilt_deg: float = 30.0) -> Floor:
    """Largest near-horizontal plane below the cameras.

    The phone is held roughly upright, so the mean camera "-y" axis is a coarse
    up direction; the floor is the dominant plane whose normal agrees with it
    and which has the cameras on its positive side.
    """
    import open3d as o3d

    c2w = m.c2w()
    up0 = -c2w[:, :3, 1].mean(0)
    up0 /= np.linalg.norm(up0)
    P = np.concatenate([m.backproject(i, stride=6)[0] for i in range(m.n)])
    C = m.centers()
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)).voxel_down_sample(0.01 * scale)
    rest = pc
    best = None
    for _ in range(6):
        if len(rest.points) < 500:
            break
        (a, b, c, d), inl = rest.segment_plane(0.01 * scale, 3, 2000)
        n = np.array([a, b, c])
        nn = np.linalg.norm(n)
        n, d = n / nn, d / nn
        if n @ up0 < 0:
            n, d = -n, -d
        tilt = np.degrees(np.arccos(np.clip(n @ up0, -1, 1)))
        below = np.mean(C @ n + d > 0)
        if tilt < max_tilt_deg and below > 0.9:
            # among horizontal planes, the floor is the lowest well-supported one
            if best is None or (len(inl) > 0.3 * best[2] and np.median(C @ n + d) > np.median(C @ best[0] + best[1])):
                best = (n, d, len(inl))
        rest = rest.select_by_index(inl, invert=True)
    if best is None:
        raise RuntimeError("no floor plane found")
    n, d, k = best
    return Floor(n, float(d), k / len(pc.points))
