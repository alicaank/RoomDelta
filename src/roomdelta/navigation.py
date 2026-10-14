"""Navigation-impact overlay on the floor plane.

Per visit, a 2D grid with three states:
  occupied       surface points in the robot's height band
  observed_free  floor surface was directly observed (line of sight over it)
  unassessed     neither
The default "previous route" is the walked camera path of the before
recording (it was physically traversed). A route can also be planned with A*
on the before map. Impact = cells of the route's swept footprint that are
occupied after but were observed free before. Cells the after recording did
not observe are reported as unassessed, never as blocked.
This is a footprint check, not a whole-body collision model.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import cv2
import numpy as np

from .colors import AFTER, BEFORE
from .geometry import VisitMap
from .scene import Floor

OCC, FREE, UNK = 2, 1, 0


@dataclass
class Grid:
    origin: np.ndarray    # (2,) floor-plane coords of cell (0,0)
    res: float            # cell size, scene units
    before: np.ndarray    # HxW uint8 states
    after: np.ndarray
    new_obstacle: np.ndarray | None = None   # after surface in space observed free before (visibility-test evidence)
    vacated: np.ndarray | None = None        # before surface the after recording sees through

    def to_cell(self, xy):
        return np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)

    def to_xy(self, ij):
        return self.origin + (np.asarray(ij) + 0.5) * self.res


def _state_grid(m: VisitMap, floor: Floor, origin, shape, res, band, floor_tol, stride=2):
    occ = np.zeros(shape, np.int32)
    free = np.zeros(shape, np.int32)
    for i in range(m.n):
        P, _ = m.backproject(i, stride=stride)
        h = floor.height(P)
        ij = np.floor((floor.to_2d(P) - origin) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[1])
        ij, h = ij[ok], h[ok]
        o = (h > band[0]) & (h < band[1])
        f = np.abs(h) < floor_tol
        np.add.at(occ, (ij[o, 0], ij[o, 1]), 1)
        np.add.at(free, (ij[f, 0], ij[f, 1]), 1)
    g = np.full(shape, UNK, np.uint8)
    g[free >= 2] = FREE
    g[occ >= 3] = OCC        # an obstacle observation overrides floor glimpses under it
    return g


def build_grid(A: VisitMap, B: VisitMap, floor: Floor, meters_per_unit: float | None, res_m: float = 0.04,
               band_m=(0.06, 1.2), floor_tol_m: float = 0.03) -> Grid:
    mpu = meters_per_unit or 1.0
    res = res_m / mpu
    band = (band_m[0] / mpu, band_m[1] / mpu)
    xy = np.concatenate([floor.to_2d(m.backproject(i, stride=8)[0]) for m in (A, B) for i in range(m.n)])
    lo, hi = np.percentile(xy, [1, 99], axis=0)
    cams = np.concatenate([floor.to_2d(A.centers()), floor.to_2d(B.centers())])
    lo, hi = np.minimum(lo, cams.min(0)), np.maximum(hi, cams.max(0))
    lo, hi = lo - 10 * res, hi + 10 * res
    shape = tuple(np.ceil((hi - lo) / res).astype(int))
    ga = _state_grid(A, floor, lo, shape, res, band, floor_tol_m / mpu)
    gb = _state_grid(B, floor, lo, shape, res, band, floor_tol_m / mpu)
    grid = Grid(lo, res, ga, gb)
    grid.band = band
    return grid


def add_change_evidence(grid: Grid, floor: Floor, P_conflict_after: np.ndarray, P_conflict_before: np.ndarray, min_pts: int = 2):
    """Mark cells from free-space-conflict points (not from differencing two noisy grids)."""
    def cells(P):
        out = np.zeros(grid.before.shape, np.int32)
        if len(P) == 0:
            return out.astype(bool)
        h = floor.height(P)
        P = P[(h > grid.band[0]) & (h < grid.band[1])]
        ij = grid.to_cell(floor.to_2d(P))
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < out.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < out.shape[1])
        np.add.at(out, (ij[ok, 0], ij[ok, 1]), 1)
        return out >= min_pts
    grid.new_obstacle = cells(P_conflict_after)
    grid.vacated = cells(P_conflict_before)
    return grid


def walked_route(A: VisitMap, floor: Floor, grid: Grid) -> np.ndarray:
    """Before camera path projected to the floor, as a polyline of xy points."""
    order = np.argsort(A.timestamps)
    return floor.to_2d(A.centers()[order])


def astar(passable: np.ndarray, start, goal, cost: np.ndarray | None = None):
    """8-connected A*; ``cost`` multiplies step cost per cell (e.g. unassessed floor)."""
    H, W = passable.shape
    start, goal = tuple(int(x) for x in start), tuple(int(x) for x in goal)
    if not (0 <= start[0] < H and 0 <= start[1] < W and 0 <= goal[0] < H and 0 <= goal[1] < W):
        return None
    if not (passable[start] and passable[goal]):
        return None
    nb = [(-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
    g = {start: 0.0}
    prev = {}
    pq = [(0.0, start)]
    while pq:
        _, u = heapq.heappop(pq)
        if u == goal:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return np.array(path[::-1])
        for di, dj, c in nb:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < H and 0 <= v[1] < W) or not passable[v]:
                continue
            ng = g[u] + c * (1.0 if cost is None else cost[v])
            if ng < g.get(v, np.inf):
                g[v] = ng
                prev[v] = u
                heapq.heappush(pq, (ng + np.hypot(goal[0] - v[0], goal[1] - v[1]), v))
    return None


def inflate(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius_cells + 1,) * 2)
    return cv2.dilate(occ.astype(np.uint8), k).astype(bool)


def swept(grid: Grid, route_xy: np.ndarray, radius_cells: int) -> np.ndarray:
    mask = np.zeros(grid.before.shape, np.uint8)
    pts = grid.to_cell(route_xy)
    for p, q in zip(pts[:-1], pts[1:]):
        cv2.line(mask, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1, thickness=2 * radius_cells + 1)
    return mask.astype(bool)


def route_impact(grid: Grid, route_xy: np.ndarray, robot_radius_m: float, clearance_m: float, meters_per_unit: float | None,
                 events=None, floor: Floor | None = None) -> dict:
    mpu = meters_per_unit or 1.0
    r_cells = max(1, int(round((robot_radius_m + clearance_m) / mpu / grid.res)))
    sw = swept(grid, route_xy, r_cells)
    new_occ = grid.new_obstacle if grid.new_obstacle is not None else (grid.after == OCC) & (grid.before == FREE)
    hits = sw & new_occ
    unassessed = sw & (grid.after == UNK)
    res = dict(robot_radius_m=robot_radius_m, clearance_m=clearance_m, swept_cells=int(sw.sum()),
               new_obstacle_cells_on_route=int(hits.sum()), unassessed_cells_on_route=int(unassessed.sum()),
               unassessed_fraction=float(unassessed.sum() / max(1, sw.sum())), blocking_events=[])
    if events is not None and floor is not None and hits.any():
        hit_xy = grid.to_xy(np.argwhere(hits))
        for e in events:
            if e.event_type not in ("moved", "added_to_observed_area") or e.after_location is None:
                continue
            c = floor.to_2d(np.asarray(e.after_location)[None])[0]
            d = np.linalg.norm(hit_xy - c, axis=1)
            near = int((d < 0.35 / mpu).sum())
            if near > 0:
                res["blocking_events"].append(dict(event_id=e.event_id, label=e.semantic_label, cells=near))
    # can the robot still get between the route's endpoints on the after map? Endpoints are the
    # first/last route vertices passable for this footprint on BOTH maps: the handheld camera can
    # hover over furniture edges, and a place one recording never observed is not evidence of
    # either free space or a blockage.
    pas_b = ~inflate(grid.before == OCC, r_cells)
    pas_a = ~inflate(grid.after == OCC, r_cells)
    cells = grid.to_cell(route_xy)
    okv = [k for k, c in enumerate(cells) if 0 <= c[0] < pas_b.shape[0] and 0 <= c[1] < pas_b.shape[1]
           and pas_b[c[0], c[1]] and pas_a[c[0], c[1]]]
    if len(okv) < 2:
        res.update(replan_before_len_m=None, replan_after_len_m=None, replan_after_unassessed_fraction=None,
                   replan_after_path_xy=None, hit_cells_xy=grid.to_xy(np.argwhere(hits)).tolist(),
                   replan_note="route endpoints are not passable for this footprint even before")
        return res
    s_xy, g_xy = route_xy[okv[0]], route_xy[okv[-1]]
    pb, ub = plan_cells(grid, grid.before, s_xy, g_xy, r_cells)
    pa, ua = plan_cells(grid, grid.after, s_xy, g_xy, r_cells)
    res["replan_before_unassessed_fraction"] = ub
    res["replan_before_len_m"] = None if pb is None else float(_plen(pb) * grid.res * mpu)
    res["replan_after_len_m"] = None if pa is None else float(_plen(pa) * grid.res * mpu)
    res["replan_after_unassessed_fraction"] = ua
    res["replan_after_path_xy"] = None if pa is None else grid.to_xy(pa).tolist()
    res["hit_cells_xy"] = grid.to_xy(np.argwhere(hits)).tolist()
    return res


def plan_cells(grid: Grid, state: np.ndarray, start_xy, goal_xy, r_cells: int, unassessed_cost: float = 4.0):
    """A* on one visit's map: occupied (inflated) is impassable; unassessed floor is allowed
    but expensive, and the fraction of the path crossing it is returned."""
    occ = state == OCC
    passable = ~inflate(occ, r_cells)
    cost = np.where(state == FREE, 1.0, unassessed_cost)

    def snap(p, max_cells=3):
        idx = np.argwhere(passable)
        if len(idx) == 0:
            return None
        d = np.linalg.norm(idx - p, axis=1)
        k = np.argmin(d)
        return idx[k] if d[k] <= max_cells else None

    s, g = snap(grid.to_cell(start_xy)), snap(grid.to_cell(goal_xy))
    if s is None or g is None:
        return None, None
    p = astar(passable, s, g, cost)
    if p is None:
        return None, None
    return p, float(np.mean(state[p[:, 0], p[:, 1]] != FREE))


def _plen(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def plan(grid: Grid, start_xy, goal_xy, robot_radius_m, clearance_m, meters_per_unit, which="before"):
    mpu = meters_per_unit or 1.0
    r_cells = max(1, int(round((robot_radius_m + clearance_m) / mpu / grid.res)))
    p, unassessed = plan_cells(grid, grid.before if which == "before" else grid.after, start_xy, goal_xy, r_cells)
    return (None, None) if p is None else (grid.to_xy(p), unassessed)


def grid_image(grid: Grid, which: str = "after") -> np.ndarray:
    """RGB visualisation: free (light), occupied (dark), unassessed (grey), new obstacles (orange)."""
    g = grid.after if which == "after" else grid.before
    img = np.full(g.shape + (3,), 150, np.uint8)
    img[g == FREE] = (225, 225, 225)
    img[g == OCC] = (60, 60, 60)
    new = grid.new_obstacle if grid.new_obstacle is not None else (grid.after == OCC) & (grid.before == FREE)
    gone = grid.vacated if grid.vacated is not None else (grid.before == OCC) & (grid.after == FREE)
    img[gone] = BEFORE    # vacated
    img[new] = AFTER      # new obstacle
    return img
