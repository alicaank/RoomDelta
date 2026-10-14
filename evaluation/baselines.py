"""Baselines on a finished comparison (same reconstruction & alignment for all).

B0  nearest-neighbour cloud difference: a before point is "gone" if no after
    point lies within tau (and vice versa for "new"); clusters -> regions.
B1  visibility-aware geometry only: clusters of free-space conflicts, no
    object association (a moved object becomes one removal + one addition).
Full RoomDelta events.

Reports claimed change regions/events and how many of B0's "gone" points the
visibility test explains as occluded or unobserved (false removals avoided).
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from roomdelta.geometry import VisitMap


def clusters(P, radius, min_pts):
    if len(P) < min_pts:
        return 0, np.full(len(P), -1)
    lab = DBSCAN(eps=radius, min_samples=8).fit_predict(P)
    ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
    good = set(ids[cnt >= min_pts])
    return len(good), np.array([l if l in good else -1 for l in lab])


def main(out):
    out = Path(out)
    va, vb = np.load(out / "visibility_A.npz"), np.load(out / "visibility_B.npz")
    d = pickle.load(open(out / "objects.pkl", "rb"))
    tau = d["tau"]
    sc = float(np.median(VisitMap.load(out / "before_map.npz").depth))
    r, m = 0.03 * sc, 30
    # B0: nearest-neighbour difference
    gone = ~np.isfinite(cKDTree(vb["P"]).query(va["P"], distance_upper_bound=tau)[0])
    new = ~np.isfinite(cKDTree(va["P"]).query(vb["P"], distance_upper_bound=tau)[0])
    n_gone, lab_gone = clusters(va["P"][gone], r, m)
    n_new, _ = clusters(vb["P"][new], r, m)
    st = va["status"][gone]
    expl = {"occluded": float((st == 2).mean()), "not_observed": float((st == 3).mean()),
            "compatible_elsewhere_in_view": float((st == 0).mean()), "free_space_conflict": float((st == 1).mean())}
    # in clustered B0 "removed" regions, how many are mostly occlusion / no-view (i.e. false removal claims)?
    false_regions = 0
    for c in np.unique(lab_gone[lab_gone >= 0]):
        s = st[lab_gone == c]
        if np.mean((s == 2) | (s == 3)) > 0.5:
            false_regions += 1
    # B1: visibility-only regions
    n1a = len(np.unique(va["cluster"][va["cluster"] >= 0]))
    n1b = len(np.unique(vb["cluster"][vb["cluster"] >= 0]))
    ev = json.load(open(out / "events.json"))
    kinds = {}
    for e in ev:
        kinds[e["event_type"]] = kinds.get(e["event_type"], 0) + 1
    res = dict(
        B0_nearest_neighbour=dict(removed_regions=n_gone, added_regions=n_new, gone_points=int(gone.sum()),
                                  gone_points_explained_by_visibility=expl,
                                  removed_regions_mostly_occluded_or_unobserved=false_regions),
        B1_visibility_only=dict(removed_regions=n1a, added_regions=n1b),
        full=kinds)
    json.dump(res, open(out / "baselines.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/room")
