"""End-to-end `compare`: two videos -> shared 3D representation -> change events.

Stages (each cached in the output directory, heavy models run one at a time):
  1. keyframes            capture.select_keyframes
  2. geometry             DA3: joint camera poses for both visits, per-visit depth
  3. alignment check      ALIKED+LightGlue + robust Sim(3) on the static background
  4. visibility           per-surface-point evidence (used for evidence display and navigation)
  5. objects              open-vocabulary detection + SAM 2.1 masks, lifted to 3D, video-tracked
  6. decision             generative rendered-hypothesis MAP (inference.py, decision.py)
  7. outputs              events.json, report.md, evidence images, PLY exports, navigation impact
"""
from __future__ import annotations

import json
import pickle
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import visibility as V
from .alignment import align_visits, cross_visit_correspondences
from .capture import select_keyframes
from .events import ScaleInfo, build_events
from .geometry import VisitMap, get_backend
from .objects import Segmenter, build_objects, evidence_fractions, geometric_proposals, merge_duplicates, semantic_proposals
from .scale import metric_scale
from .scene import estimate_floor


@dataclass
class Config:
    geometry_backend: str = "da3-base"
    geometry_mode: str = "shared_pose"    # "shared_pose" (joint cameras, per-visit depth) | "independent"
    track_objects: bool = True            # SAM 2.1 video tracking of every object through its visit
    max_keyframes: int = 32
    long_side: int = 504
    view_group_pos_frac: float = 0.15     # view-group radius, fraction of scene scale
    view_group_angle_deg: float = 20.0
    surface_voxel_frac: float = 0.008
    object_voxel_frac: float = 0.02
    eps_sigma: float = 3.0                # display/navigation tolerance = eps_sigma * within-visit robust sigma
    min_conflict_groups: int = 2
    cluster_radius_frac: float = 0.03
    min_motion_frac: float = 0.06         # smallest displacement considered a move, fraction of scene scale
    max_object_travel_frac: float = 1.2
    metric_model: str | None = "depth-anything/DA3METRIC-LARGE"   # learned metric depth for scale (None: skip)
    camera_height_prior_m: float | None = 1.40   # handheld phone height prior: fallback scale and cross-check
    meters_per_unit: float | None = None          # calibrated override (e.g. from a measured distance)
    robot_radius_m: float = 0.22                  # navigation footprint: humanoid shoulder width ~0.45 m (disc)
    clearance_m: float = 0.05
    seed: int = 0
    phase_before: int = 0                 # candidate sampling phase (self-tests use disjoint frames of one video)
    phase_after: int = 0


def _log(msg, t0):
    print(f"[roomdelta {time.time() - t0:7.1f}s] {msg}", flush=True)


def _keyframes(cfg: Config, path, tag):
    return select_keyframes(path, max_keyframes=cfg.max_keyframes, long_side=cfg.long_side,
                            phase=cfg.phase_before if tag == "before" else cfg.phase_after)


def _geometry(cfg: Config, before, after, out: Path, force: bool, t0):
    fa, fb = out / "before_map.npz", out / "after_map.npz"
    if fa.exists() and fb.exists() and not force:
        return VisitMap.load(fa), VisitMap.load(fb)
    caps = {t: _keyframes(cfg, p, t) for t, p in (("before", before), ("after", after))}
    for t, c in caps.items():
        _log(f"{t}: {c.report['num_keyframes']} keyframes from {c.report['num_candidates']} candidates "
             f"({c.report['num_blurry_rejected']} blurry)", t0)
    backend = get_backend(cfg.geometry_backend)
    if cfg.geometry_mode == "shared_pose":
        from .geometry import reconstruct_shared_pose

        A, B = reconstruct_shared_pose(backend, caps["before"], caps["after"])
    else:
        A, B = backend.reconstruct(caps["before"], "before"), backend.reconstruct(caps["after"], "after")
    A.save(fa); B.save(fb)
    _log(f"geometry: {cfg.geometry_backend}, {cfg.geometry_mode}, peak {A.meta['peak_gpu_gb']} GB", t0)
    return A, B


def _scale(cfg: Config, A, B, floor, out: Path, force: bool, t0) -> ScaleInfo:
    h_cam = float(np.median(floor.height(A.centers())))
    metric = None
    if cfg.metric_model and not cfg.meters_per_unit:
        mf = out / "metric_scale.json"
        if mf.exists() and not force and json.load(open(mf)).get("model_id") == cfg.metric_model:
            metric = json.load(open(mf))
        else:
            try:
                metric = metric_scale([A, B], cfg.metric_model)
                json.dump(metric, open(mf, "w"), indent=1)
            except Exception as e:  # noqa: BLE001 - fall back to the height prior
                _log(f"metric scale failed ({e.__class__.__name__}: {e}); falling back", t0)
    if cfg.meters_per_unit:
        scale = ScaleInfo(cfg.meters_per_unit, "calibrated", "user-supplied measurement")
    elif metric is not None:
        scale = ScaleInfo(metric["meters_per_unit"], "learned_metric",
                          f"{metric['model_id'].split('/')[-1]} monocular metric depth vs reconstruction, median of "
                          f"{metric['n_frames']} keyframes (16-84%: {metric['p16']:.3f}-{metric['p84']:.3f} m/unit)",
                          rel_uncertainty=metric["rel_spread"])
    elif cfg.camera_height_prior_m:
        scale = ScaleInfo(cfg.camera_height_prior_m / h_cam, "approximate",
                          f"handheld camera height prior {cfg.camera_height_prior_m} m (measured {h_cam:.3f} units above floor)",
                          rel_uncertainty=0.12)
    else:
        scale = ScaleInfo(None, "scene_units", "RGB-only reconstruction")
    if scale.meters_per_unit:
        scale.cross_check = dict(implied_camera_height_m=round(h_cam * scale.meters_per_unit, 3),
                                 height_prior_m=cfg.camera_height_prior_m)
    _log(f"scale: {scale.status} {scale.meters_per_unit and round(scale.meters_per_unit, 4)} m/unit; {scale.cross_check}", t0)
    return scale


def compare(before: str, after: str, out: str | Path, cfg: Config = Config(), force: bool = False) -> dict:
    t0 = time.time()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}

    def stage(name):
        timings[name] = round(time.time() - t0, 1)

    # 1-2 ---------------------------------------------------------------- geometry
    A, B_own = _geometry(cfg, before, after, out, force, t0)
    stage("geometry")

    # 3 ------------------------------------------------------------------ alignment check / refinement
    from .matching import Matcher

    matcher = Matcher()
    fa, fb = matcher.extract(A.images), matcher.extract(B_own.images)
    cf = out / "correspondences.pkl"
    if cf.exists() and not force:
        corr = pickle.load(open(cf, "rb"))
    else:
        corr = cross_visit_correspondences(A, B_own, matcher, fa, fb)
        pickle.dump(corr, open(cf, "wb"))
    al = align_visits(A, B_own, corr, seed=cfg.seed)
    B = B_own.transformed(al.s, al.R, al.t)
    sc = al.scene_scale
    json.dump(al.as_dict(), open(out / "alignment.json", "w"), indent=1, default=float)
    _log(f"alignment: s={al.s:.3f}, {al.stats['n_inliers']} inliers / {al.stats['n_inlier_pairs']} frame pairs, "
         f"median residual {al.stats['median_inlier_residual'] / sc:.2%} of scene scale, ok={al.ok} {al.stats['failure_reasons']}", t0)
    stage("alignment")

    # 4 ------------------------------------------------------------------ per-point visibility evidence
    vis, groups = {}, {}
    for tag, S, T in (("A", A, B), ("B", B, A)):
        gS = V.view_groups(S, cfg.view_group_pos_frac * sc, cfg.view_group_angle_deg)
        gT = V.view_groups(T, cfg.view_group_pos_frac * sc, cfg.view_group_angle_deg)
        pts = V.surface_points(S, cfg.surface_voxel_frac * sc)
        mad, _ = V.calibrate_eps(S, pts, gS, seed=cfg.seed)
        eps_rel, eps_abs = cfg.eps_sigma * mad, 2 * al.sigma_p95
        sup, thru = V.self_support(S, pts, gS, eps_rel, eps_abs)
        keep = (sup >= 1) & (thru == 0)   # surfaces confirmed by another own-visit view group, no floaters
        pts = V.SurfacePoints(pts.P[keep], pts.rgb[keep], pts.frame[keep], pts.uv[keep], sup[keep])
        vis[tag] = V.visibility_test(S, T, pts, gT, eps_rel, eps_abs, cfg.min_conflict_groups)
        groups[tag] = gS
    clusters = {k: V.change_clusters(vis[k], cfg.cluster_radius_frac * sc) for k in "AB"}
    stage("visibility")

    # 5 ------------------------------------------------------------------ objects
    pf = out / "proposals.pkl"
    if pf.exists() and not force:
        props = pickle.load(open(pf, "rb"))
    else:
        seg = Segmenter()
        props = {}
        for tag, M in (("A", A), ("B", B)):
            props[tag] = (semantic_proposals(M, seg),
                          geometric_proposals(M, seg, vis[tag].pts.frame, vis[tag].pts.uv, clusters[tag]))
        seg.free()
        del seg
        pickle.dump(props, open(pf, "wb"))
    voxel = cfg.object_voxel_frac * sc
    objs = {tag: merge_duplicates(build_objects(M, [*props[tag][0], *props[tag][1]], voxel, prefix=tag), voxel, M)
            for tag, M in (("A", A), ("B", B))}
    if cfg.track_objects:
        tf = out / "objects_tracked.pkl"
        if tf.exists() and not force:
            objs, track_stats = pickle.load(open(tf, "rb"))
        else:
            from .tracking import VideoTracker, track_objects

            tracker = VideoTracker()
            track_stats = {}
            for tag, M in (("A", A), ("B", B)):
                objs[tag], track_stats[tag] = track_objects(M, objs[tag], tracker, voxel)
            tracker.free()
            del tracker
            pickle.dump((objs, track_stats), open(tf, "wb"))
        n_merged = sum(len(v.get("merged", [])) for t in track_stats.values() for v in t.values())
        _log(f"tracking: {n_merged} duplicate objects merged by shared tracks", t0)
    _log(f"objects: before {len(objs['A'])}, after {len(objs['B'])}", t0)
    stage("objects")

    # 6 ------------------------------------------------------------------ decision
    from .appearance import DenseFeatures, appearance_model, object_embeddings
    from .association import resolve_other_side
    from .decision import DecisionConfig, decide, self_consistency, to_association
    from .inference import SceneRenderer, calibrate_cross_sensor, calibrate_sensor, label_surface_points

    floor = estimate_floor(A, sc)
    tau = max(1.5 * voxel, 0.5 * vis["A"].eps_rel * float(np.median(A.depth[A.valid])))
    scale = _scale(cfg, A, B, floor, out, force, t0)
    dcfg = DecisionConfig()

    # self-consistency first: objects their own visit contradicts are reconstruction ghosts
    ghosts = {}
    for tag, M in (("A", A), ("B", B)):
        P = vis[tag].pts.P
        own_sensor = calibrate_sensor(M, P, vis[tag].pts.frame, vis[tag].pts.uv, groups[tag], 0.0, seed=cfg.seed)
        own = SceneRenderer(P, label_surface_points(P, objs[tag], 1.5 * voxel), len(objs[tag]), M, own_sensor, groups[tag],
                            cfg.surface_voxel_frac * sc)
        ghosts[tag] = {k: v for k, v in self_consistency(objs[tag], own, groups[tag], dcfg).items() if v["ghost"]}
        objs[tag] = [o for o in objs[tag] if o.object_id not in ghosts[tag]]
    if any(ghosts.values()):
        _log(f"ignored reconstruction ghosts: {ghosts}", t0)

    # sensors[X]: noise of comparing the OTHER visit's surfaces with X's depth maps (fitted on this pair)
    sensors = {tag: calibrate_cross_sensor(M, vis[src].pts.P, 2 * al.sigma_p95, seed=cfg.seed)
               for tag, M, src in (("A", A, "B"), ("B", B, "A"))}
    gT = {"A": V.view_groups(B, cfg.view_group_pos_frac * sc, cfg.view_group_angle_deg),
          "B": V.view_groups(A, cfg.view_group_pos_frac * sc, cfg.view_group_angle_deg)}
    rend = {tag: SceneRenderer(vis[tag].pts.P, label_surface_points(vis[tag].pts.P, objs[tag], 1.5 * voxel), len(objs[tag]),
                               T, sensors[other], gT[tag], cfg.surface_voxel_frac * sc)
            for tag, T, other in (("A", B, "B"), ("B", A, "A"))}

    # object-level appearance identity, self-calibrated on this scene
    dfe = DenseFeatures(upscale=1.0)
    emb = {"A": object_embeddings(objs["A"], dfe(A.images), A.hw), "B": object_embeddings(objs["B"], dfe(B.images), B.hw)}
    dfe.free()
    app_llr, app_stats = appearance_model(emb["A"], emb["B"], objs["A"], objs["B"], tau)

    dec = decide(objs["A"], objs["B"], rend["A"], rend["B"], floor, tau, scale.meters_per_unit or 1.0,
                 cfg.max_object_travel_frac * sc, cfg.min_motion_frac * sc, A, B, fa, fb, matcher, dcfg,
                 emb["A"], emb["B"], app_llr)
    matcher.free()
    asc = to_association(dec, dcfg.odds_min)
    ev = {tag: {o.object_id: evidence_fractions(o, vis[tag], 0.75 * tau) for o in objs[tag]} for tag in "AB"}
    asc = resolve_other_side(asc, objs["A"], objs["B"], ev["A"], ev["B"], A, B, floor,
                             vis["A"].eps_rel, vis["A"].eps_abs, radius=1.5 * voxel)
    json.dump(dict(appearance=app_stats, ghosts=ghosts,
                   sensors={k: dict(a=v.a, b=v.b, outlier=v.outlier, **v.meta) for k, v in sensors.items()},
                   matches=[{k: v for k, v in m.items() if k not in ("R", "t")} for m in dec.matches],
                   singles_a=dec.singles_a, singles_b=dec.singles_b, config=dcfg.__dict__),
              open(out / "decision.json", "w"), indent=1, default=float)
    events = build_events(objs["A"], objs["B"], asc, ev["A"], ev["B"], vis["A"], vis["B"], A, B, al, floor, scale, tau,
                          cfg.min_motion_frac * sc, dcfg.odds_min)
    stage("decision")
    _log("events: " + ", ".join(f"{t}={sum(e.event_type == t for e in events)}" for t in sorted({e.event_type for e in events})), t0)

    # 7 ------------------------------------------------------------------ outputs
    B.save(out / "after_aligned_map.npz")
    for tag in "AB":
        r = vis[tag]
        np.savez_compressed(out / f"visibility_{tag}.npz", P=r.pts.P, rgb=r.pts.rgb, frame=r.pts.frame, uv=r.pts.uv,
                            status=r.status, counts=r.counts, cluster=clusters[tag], eps_rel=r.eps_rel, eps_abs=r.eps_abs)
    pickle.dump(dict(objects=objs, evidence=ev, association=asc, floor=floor, tau=tau, scale=scale),
                open(out / "objects.pkl", "wb"))
    json.dump([e.as_dict() for e in events], open(out / "events.json", "w"), indent=1, default=float)

    from .export import export_scene

    export_scene(out, A, B, objs, events, floor, scale)

    from . import navigation as N

    grid = N.build_grid(A, B, floor, scale.meters_per_unit)
    N.add_change_evidence(grid, floor, vis["B"].pts.P[vis["B"].status == V.CONFLICT], vis["A"].pts.P[vis["A"].status == V.CONFLICT])
    route = N.walked_route(A, floor, grid)
    nav = N.route_impact(grid, route, cfg.robot_radius_m, cfg.clearance_m, scale.meters_per_unit, events, floor)
    nav["radius_sensitivity"] = []
    for r in (0.15, 0.20, 0.25, 0.30, 0.35):
        x = N.route_impact(grid, route, r, cfg.clearance_m, scale.meters_per_unit, events, floor)
        nav["radius_sensitivity"].append(dict(robot_radius_m=r, new_obstacle_cells=x["new_obstacle_cells_on_route"],
                                              intersecting=[b["label"] + " (" + b["event_id"] + ")" for b in x["blocking_events"]]))
    json.dump({k: v for k, v in nav.items() if k not in ("hit_cells_xy", "replan_after_path_xy")},
              open(out / "navigation.json", "w"), indent=1)

    from .report import write_report

    write_report(out, events, al, scale, vis, A, B, cfg, nav)
    stage("outputs")
    run = dict(config=asdict(cfg), before=Path(before).name, after=Path(after).name, timings_s=timings,
               total_s=round(time.time() - t0, 1), versions=_versions(), platform=platform.platform(),
               models=dict(geometry=A.meta.get("model_id"), metric=cfg.metric_model, detector="IDEA-Research/grounding-dino-base",
                           segmenter="facebook/sam2.1-hiera-large", features="ALIKED (n16) + LightGlue",
                           appearance="facebook/dinov2-base"),
               floor=dict(normal=floor.normal.tolist(), d=floor.d))
    json.dump(run, open(out / "run.json", "w"), indent=1, default=str)
    _log(f"done -> {out}", t0)
    return run


def _versions():
    import importlib.metadata as md

    v = {}
    for p in ("torch", "numpy", "open3d", "transformers", "viser", "lightglue", "depth-anything-3", "opencv-python-headless", "scipy"):
        try:
            v[p] = md.version(p)
        except Exception:
            v[p] = None
    return v
