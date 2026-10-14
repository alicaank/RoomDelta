"""Interactive evidence viewer (Viser). Reads only cached outputs; no models needed.

* every change event is clickable (outline box in 3D, or the event list) and shows
  before/after masks, supporting keyframes, displacement and the witness rays:
  after-cameras that look *through* the old surface to what is there now;
* "uncertain only" shows what the system declined to conclude;
* navigation overlay: walked route of the before recording, new obstacles backed
  by free-space evidence, footprint sliders and click-to-plan start/goal.

Viewer world: floor plane z = 0, z up, metres when a metric scale is available.
Colours: before = orange, after = blue, witness rays = green (see colors.py).
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from . import navigation as N
from .geometry import VisitMap

from .colors import AFTER, BEFORE, INK, OCCLUDED as WITNESS
# compatible, seen-through, occluded, not observed, single-view conflict
STATUS_COL = np.array([[170, 170, 165], [235, 104, 52], [27, 175, 122], [220, 219, 214], [240, 180, 150]], np.uint8)


class Scene:
    """Shared world -> viewer world (floor basis, metres)."""

    def __init__(self, floor, mpu: float):
        self.floor, self.mpu = floor, mpu
        self.e1, self.e2, self.up = floor.basis()

    def __call__(self, P):
        P = np.atleast_2d(P)
        return (np.stack([P @ self.e1, P @ self.e2, self.floor.height(P)], 1) * self.mpu).astype(np.float32)

    def rot(self, R):
        return np.stack([self.e1, self.e2, self.up]) @ R


def _cloud(m: VisitMap, stride=3, max_pts=220_000, seed=0):
    P, C = [], []
    for i in range(m.n):
        p, uv = m.backproject(i, stride=stride)
        P.append(p); C.append(m.images[i][uv[:, 1].astype(int), uv[:, 0].astype(int)])
    P, C = np.concatenate(P), np.concatenate(C)
    if len(P) > max_pts:
        idx = np.random.default_rng(seed).choice(len(P), max_pts, replace=False)
        P, C = P[idx], C[idx]
    return P, C


def _quat_from_R(R):
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _box_edges(lo, hi):
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    e = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    return np.stack([np.stack([c[a], c[b]]) for a, b in e]).astype(np.float32)


def _arrow_mesh(a, b, radius):
    """Shaft + cone from a to b, as one trimesh."""
    import trimesh

    v = b - a
    L = float(np.linalg.norm(v))
    head = min(0.35 * L, 6 * radius)
    shaft = trimesh.creation.cylinder(radius=radius, height=max(L - head, 1e-4), sections=16)
    shaft.apply_translation([0, 0, (L - head) / 2])
    cone = trimesh.creation.cone(radius=2.6 * radius, height=head, sections=20)
    cone.apply_translation([0, 0, L - head])
    m = trimesh.util.concatenate([shaft, cone])
    z = v / max(L, 1e-9)
    x = np.cross(z, [0, 0, 1.0]) if abs(z[2]) < 0.99 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], 1)
    T[:3, 3] = a
    m.apply_transform(T)
    m.visual.face_colors = np.array([*INK, 255], np.uint8)
    return m


def serve(out: str | Path, host: str = "0.0.0.0", port: int = 8080, block: bool = True):
    import viser

    out = Path(out)
    A = VisitMap.load(out / "before_map.npz")
    B = VisitMap.load(out / "after_aligned_map.npz")
    d = pickle.load(open(out / "objects.pkl", "rb"))
    floor, scale, tau = d["floor"], d["scale"], d["tau"]
    objs = {o.object_id: o for tag in "AB" for o in d["objects"][tag]}
    events = json.load(open(out / "events.json"))
    vis = {k: np.load(out / f"visibility_{k}.npz") for k in "AB"}
    nav_cfg = json.load(open(out / "navigation.json")) if (out / "navigation.json").exists() else {}
    mpu = scale.meters_per_unit or 1.0
    S = Scene(floor, mpu)
    unit = "m" if scale.meters_per_unit else "units"

    server = viser.ViserServer(host=host, port=port, label="RoomDelta")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(control_width="large", brand_color=AFTER)

    # ---------------------------------------------------------------- base layers
    Pa, Ca = _cloud(A)
    Pb, Cb = _cloud(B, seed=1)
    Qa, Qb = S(Pa), S(Pb)
    ps = 0.006                                   # metres
    dim = lambda C: (0.35 * C.mean(1, keepdims=True) + 0.65 * 235).astype(np.uint8).repeat(3, 1)
    h_before = server.scene.add_point_cloud("/scene/before", Qa, Ca, point_size=ps, point_shape="rounded")
    h_after = server.scene.add_point_cloud("/scene/after", Qb, Cb, point_size=ps, point_shape="rounded", visible=False)
    h_stat_a = server.scene.add_point_cloud("/evidence/before_status", S(vis["A"]["P"]), STATUS_COL[vis["A"]["status"]],
                                            point_size=ps, visible=False)
    h_stat_b = server.scene.add_point_cloud("/evidence/after_status", S(vis["B"]["P"]), STATUS_COL[vis["B"]["status"]],
                                            point_size=ps, visible=False)
    lo_xy, hi_xy = np.percentile(Qa[:, :2], [1, 99], axis=0)
    gc = (lo_xy + hi_xy) / 2
    gsz = float(np.max(hi_xy - lo_xy) * 1.15)
    h_grid_floor = server.scene.add_grid("/floor_grid", gsz, gsz, plane="xy", cell_size=0.25 if unit == "m" else 0.1,
                                         section_size=1.0, cell_color=(225, 225, 220), section_color=(170, 170, 165),
                                         position=(float(gc[0]), float(gc[1]), -0.003))

    def frustums(m: VisitMap, name, color):
        hs = []
        H, W = m.hw
        for i in range(m.n):
            c2w = np.linalg.inv(m.w2c[i])
            fov = 2 * np.arctan2(H / 2, m.K[i][1, 1])
            hs.append(server.scene.add_camera_frustum(f"/cams/{name}/{i:02d}", fov=fov, aspect=W / H, scale=0.06,
                                                      line_width=1.5, color=color, wxyz=_quat_from_R(S.rot(c2w[:3, :3])),
                                                      position=S(c2w[:3, 3])[0], visible=False))
        return hs

    cams = {"A": frustums(A, "before", BEFORE), "B": frustums(B, "after", AFTER)}

    with server.gui.add_folder("Layers"):
        cb_before = server.gui.add_checkbox("Before point cloud", True)
        cb_after = server.gui.add_checkbox("After point cloud (aligned)", False)
        cb_dim = server.gui.add_checkbox("Dim the scene (highlight changes)", False)
        cb_status = server.gui.add_dropdown("Visibility evidence", ["off", "before surfaces vs after", "after surfaces vs before"])
        server.gui.add_markdown("<small><b style='color:#eb6834'>■ seen through</b> (place now empty) · "
                                "<b style='color:#1baf7a'>■ occluded</b> (no conclusion) · ■ grey compatible · "
                                "□ light not observed</small>")
        cb_cams = server.gui.add_checkbox("All camera frustums", False)
        cb_floor = server.gui.add_checkbox(f"Floor grid (1 {unit} sections)", True)

    def refresh_scene():
        h_before.visible = cb_before.value and cb_status.value == "off"
        h_after.visible = cb_after.value and cb_status.value == "off"
        h_before.colors = dim(Ca) if cb_dim.value else Ca
        h_after.colors = dim(Cb) if cb_dim.value else Cb
        h_stat_a.visible = cb_status.value == "before surfaces vs after"
        h_stat_b.visible = cb_status.value == "after surfaces vs before"

    for h in (cb_before, cb_after, cb_dim, cb_status):
        h.on_update(lambda _: refresh_scene())

    @cb_cams.on_update
    def _(_):
        for hs in cams.values():
            for h in hs:
                h.visible = cb_cams.value

    @cb_floor.on_update
    def _(_):
        h_grid_floor.visible = cb_floor.value

    # ---------------------------------------------------------------- events
    ev_nodes, ev_focus = {}, {}

    def obj_points(ids):
        return np.concatenate([objs[k].points for k in ids]) if ids else None

    def outline(Q, name, color):
        lo, hi = np.percentile(Q, [5, 95], axis=0)
        lo[2] = max(lo[2], 0.0)
        edges = server.scene.add_line_segments(name + "_edges", _box_edges(lo, hi), color, line_width=2.0)
        hit = server.scene.add_box(name + "_box", color=color, dimensions=np.maximum(hi - lo, 0.02), position=(lo + hi) / 2,
                                   opacity=0.06)
        return [edges, hit]

    for e in events:
        if e["event_type"] == "unchanged":
            continue
        eid = e["event_id"]
        nodes = []
        pa = obj_points(e["before_object_ids"])
        pb = obj_points(e["after_object_ids"])
        if pa is None and pb is None and e["geometric_evidence"].get("points"):
            pts = np.asarray(e["geometric_evidence"]["points"])
            pa, pb = (pts, None) if e["event_type"].startswith("removed") else (None, pts)
        qa = S(pa) if pa is not None else None
        qb = S(pb) if pb is not None else None
        if qa is not None:
            nodes.append(server.scene.add_point_cloud(f"/events/{eid}/before_pts", qa, BEFORE, point_size=ps * 1.3))
            nodes += outline(qa, f"/events/{eid}/before", BEFORE)
        if qb is not None:
            nodes.append(server.scene.add_point_cloud(f"/events/{eid}/after_pts", qb, AFTER, point_size=ps * 1.3))
            nodes += outline(qb, f"/events/{eid}/after", AFTER)
        top = max(np.percentile(q[:, 2], 98) for q in (qa, qb) if q is not None)
        if qa is not None and qb is not None:
            a, b = qa.mean(0), qb.mean(0)
            a[2] = b[2] = top + 0.06
            nodes.append(server.scene.add_mesh_trimesh(f"/events/{eid}/arrow", _arrow_mesh(a, b, 0.008)))
        anchor = (qb if qb is not None else qa).mean(0)
        anchor[2] = top + 0.16
        txt = f"{eid} {e['event_type'].split('_')[0]}: {e['semantic_label']}"
        if e.get("displacement_units") is not None and e["event_type"] == "moved":
            txt += f" · {scale.fmt(e['displacement_units'])}"
        nodes.append(server.scene.add_label(f"/events/{eid}/label", txt, position=anchor, font_screen_scale=0.9))
        rays = e.get("witness_rays") or []
        if rays:
            seg = np.stack([np.stack([S(np.asarray(r[0]))[0], S(np.asarray(r[1]))[0]]) for r in rays])
            nodes.append(server.scene.add_line_segments(f"/events/{eid}/rays", seg, WITNESS, line_width=1.6, visible=False))
        allq = np.concatenate([q for q in (qa, qb) if q is not None])
        lo_q, hi_q = np.percentile(allq, [5, 95], axis=0)
        ev_focus[eid] = ((lo_q + hi_q) / 2, float(np.linalg.norm((hi_q - lo_q)[:2])))
        ev_nodes[eid] = nodes

    by_id = {e["event_id"]: e for e in events}

    def ids_for(filt):
        keep = []
        for e in events:
            t = e["event_type"]
            if (filt == "changes" and t in ("moved", "added_to_observed_area", "removed_from_previous_location")) or \
               (filt == "uncertain only" and t in ("uncertain", "newly_observed")) or \
               (filt == "all non-unchanged" and t != "unchanged"):
                keep.append(e["event_id"])
        return keep

    with server.gui.add_folder("Change events"):
        dd_filter = server.gui.add_dropdown("Show", ["all non-unchanged", "changes", "uncertain only"])
        dd_event = server.gui.add_dropdown("Event", ["(none)"] + ids_for("all non-unchanged"))
        cb_rays = server.gui.add_checkbox("Witness rays for selected", True)
        cb_focus = server.gui.add_checkbox("Fly camera to selected", True)
        cb_overlays = server.gui.add_checkbox("Show event overlays", True)
        md = server.gui.add_markdown("Select an event or click a box in the scene.")
        img_handle = server.gui.add_image(np.full((8, 8, 3), 252, np.uint8), label="evidence")

    def describe(e):
        lines = [f"**{e['event_id']} · {e['event_type'].replace('_', ' ')} · {e['semantic_label']}**", "", e["description"], ""]
        if e.get("displacement_units") is not None and e["event_type"] == "moved":
            lines.append(f"- displacement: {scale.fmt(e['displacement_units'], tau)} ({scale.status} scale)")
        if e.get("rotation_deg") is not None and e["event_type"] == "moved":
            lines.append(f"- rotation: {e['rotation_deg']:.0f}°")
        lines.append(f"- identity: {e['identity_quality']}")
        for k, v in e["geometric_evidence"].items():
            if isinstance(v, dict):
                lines.append(f"- {k.replace('_', ' ')}: " + ", ".join(
                    f"{kk.replace('_', ' ')} {vv:.0%}" for kk, vv in v.items() if isinstance(vv, float) and vv > 0.005))
        fr = e["supporting_frames"]
        lines.append(f"- supporting keyframes: before {fr.get('before', [])[:12]}, after {fr.get('after', [])[:12]}")
        lines.append(f"- witness rays: {len(e.get('witness_rays') or [])}")
        for l in e.get("limitations", []):
            lines.append(f"- _{l}_")
        return "\n".join(lines)

    def look(position, look_at):
        for c in server.get_clients().values():
            c.camera.up_direction = (0.0, 0.0, 1.0)
            c.camera.position = tuple(map(float, position))
            c.camera.look_at = tuple(map(float, look_at))

    def select(eid):
        for k, nodes in ev_nodes.items():
            for n in nodes:
                if n.name.endswith("/rays"):
                    n.visible = (k == eid) and cb_rays.value
        for i, h in enumerate(cams["A"]):
            h.visible = cb_cams.value or (eid in by_id and i in by_id[eid]["supporting_frames"].get("before", []))
        for i, h in enumerate(cams["B"]):
            h.visible = cb_cams.value or (eid in by_id and i in by_id[eid]["supporting_frames"].get("after", []))
        if eid not in by_id:
            md.content = "Select an event or click a box in the scene."
            return
        e = by_id[eid]
        md.content = describe(e)
        f = out / "evidence" / f"{eid}.jpg"
        if f.exists():
            img_handle.image = cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB)
        if cb_focus.value and eid in ev_focus:
            # view from the side the cameras were on, at a distance that frames before+after
            c, ext = ev_focus[eid]
            r = float(np.clip(1.3 * ext, 1.2, 2.4))
            cams_c = S(A.centers()).mean(0)
            dvec = cams_c[:2] - c[:2]
            dvec /= np.linalg.norm(dvec) + 1e-9
            # steep oblique view from the side, perpendicular to the viewing rays, so the
            # cameras do not stand between the viewer and the object
            side = np.array([-dvec[1], dvec[0]])
            look(c + np.array([side[0] * 0.9 * r + 0.3 * dvec[0] * r, side[1] * 0.9 * r + 0.3 * dvec[1] * r, 1.2 * r]), c)

    def refresh_events():
        keep = ids_for(dd_filter.value)
        for k, nodes in ev_nodes.items():
            for n in nodes:
                if not n.name.endswith("/rays"):
                    n.visible = cb_overlays.value and k in keep

    @dd_filter.on_update
    def _(_):
        dd_event.options = ["(none)"] + ids_for(dd_filter.value)
        refresh_events()

    cb_overlays.on_update(lambda _: refresh_events())

    @dd_event.on_update
    def _(_):
        select(dd_event.value)

    @cb_rays.on_update
    def _(_):
        select(dd_event.value)

    for k, nodes in ev_nodes.items():
        for n in nodes:
            if n.name.endswith("_box"):
                def make(eid):
                    def cb(_):
                        dd_event.value = eid
                    return cb
                n.on_click(make(k))

    # ---------------------------------------------------------------- navigation
    grid = N.build_grid(A, B, floor, scale.meters_per_unit)
    N.add_change_evidence(grid, floor, vis["B"]["P"][vis["B"]["status"] == 1], vis["A"]["P"][vis["A"]["status"] == 1])
    route = N.walked_route(A, floor, grid)
    gimg = N.grid_image(grid)
    ext = np.array(gimg.shape[:2]) * grid.res * mpu
    gimg = np.repeat(np.repeat(gimg, 6, 0), 6, 1)          # nearest-neighbour upsample: crisp cells
    center = grid.origin * mpu + ext / 2
    img_xy = np.ascontiguousarray(np.transpose(gimg, (1, 0, 2)))   # image columns -> +x, rows -> +y
    h_navimg = server.scene.add_image("/nav/grid", img_xy, render_width=float(ext[0]), render_height=float(ext[1]),
                                      position=(float(center[0]), float(center[1]), -0.001), visible=False)
    route3 = np.concatenate([route * mpu, np.full((len(route), 1), 0.01)], 1).astype(np.float32)
    h_route = server.scene.add_spline_catmull_rom("/nav/walked_route", route3, color=INK, line_width=3.0, visible=False)
    nav_nodes = [h_navimg, h_route]

    with server.gui.add_folder("Navigation impact", expand_by_default=False):
        cb_nav = server.gui.add_checkbox("Floor map + walked route", False)
        sl_r = server.gui.add_slider("Footprint radius (m)", 0.1, 0.5, 0.01, nav_cfg.get("robot_radius_m", 0.22))
        sl_c = server.gui.add_slider("Clearance (m)", 0.0, 0.2, 0.01, nav_cfg.get("clearance_m", 0.05))
        bt_check = server.gui.add_button("Check walked route")
        bt_pick = server.gui.add_button("Pick start + goal (click floor twice)")
        server.gui.add_markdown("<small>light: observed free · dark: occupied · grey: unassessed · "
                                "<b style='color:#2a78d6'>blue: new obstacle</b> (free before) · "
                                "<b style='color:#eb6834'>orange: vacated</b> · <b style='color:#c81e1e'>red: on the route</b></small>")
        md_res = server.gui.add_markdown("")

    @cb_nav.on_update
    def _(_):
        for n in nav_nodes:
            n.visible = cb_nav.value

    def show_hits(res, name):
        hits = np.asarray(res.get("hit_cells_xy") or np.zeros((0, 2))).reshape(-1, 2) * mpu
        if len(hits):
            server.scene.add_point_cloud(f"/nav/{name}_hits", np.concatenate([hits, np.full((len(hits), 1), 0.02)], 1).astype(np.float32),
                                         (200, 30, 30), point_size=grid.res * mpu * 1.2, point_shape="square")
        if res.get("replan_after_path_xy"):
            p = np.asarray(res["replan_after_path_xy"]) * mpu
            server.scene.add_spline_catmull_rom(f"/nav/{name}_replan_after", np.concatenate([p, np.full((len(p), 1), 0.015)], 1),
                                                color=AFTER, line_width=3.0)

    def fmt_res(res, what):
        s = [f"**{what}**", f"- new-obstacle cells in swept footprint: {res['new_obstacle_cells_on_route']}",
             f"- unassessed after: {res['unassessed_fraction']:.0%} of the swept footprint"]
        for b in res["blocking_events"]:
            s.append(f"- **the {b['label']} ({b['event_id']}) intersects the route**")
        if res.get("replan_after_len_m") is not None:
            s.append(f"- replanned on the after map: {res['replan_after_len_m']:.2f} m, "
                     f"{res['replan_after_unassessed_fraction']:.0%} over unassessed floor")
        else:
            s.append("- no route found on the after map for this footprint")
        return "\n".join(s)

    def check_walked():
        cb_nav.value = True
        res = N.route_impact(grid, route, sl_r.value, sl_c.value, scale.meters_per_unit,
                             [SimpleNamespace(**e) for e in events], floor)
        show_hits(res, "walked")
        md_res.content = fmt_res(res, "Walked route of the before recording")

    bt_check.on_click(lambda _: check_walked())
    picks = []

    @bt_pick.on_click
    def _(_):
        picks.clear()
        md_res.content = "Click the start point on the floor…"
        cb_nav.value = True

        @server.scene.on_pointer_event(event_type="click")
        def _(ev):
            o, dvec = np.asarray(ev.ray_origin), np.asarray(ev.ray_direction)
            if abs(dvec[2]) < 1e-6 or -o[2] / dvec[2] <= 0:
                return
            p = (o + (-o[2] / dvec[2]) * dvec)[:2]
            picks.append(p)
            server.scene.add_icosphere(f"/nav/pick{len(picks)}", radius=0.03, color=INK, position=(p[0], p[1], 0.03))
            if len(picks) == 1:
                md_res.content = "Click the goal point…"
                return
            server.scene.remove_pointer_callback()
            path, _u = N.plan(grid, picks[0] / mpu, picks[1] / mpu, sl_r.value, sl_c.value, scale.meters_per_unit, which="before")
            if path is None:
                md_res.content = "No route on the before map for this footprint."
                return
            server.scene.add_spline_catmull_rom("/nav/planned_before", np.concatenate([path * mpu, np.full((len(path), 1), 0.012)], 1),
                                                color=INK, line_width=3.0)
            res = N.route_impact(grid, path, sl_r.value, sl_c.value, scale.meters_per_unit,
                                 [SimpleNamespace(**e) for e in events], floor)
            show_hits(res, "planned")
            md_res.content = fmt_res(res, "Planned route (before map)")

    # ---------------------------------------------------------------- camera presets
    centre = np.median(Qa, axis=0)
    centre[2] = 0.3
    back = S(A.centers()).mean(0) - centre
    back[2] = 0
    back /= np.linalg.norm(back) + 1e-9
    radius = float(np.percentile(np.linalg.norm(Qa[:, :2] - centre[:2], axis=1), 90))
    presets = {"overview": (centre + back * 1.5 * radius + np.array([0, 0, 1.2 * radius]), centre),
               "top-down": (centre + np.array([0.0, -0.02, 2.4 * radius]), centre)}
    with server.gui.add_folder("View", expand_by_default=False):
        dd_view = server.gui.add_dropdown("Camera preset", list(presets))
        bt_view = server.gui.add_button("Go")
    bt_view.on_click(lambda _: look(*presets[dd_view.value]))
    server.initial_camera.look_at = tuple(map(float, presets["overview"][1]))
    server.initial_camera.position = tuple(map(float, presets["overview"][0]))
    server.initial_camera.up = (0.0, 0.0, 1.0)

    print(f"RoomDelta viewer: http://localhost:{port}  ({len(ev_nodes)} non-unchanged events)", flush=True)
    handles = SimpleNamespace(server=server, select=lambda eid: setattr(dd_event, "value", eid), look=look, presets=presets,
                              ev_focus=ev_focus, cb_dim=cb_dim, cb_status=cb_status, cb_after=cb_after, cb_before=cb_before,
                              cb_nav=cb_nav, check_walked=check_walked, dd_filter=dd_filter, cb_focus=cb_focus, S=S,
                              cb_overlays=cb_overlays, cb_floor=cb_floor)
    if block:
        while True:
            time.sleep(1.0)
    return handles
