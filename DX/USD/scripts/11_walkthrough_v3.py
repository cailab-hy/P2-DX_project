"""Stage 11 (v3): walkthrough video of DX_Hub_Building_no_obj_v1 (Isaac Sim 6.1 headless, RTX real-time).

Route and camera settings: config/dx_hub_no_obj_v1_walkthrough.json (WALK_CFG=<json> selects another config, e.g.
config/dx_hub_add_obj_v1_walkthrough.json for the furnished scene: scene / variant_prim / out_dir / frames_dir / video /
frame_step (2 = double speed) / settle_s (run physics first so deformables drape) / draw_colliders_under). Before rendering, every camera position is tested with a
PhysX sphere overlap (radius clearance_m) against the building colliders, and the route is drawn on the plan
-> walkthrough/walkthrough_path.png. Frames -> WORK/walkthrough_frames/*.jpg; encode with 11b_encode_walkthrough.py.

Env: WALK_MODE=check (path + clearance only) | render (default); WALK_START / WALK_END / WALK_STEP (frame subset);
     WALK_FORCE=1 renders even if the clearance test fails.
"""
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

USD = "/home/cai/P2&DX_project/P2&DX/USD"
V3 = f"{USD}/DX_Hub_Building_no_obj_v1"
CFG = json.load(open(os.environ.get("WALK_CFG", f"{USD}/config/dx_hub_no_obj_v1_walkthrough.json")))
OUTD = CFG.get("out_dir", f"{V3}/walkthrough")
FRAMES = CFG.get("frames_dir", f"{USD}/work/walkthrough_frames")
PLAN = json.load(open(f"{USD}/config/dx_hub_no_obj_v1_plan.json"))
MODE = os.environ.get("WALK_MODE", "render")


# ------------------------------------------------------------------ route -> per-frame (x, y, yaw)
def rounded_polyline(pts, r):
    """Dense polyline (0.01 m) through pts with each interior corner replaced by a quadratic Bezier of radius ~r."""
    pts = [np.asarray(p, float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts)):
        p, q = pts[i - 1], pts[i]
        start = out[-1]
        if i < len(pts) - 1:
            a = (q - p) / np.linalg.norm(q - p)
            b = (pts[i + 1] - q) / np.linalg.norm(pts[i + 1] - q)
            rr = min(r, 0.45 * np.linalg.norm(q - p), 0.45 * np.linalg.norm(pts[i + 1] - q))
            c0, c1 = q - a * rr, q + b * rr
        else:
            c0 = c1 = q
        n = max(2, int(np.linalg.norm(c0 - start) / 0.01))
        out += list(start + (c0 - start) * np.linspace(0, 1, n)[1:, None])
        if c1 is not c0:
            u = np.linspace(0, 1, 40)[1:, None]
            out += list((1 - u) ** 2 * c0 + 2 * u * (1 - u) * q + u ** 2 * c1)
    return np.array(out)


def trapezoid_s(L, v, a, dt):
    """Arc length at each frame for a rest-to-rest move of length L (max speed v, acceleration a)."""
    d_acc = v * v / (2 * a)
    if L < 2 * d_acc:
        v = math.sqrt(L * a)
        d_acc = L / 2
    t_acc = v / a
    T = 2 * t_acc + (L - 2 * d_acc) / v
    ts = np.arange(0, T + dt, dt)
    s = np.where(ts < t_acc, 0.5 * a * ts ** 2,
                 np.where(ts < T - t_acc, d_acc + v * (ts - t_acc), L - 0.5 * a * np.clip(T - ts, 0, None) ** 2))
    return np.clip(s, 0, L)


def ease_turn(y0, delta, dps, dt):
    n = max(2, int(abs(delta) / dps / dt))
    u = np.linspace(0, 1, n)
    return y0 + delta * (1 - np.cos(np.pi * u)) / 2


def build_route(cfg):
    dt = 1.0 / cfg["fps"]
    la = cfg["lookahead_m"]
    xy = [np.asarray(cfg["start"], float)]
    yaw = None
    base = math.radians(cfg["pitch_deg"])
    pit = base
    X, Y, YAW, PIT, LABEL = [], [], [], [], []

    def emit(px, py, pyaw, label, ppit=None):
        X.extend(px); Y.extend(py); YAW.extend(pyaw); LABEL.extend([label] * len(px))
        PIT.extend([pit] * len(px) if ppit is None else ppit)

    for seg in cfg["route"]:
        cur = xy[-1]
        if "look" in seg:                                      # optional "pitch" (deg, absolute; next look without it returns
            ys = ease_turn(yaw, math.radians(seg["look"]), math.radians(cfg["look_dps"]), dt)   # to pitch_deg), "hold_s" pause
            p1 = math.radians(seg["pitch"]) if "pitch" in seg else base
            ps = pit + (p1 - pit) * (1 - np.cos(np.pi * np.linspace(0, 1, len(ys)))) / 2
            emit([cur[0]] * len(ys), [cur[1]] * len(ys), ys, "look", list(ps))
            yaw, pit = ys[-1], p1
            nh = int(seg.get("hold_s", 0) / dt)
            emit([cur[0]] * nh, [cur[1]] * nh, [yaw] * nh, "look")
            continue
        poly = rounded_polyline([cur] + seg["go"], cfg["corner_radius_m"])
        cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]
        L = cum[-1]

        def at(s):
            return np.stack([np.interp(s, cum, poly[:, 0]), np.interp(s, cum, poly[:, 1])], -1)

        s = trapezoid_s(L, cfg["speed_mps"], cfg["accel_mps2"], dt)
        ahead = np.minimum(s + la, L)
        d = at(ahead) - at(np.maximum(ahead - la, 0))
        heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        if yaw is None:
            yaw = heading[0]
            emit([cur[0]] * int(cfg["hold_start_s"] / dt), [cur[1]] * int(cfg["hold_start_s"] / dt),
                 [yaw] * int(cfg["hold_start_s"] / dt), "hold")
        heading += 2 * math.pi * round((yaw - heading[0]) / (2 * math.pi))
        if abs(heading[0] - yaw) > math.radians(3):
            ys = ease_turn(yaw, heading[0] - yaw, math.radians(cfg["turn_dps"]), dt)
            emit([cur[0]] * len(ys), [cur[1]] * len(ys), ys, "turn")
        p = at(s)
        if abs(pit - base) > 1e-6:                             # walking always uses the base pitch
            pit = base
        emit(p[:, 0], p[:, 1], heading, seg.get("name", "go"))
        yaw = heading[-1]
        xy.append(p[-1])
    n_end = int(cfg["hold_end_s"] / dt)
    emit([X[-1]] * n_end, [Y[-1]] * n_end, [YAW[-1]] * n_end, "hold")
    yaw_arr = np.array(YAW)
    sig = cfg["yaw_smooth_s"] / dt
    k = np.arange(-int(4 * sig), int(4 * sig) + 1)
    w = np.exp(-0.5 * (k / sig) ** 2)
    w /= w.sum()
    yaw_arr = np.convolve(np.pad(yaw_arr, len(k) // 2, mode="edge"), w, mode="valid")
    pit_arr = np.convolve(np.pad(np.array(PIT), len(k) // 2, mode="edge"), w, mode="valid")
    return np.array(X), np.array(Y), yaw_arr, pit_arr, LABEL


X, Y, YAW, PITCH, LABEL = build_route(CFG)
n_frames = len(X)
dist = float(np.sum(np.hypot(np.diff(X), np.diff(Y))))
print(f"route: {n_frames} frames = {n_frames / CFG['fps']:.1f} s, {dist:.1f} m walked")

# ------------------------------------------------------------------ Isaac
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

W_, H_ = CFG["resolution"]
app = SimulationApp({"headless": True, "width": W_, "height": H_, "renderer": "RaytracedLighting"})
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

ctx = omni.usd.get_context()
ctx.open_stage(CFG.get("scene", f"{V3}/DX_Hub_Building_no_obj_v1_scene.usda"))
for _ in range(40):
    app.update()
stage = ctx.get_stage()
hub = stage.GetPrimAtPath(CFG.get("variant_prim", "/World/DX_Hub_Building"))
for vs, sel in CFG["variants"].items():
    hub.GetVariantSet(vs).SetVariantSelection(sel)
for _ in range(30):
    app.update()

# clearance: sphere overlaps at the eye along the whole route
tl = omni.timeline.get_timeline_interface()
tl.play()
for _ in range(5):
    app.update()
sq = omni.physx.get_physx_scene_query_interface()
hits = []
for i in range(0, n_frames, 2):
    found = []

    def report(h, found=found):
        found.append(h.collision.split("/")[-1])
        return True

    sq.overlap_sphere(CFG["clearance_m"], Gf.Vec3d(float(X[i]), float(Y[i]), CFG["eye_z"]), report, False)
    if found:
        hits.append((i, round(float(X[i]), 2), round(float(Y[i]), 2), LABEL[i], sorted(set(found))))
probe = []                                                      # sanity: a point inside the corridor wall must hit
sq.overlap_sphere(CFG["clearance_m"], Gf.Vec3d(3.62, -18.0, CFG["eye_z"]),
                  lambda h: probe.append(h.collision) or True, False)
assert probe, "overlap query returned nothing for a point inside a wall"
tl.stop()
app.update()
os.makedirs(OUTD, exist_ok=True)
print(f"clearance {CFG['clearance_m']} m: {len(hits)} sampled frames touch a collider")
for h in hits[:40]:
    print("  ", h)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(figsize=(9, 14))
for w in PLAN["walls"]:
    t = w.get("t", 0.12)
    if w["axis"] == "x":
        ax.add_patch(plt.Rectangle((w["c"] - t / 2, w["v0"]), t, w["v1"] - w["v0"], color="0.25"))
    else:
        ax.add_patch(plt.Rectangle((w["v0"], w["c"] - t / 2), w["v1"] - w["v0"], t, color="0.25"))
for d in PLAN["diagonal_walls"]:
    ax.plot([d["p0"][0], d["p1"][0]], [d["p0"][1], d["p1"][1]], color="0.25", lw=2)
for g in json.load(open(f"{USD}/config/dx_hub_no_obj_v1_openings.json"))["glass_lines"]:
    a = (g["c"], g["c"]) if g["axis"] == "x" else (g["a0"], g["a1"])
    b = (g["a0"], g["a1"]) if g["axis"] == "x" else (g["c"], g["c"])
    ax.plot(a, b, color="deepskyblue", lw=2)
for c in PLAN["columns"]:
    ax.add_patch(plt.Rectangle((c["x0"], c["y0"]), c["x1"] - c["x0"], c["y1"] - c["y0"], color="green"))
if CFG.get("draw_colliders_under"):                               # furniture footprints
    from pxr import Usd, UsdPhysics  # noqa: E402
    bbc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "guide"])
    for p_ in Usd.PrimRange(stage.GetPrimAtPath(CFG["draw_colliders_under"])):
        if p_.HasAPI(UsdPhysics.CollisionAPI):
            r_ = bbc.ComputeWorldBound(p_).ComputeAlignedRange()
            ax.add_patch(plt.Rectangle((r_.GetMin()[0], r_.GetMin()[1]), r_.GetSize()[0], r_.GetSize()[1], color="sienna", alpha=0.5, lw=0))
sc = ax.scatter(X[::5], Y[::5], c=np.arange(0, n_frames, 5) / CFG["fps"], cmap="viridis", s=3)
for i in range(0, n_frames, int(10 * CFG["fps"])):
    ax.annotate("", xy=(X[i] + 0.8 * math.cos(YAW[i]), Y[i] + 0.8 * math.sin(YAW[i])), xytext=(X[i], Y[i]),
                arrowprops=dict(arrowstyle="->", color="orangered"))
    ax.text(X[i] + 0.2, Y[i] + 0.2, f"{i // CFG['fps']}s", fontsize=6, color="orangered")
if hits:
    hh = np.array([(h[1], h[2]) for h in hits])
    ax.scatter(hh[:, 0], hh[:, 1], color="red", s=25, marker="x", label="clearance hit")
    ax.legend(loc="upper right")
plt.colorbar(sc, ax=ax, fraction=0.03, label="time [s]")
ax.set_aspect("equal")
ax.set_xlim(-14, 14)
ax.set_ylim(-23.5, 23.5)
ax.grid(alpha=0.3)
ax.set_title(f"v3 walkthrough route: {n_frames / CFG['fps']:.0f} s, {dist:.0f} m (arrows every 10 s)")
fig.savefig(f"{OUTD}/walkthrough_path.png", dpi=110, bbox_inches="tight")
json.dump({"frames": n_frames, "seconds": n_frames / CFG["fps"], "metres": dist, "clearance_hits": hits},
          open(f"{OUTD}/walkthrough_check.json", "w"), indent=1)

if MODE == "check" or (hits and os.environ.get("WALK_FORCE") != "1"):
    app.close()
    raise SystemExit

# ------------------------------------------------------------------ render
cam = UsdGeom.Camera.Define(stage, "/World/WalkCam")
cam.CreateFocalLengthAttr(CFG["focal_mm"])
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955 * H_ / W_)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
xop = UsdGeom.Xformable(cam).AddTransformOp()
ann = rep.AnnotatorRegistry.get_annotator("rgb")
rp = rep.create.render_product(str(cam.GetPath()), (W_, H_))
ann.attach([rp])
os.makedirs(FRAMES, exist_ok=True)
i0 = int(os.environ.get("WALK_START", 0))
i1 = int(os.environ.get("WALK_END", n_frames))
step = int(os.environ.get("WALK_STEP", CFG.get("frame_step", 1)))
if CFG.get("settle_s"):                                         # let deformables / bodies settle; frames then pause the timeline
    from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
    SimulationManager.set_physics_sim_device("cuda:0")
    tl.play()
    for _ in range(int(CFG["settle_s"] * 60)):
        app.update()
pool = ThreadPoolExecutor(4)
for _ in range(10):                                             # warm-up: textures, BVH
    rep.orchestrator.step(rt_subframes=CFG["rt_subframes"])
import time  # noqa: E402
t0 = time.time()
for n, i in enumerate(range(i0, i1, step)):
    eye = Gf.Vec3d(float(X[i]), float(Y[i]), CFG["eye_z"])
    pitch = float(PITCH[i])
    fwd = Gf.Vec3d(math.cos(YAW[i]) * math.cos(pitch), math.sin(YAW[i]) * math.cos(pitch), math.sin(pitch))
    xop.Set(Gf.Matrix4d().SetLookAt(eye, eye + fwd, Gf.Vec3d(0, 0, 1)).GetInverse())
    rep.orchestrator.step(rt_subframes=CFG["rt_subframes"])
    img = np.asarray(ann.get_data())[..., :3].copy()
    pool.submit(lambda a, k: Image.fromarray(a).save(f"{FRAMES}/f{k:05d}.jpg", quality=93), img, i)
    if n % 200 == 0:
        print(f"frame {i}/{n_frames}  {(time.time() - t0) / (n + 1):.3f} s/frame", flush=True)
pool.shutdown(wait=True)
print(f"rendered {len(range(i0, i1, step))} frames in {time.time() - t0:.0f} s -> {FRAMES}")
ann.detach()
rp.destroy()
app.close()
