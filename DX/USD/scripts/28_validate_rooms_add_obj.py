"""Stage 28 (add_obj_v1): check the per-space USDs (27_split_rooms_add_obj.py) in Isaac Sim 6.1 (GPU physics).

For each space scene: active furniture groups (only its own), rigid-body drift over 3 s, deformables finite, and two
renders - an orthographic top-down view of the space boxes (camera below the ceiling) and a perspective view.
The sync test (edit of the full file shows up in the space files) runs outside Isaac Sim: 27_split_rooms_add_obj.py --sync-test.
Outputs: DX_Hub_Building_add_obj_v1/validation/rooms_split_check.json, rooms_split_views.png
"""
import json
import math
import os
import shutil

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

USD = "/home/cai/P2&DX_project/P2&DX/USD"
DST = f"{USD}/DX_Hub_Building_add_obj_v1"
CFG = json.load(open(f"{USD}/config/add_obj_v1_furniture.json"))
ROOMS = {k: v for k, v in CFG["rooms"].items() if not k.startswith("_") and isinstance(v, dict)}
PERSP = {"Hall": ((-1.5, 3.0, 2.0), (-8.0, 14.0, 0.6)), "Kitchen": ((-1.8, 13.4, 2.0), (2.0, 7.0, 0.8)),
         "CareRoom": ((-3.8, 17.0, 2.0), (2.0, 20.0, 0.6))}
FU = "/World/DX_Hub_Building_add_obj/Furniture"
ctx = omni.usd.get_context()
tl = omni.timeline.get_timeline_interface()
px = omni.physx.get_physx_interface()
report, panels = {}, {}

for name, rc in ROOMS.items():
    ctx.open_stage(f"{DST}/DX_Hub_Building_add_obj_v1_{name}_scene.usda")
    for _ in range(40):
        app.update()
    stage = ctx.get_stage()
    stage.GetPrimAtPath("/World/DX_Hub_Building_add_obj/Building").GetVariantSet("lighting").SetVariantSelection("day")
    for _ in range(10):
        app.update()
    groups = [c.GetName() for c in stage.GetPrimAtPath(FU).GetChildren()]   # active ones only
    bodies = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(FU)) if p.HasAPI(UsdPhysics.RigidBodyAPI)
              and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get() is not False]
    defs = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(FU)) if "OmniPhysicsDeformableBodyAPI" in p.GetAppliedSchemas()]
    n_build = sum(1 for p in Usd.PrimRange(stage.GetPrimAtPath("/World/DX_Hub_Building_add_obj/Building")) if p.IsA(UsdGeom.Gprim))
    # cameras
    bx = np.array(rc["boxes"])
    x0, x1, y0, y1 = bx[:, 0].min(), bx[:, 1].max(), bx[:, 2].min(), bx[:, 3].max()
    W, H = x1 - x0, y1 - y0
    ppm = 700 / max(W, H)
    top = UsdGeom.Camera.Define(stage, "/World/RoomTop")
    top.CreateProjectionAttr("orthographic")
    top.CreateHorizontalApertureAttr(W * 10)
    top.CreateVerticalApertureAttr(H * 10)
    top.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100))
    c = Gf.Vec3d((x0 + x1) / 2, (y0 + y1) / 2, 2.35)
    UsdGeom.Xformable(top).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(c, c - Gf.Vec3d(0, 0, 1), Gf.Vec3d(0, 1, 0)).GetInverse())
    per = UsdGeom.Camera.Define(stage, "/World/RoomPersp")
    per.CreateFocalLengthAttr(12.0)
    per.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200))
    e, t = PERSP[name]
    UsdGeom.Xformable(per).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*e), Gf.Vec3d(*t), Gf.Vec3d(0, 0, 1)).GetInverse())
    anns = {}
    for k, cam, res in (("top", top, (int(W * ppm), int(H * ppm))), ("persp", per, (800, 500))):
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        rp = rep.create.render_product(str(cam.GetPath()), res)
        a.attach([rp])
        anns[k] = (a, rp)
    SimulationManager.set_physics_sim_device("cuda:0")              # after the stage is open (before it, Kit crashes)
    tl.play()
    for _ in range(5):
        app.update()
    p0 = {b: np.array(px.get_rigidbody_transformation(b)["position"]) for b in bodies}
    for _ in range(180):
        app.update()
    p1 = {b: np.array(px.get_rigidbody_transformation(b)["position"]) for b in bodies}
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=12, pause_timeline=False)
    panels[name] = {k: Image.fromarray(np.asarray(a.get_data())[..., :3]) for k, (a, _) in anns.items()}
    tl.stop()
    for a, rp in anns.values():
        a.detach()
        rp.destroy()
    drift = max([float(np.linalg.norm(p1[b] - p0[b])) for b in bodies] or [0.0])
    report[name] = {"furniture_groups_active": groups, "rigid_bodies": len(bodies), "max_drift_m": round(drift, 4),
                    "deformables": [d.split("/")[-1] for d in defs], "building_gprims_active": n_build}
    print(name, report[name], flush=True)
json.dump(report, open(f"{DST}/validation/rooms_split_check.json", "w"), indent=1)

# ------------------------------------------------------------------ sheet: top view | perspective per space
rows = []
for name in ROOMS:
    tp, pp = panels[name]["top"], panels[name]["persp"]
    tp.thumbnail((700, 700))
    h = max(tp.height, pp.height) + 30
    r_ = Image.new("RGB", (1520, h), "white")
    r_.paste(tp, (0, 30))
    r_.paste(pp, (720, 30))
    ImageDraw.Draw(r_).text((8, 8), f"{name}: top view (left) and perspective (right)", fill=(0, 0, 0))
    rows.append(r_)
S = Image.new("RGB", (1520, sum(r_.height for r_ in rows)), "white")
y = 0
for r_ in rows:
    S.paste(r_, (0, y))
    y += r_.height
S.save(f"{DST}/validation/rooms_split_views.png")
app.close()
