"""Stage 10 (v3): Isaac Sim 6.1 headless checks of DX_Hub_Building_no_obj_v1.

Physics : cubes dropped in every room and on the counters must rest on the measured surfaces; horizontal rays must
          stop at the glass partitions A-F; a closed door blocks a ray that passes once doors=open.
Render  : the same camera poses rendered for v2 (DX_Hub_Building), v3 and the Gaussian splat (DX_Hub, appearance=gsplat)
          -> validation/compare_<view>.png (3 columns) + compare_all.png.
"""
import json
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

USD = "/home/cai/P2&DX_project/P2&DX/USD"
V3 = f"{USD}/DX_Hub_Building_no_obj_v1"
OUTD = f"{V3}/validation"
os.makedirs(OUTD, exist_ok=True)
ctx = omni.usd.get_context()
report = {}


def open_stage(path, settle=40):
    ctx.open_stage(path)
    for _ in range(settle):
        app.update()
    return ctx.get_stage()


# ------------------------------------------------------------------ physics on v3
stage = open_stage(f"{V3}/DX_Hub_Building_no_obj_v1_scene.usda")
hub = stage.GetPrimAtPath("/World/DX_Hub_Building")
drops = {"hall": (-8.0, 10.0), "corridor_NS": (5.0, 0.0), "east_room": (9.0, 12.0), "pantry": (1.5, 15.0),
         "SW_room": (-8.0, -10.0), "robot_room": (-9.0, -3.0), "room_A": (1.5, -8.0),
         "U_north_top(arm)": (3.25, 8.0), "U_south_top": (1.5, -0.4), "bar_mid_top": (-0.3, 9.6), "low_base_top": (-0.3, 10.7), "showcase_top": (-0.3, 11.1)}
for i, (n, (x, y)) in enumerate(drops.items()):
    c = UsdGeom.Cube.Define(stage, f"/World/Test/C{i}")
    c.CreateSizeAttr(0.2)
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(x, y, 1.8))
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
tl = omni.timeline.get_timeline_interface()
tl.play()
for _ in range(240):
    app.update()
px = omni.physx.get_physx_interface()
sq = omni.physx.get_physx_scene_query_interface()
report["drops_rest_z"] = {n: round(px.get_rigidbody_transformation(f"/World/Test/C{i}")["position"][2], 3)
                          for i, n in enumerate(drops)}
rays = {"glass A (room A west)": ((-2.2, -8.5, 1.0), (1, 0, 0)), "glass B (south rooms)": ((-3.0, -14.2, 1.0), (0, -1, 0)),
        "glass C doors (corridor east)": ((4.5, -2.9, 1.0), (1, 0, 0)), "glass C wall panel": ((4.5, -0.9, 1.0), (1, 0, 0)), "glass D (glass-door corridor)": ((-2.2, -8.5, 1.0), (-1, 0, 0)),
        "glass E (north glass doors)": ((-2.0, 15.0, 1.0), (0, 1, 0)), "glass F (robot room north)": ((-9.0, 2.0, 1.0), (0, -1, 0)),
        "east room door (closed)": ((5.0, 12.6, 1.0), (1, 0, 0))}


def cast(o, d):
    h = sq.raycast_closest(Gf.Vec3d(*o), Gf.Vec3d(*d), 30.0)
    if not h["hit"]:
        return None
    return [round(o[k] + d[k] * h["distance"], 3) for k in range(3)], h["collision"].split("/")[-1]


report["rays"] = {n: cast(o, d) for n, (o, d) in rays.items()}
tl.stop()
app.update()
stage.RemovePrim("/World/Test")
hub.GetVariantSet("doors").SetVariantSelection("open")
for _ in range(20):
    app.update()
tl.play()
for _ in range(5):
    app.update()
report["rays"]["east room door (doors=open)"] = cast((5.0, 12.6, 1.0), (1, 0, 0))
tl.stop()
app.update()
print(json.dumps(report, indent=1))

# ------------------------------------------------------------------ renders: v2 / v3 / gsplat at the same poses
VIEWS = {
    "counter": ((1.2, -2.2, 2.0), (2.0, 12.0, 0.6)), "hall_pillar": ((1.0, 1.5, 1.8), (-9.0, 9.0, 0.8)),
    "hall": ((-11.0, 1.5, 1.6), (-4.0, 12.0, 1.0)), "east_room": ((7.5, 9.5, 1.6), (11.5, 17.0, 0.8)),
    "corridor_NS": ((5.0, -18.0, 1.6), (5.0, 10.0, 1.2)), "west_windows": ((-9.0, 8.0, 1.5), (-13.5, 8.0, 1.3)),
    "glass_door_corridor": ((-2.2, -12.4, 1.6), (-2.2, -4.5, 1.2)), "east_corridor": ((6.6, -14.0, 1.6), (13.5, -14.0, 1.3)),
    "robot_room": ((-6.0, -5.5, 1.6), (-12.0, -0.5, 1.1)), "pantry": ((2.0, 12.9, 1.6), (1.0, 16.2, 1.2)),
}
SOURCES = [("v2", f"{USD}/DX_Hub_Building/DX_Hub_Building_scene.usda", None),
           ("v3", f"{V3}/DX_Hub_Building_no_obj_v1_scene.usda", None),
           ("gsplat", f"{USD}/DX_Hub/DX_Hub_scene.usda", ("/World/DX_Hub", "appearance", "gsplat"))]
imgs = {}
for tag, path, var in (SOURCES if os.environ.get("RENDER", "1") == "1" else []):
    st = open_stage(path)
    if var:
        st.GetPrimAtPath(var[0]).GetVariantSet(var[1]).SetVariantSelection(var[2])
        for _ in range(30):
            app.update()
    anns = {}
    for k, (eye, tgt) in VIEWS.items():
        cam = UsdGeom.Camera.Define(st, f"/World/Cam_{k}")
        cam.CreateFocalLengthAttr(14.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
        UsdGeom.Xformable(cam).AddTransformOp().Set(
            Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1)).GetInverse())
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        rp = rep.create.render_product(str(cam.GetPath()), (960, 540))
        a.attach([rp])
        anns[k] = (a, rp)
    for k in VIEWS:
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=16)
        imgs[(tag, k)] = Image.fromarray(np.asarray(anns[k][0].get_data())[..., :3])
    for a, rp in anns.values():
        a.detach()
        rp.destroy()
    print("rendered", tag)

rows = []
if not imgs:
    json.dump(report, open(f"{OUTD}/isaac_checks.json", "w"), indent=1)
    app.close()
    raise SystemExit
for k in VIEWS:
    W = Image.new("RGB", (3 * 640, 360 + 26), "white")
    dr = ImageDraw.Draw(W)
    for c, tag in enumerate(("v2", "v3", "gsplat")):
        W.paste(imgs[(tag, k)].resize((640, 360)), (c * 640, 26))
        dr.text((c * 640 + 8, 6), f"{k} - {tag}", fill=(0, 0, 0))
    W.save(f"{OUTD}/compare_{k}.png")
    rows.append(W)
ALL = Image.new("RGB", (rows[0].width, sum(r.height for r in rows)), "white")
y = 0
for r in rows:
    ALL.paste(r, (0, y))
    y += r.height
ALL.save(f"{OUTD}/compare_all.png")
json.dump(report, open(f"{OUTD}/isaac_checks.json", "w"), indent=1)
app.close()
