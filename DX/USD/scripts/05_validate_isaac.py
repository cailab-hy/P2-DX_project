"""Stage 5: open DX_Hub_scene.usda in Isaac Sim (headless) and check physics + rendering.

  * PhysX raycasts onto the collision mesh (floor height, wall distance)
  * drop test: rigid cubes fall onto the floor and must come to rest at z ~= half size
  * RTX renders of the "clean", "gsplat" and "scan" appearance variants -> OUT/validation/*.png
"""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, UsdGeom, UsdPhysics  # noqa: E402

OUT = "/home/cai/P2&DX_project/P2&DX/USD/DX_Hub"
os.makedirs(f"{OUT}/validation", exist_ok=True)
ext = omni.kit.app.get_app().get_extension_manager()
for e in ["omni.usd.schema.usd_particle_field"]:
    ext.set_extension_enabled_immediate(e, True)

ctx = omni.usd.get_context()
ctx.open_stage(f"{OUT}/DX_Hub_scene.usda")
for _ in range(30):
    app.update()
stage = ctx.get_stage()
hub = stage.GetPrimAtPath("/World/DX_Hub")
splat = stage.GetPrimAtPath("/World/DX_Hub/GaussianSplat/Splat")
print("[check] hub variants:", hub.GetVariantSets().GetNames(), "| structure loaded:",
      stage.GetPrimAtPath("/World/DX_Hub/Structure/Walls").IsValid())

# ---------------- physics ----------------
cubes = [(-8.0, 10.0), (0.0, 0.0), (4.0, -15.0), (8.0, 12.0)]
for i, (x, y) in enumerate(cubes):
    c = UsdGeom.Cube.Define(stage, f"/World/Test/Cube{i}")
    c.CreateSizeAttr(0.2)
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(x, y, 1.0))
    UsdPhysics.RigidBodyAPI.Apply(c.GetPrim())
    UsdPhysics.CollisionAPI.Apply(c.GetPrim())
    UsdPhysics.MassAPI.Apply(c.GetPrim()).CreateMassAttr(1.0)
tl = omni.timeline.get_timeline_interface()
tl.play()
for _ in range(240):
    app.update()
physx = omni.physx.get_physx_interface()
sq = omni.physx.get_physx_scene_query_interface()
for i, (x, y) in enumerate(cubes):
    tr = physx.get_rigidbody_transformation(f"/World/Test/Cube{i}")
    print(f"[physics] cube{i} start ({x},{y},1.0) -> rest z = {tr['position'][2]:.3f} (expect ~0.10)")
for x, y in [(-8, 10), (0, 0), (4, -15), (-10, -8), (10, 5)]:
    hit = sq.raycast_closest(Gf.Vec3d(x, y, 1.5), Gf.Vec3d(0, 0, -1), 5.0)
    print(f"[physics] floor ray at ({x},{y}): hit={hit['hit']} z={1.5 - hit.get('distance', np.nan):.3f} "
          f"collider={hit.get('collision', '')}")
# floor coverage: one downward ray per 1 m inside the interior footprint
mask = np.load("/home/cai/P2&DX_project/P2&DX/USD/work/interior_mask.npy")
mj = __import__("json").load(open("/home/cai/P2&DX_project/P2&DX/USD/work/footprint_mask.json"))
cells = np.argwhere(mask)[::16]
zs = []
for i, j in cells:
    x, y = (i + mj["lo"][0] + 0.5) * mj["cell"], (j + mj["lo"][1] + 0.5) * mj["cell"]
    h = sq.raycast_closest(Gf.Vec3d(x, y, 0.15), Gf.Vec3d(0, 0, -1), 1.0)
    zs.append(0.15 - h["distance"] if h["hit"] else np.nan)
zs = np.array(zs)
print(f"[physics] floor coverage: {np.isfinite(zs).mean() * 100:.1f}% of {len(zs)} interior rays hit; "
      f"|z|<3cm: {(np.abs(zs) < 0.03).mean() * 100:.1f}%; z pct(5,50,95) = {np.nanpercentile(zs, [5, 50, 95]).round(3)}")
up = sq.raycast_closest(Gf.Vec3d(0, 0, 1.0), Gf.Vec3d(0, 0, 1), 10.0)
print(f"[physics] ceiling above origin at z={1.0 + up.get('distance', np.nan):.3f}")
tl.stop()
app.update()
stage.RemovePrim("/World/Test")

# ---------------- rendering ----------------
def camera(path, eye, target):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateFocalLengthAttr(14.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
    m = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(cam).AddTransformOp().Set(m)
    return path


views = {
    "counter": camera("/World/Cams/Counter", (1.2, -2.2, 2.0), (2.0, 12.0, 0.6)),
    "hall_pillar": camera("/World/Cams/HallPillar", (1.0, 1.5, 1.8), (-9.0, 9.0, 0.8)),
    "hall": camera("/World/Cams/Hall", (-11.0, 1.5, 1.6), (-4.0, 12.0, 1.0)),
    "corridor": camera("/World/Cams/Corridor", (5.0, -18.0, 1.6), (5.0, 10.0, 1.2)),
    "east_room": camera("/World/Cams/East", (7.5, 9.5, 1.6), (11.5, 17.0, 0.8)),
    "overview": camera("/World/Cams/Overview", (0.0, -38.0, 30.0), (0.0, -2.0, 0.0)),
}
rps = {k: rep.create.render_product(p, (1280, 720)) for k, p in views.items()}
annot = {}
for k, rp in rps.items():
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rp])
    annot[k] = a
ceiling = [UsdGeom.Imageable(stage.GetPrimAtPath(f"/World/DX_Hub/Structure/{n}")) for n in ("Ceiling", "LightPanels")]

for variant in ["clean", "clean+gsplat_objects", "gsplat", "scan"]:
    hub.GetVariantSet("appearance").SetVariantSelection(variant.split("+")[0])
    hub.GetVariantSet("objects").SetVariantSelection("gsplat" if "gsplat_objects" in variant else "scan")
    for _ in range(20):
        app.update()
    if variant == "gsplat":
        print("[check] splat prim type:", stage.GetPrimAtPath("/World/DX_Hub/GaussianSplat/Splat").GetTypeName())
    for k in views:
        if k == "overview" and variant.startswith("clean"):
            for c_ in ceiling:
                c_.MakeInvisible()
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=16)
        img = annot[k].get_data()
        Image.fromarray(np.asarray(img)[..., :3]).save(f"{OUT}/validation/{variant}_{k}.png")
        print(f"[render] {variant}_{k}.png mean={np.asarray(img)[..., :3].mean():.1f}")
        if k == "overview" and variant.startswith("clean"):
            for c_ in ceiling:
                c_.MakeVisible()
app.close()
