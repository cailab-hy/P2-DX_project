"""Stage 26 (add_obj_v1): quick renders of selected config validation_views (real photo | USD) without the physics checks.

Usage: python 26_render_views_add_obj.py <out_tag> view1 [view2 ...]   -> validation/views_<out_tag>.png
"""
import json
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

USD = "/home/cai/P2&DX_project/P2&DX/USD"
DST = f"{USD}/DX_Hub_Building_add_obj_v1"
CFG = json.load(open(f"{USD}/config/add_obj_v1_furniture.json"))
tag, names = sys.argv[1], sys.argv[2:]
V = CFG["validation_views"]
ctx = omni.usd.get_context()
ctx.open_stage(f"{DST}/DX_Hub_Building_add_obj_v1_scene.usda")
for _ in range(40):
    app.update()
st = ctx.get_stage()
st.GetPrimAtPath("/World/DX_Hub_Building_add_obj/Building").GetVariantSet("lighting").SetVariantSelection("day")
for _ in range(20):
    app.update()
anns = {}
for k in names:
    cam = UsdGeom.Camera.Define(st, f"/World/QCam_{k}")
    cam.CreateFocalLengthAttr(13.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 3 / 4)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200))
    UsdGeom.Xformable(cam).AddTransformOp().Set(
        Gf.Matrix4d().SetLookAt(Gf.Vec3d(*V[k]["eye"]), Gf.Vec3d(*V[k]["target"]), Gf.Vec3d(0, 0, 1)).GetInverse())
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    a.attach([rep.create.render_product(str(cam.GetPath()), (800, 600))])
    anns[k] = a
for _ in range(3):
    rep.orchestrator.step(rt_subframes=16)
S = Image.new("RGB", (1600, 626 * len(names)), "white")
d = ImageDraw.Draw(S)
for i, k in enumerate(names):
    if V[k]["ref"]:
        S.paste(ImageOps.fit(ImageOps.exif_transpose(Image.open(f"{USD}/work/add_obj_v1/{V[k]['ref']}")).convert("RGB"), (800, 600)), (0, 626 * i + 26))
    S.paste(Image.fromarray(np.asarray(anns[k].get_data())[..., :3]), (800, 626 * i + 26))
    d.text((8, 626 * i + 6), f"{k} - real", fill=(0, 0, 0))
    d.text((808, 626 * i + 6), f"{k} - USD", fill=(0, 0, 0))
S.save(f"{DST}/validation/views_{tag}.png")
print("wrote", f"views_{tag}.png")
app.close()
