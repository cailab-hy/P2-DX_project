"""Render named viewpoints of v3 (and the Gaussian splat for reference) -> validation/views_<tag>.png.

Usage: python 10b_render_views_v3.py <tag> name:ex,ey,ez:tx,ty,tz [...]
"""
import os
import sys

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Tf, UsdGeom  # noqa: E402

USD = "/home/cai/P2&DX_project/P2&DX/USD"
tag = sys.argv[1]
views = []
for a in sys.argv[2:]:
    n, e, t = a.split(":")
    views.append((n, tuple(map(float, e.split(","))), tuple(map(float, t.split(",")))))
VAR = os.environ.get("V3_VARIANT")                     # e.g. "lighting=night"
SOURCES = [("v3", f"{USD}/DX_Hub_Building_no_obj_v1/DX_Hub_Building_no_obj_v1_scene.usda",
            ("/World/DX_Hub_Building", *VAR.split("=")) if VAR else None)]
if os.environ.get("NO_GSPLAT") != "1":
    SOURCES.append(("gsplat", f"{USD}/DX_Hub/DX_Hub_scene.usda", ("/World/DX_Hub", "appearance", "gsplat")))
ctx = omni.usd.get_context()
imgs = {}
for src, path, var in SOURCES:
    ctx.open_stage(path)
    for _ in range(40):
        app.update()
    st = ctx.get_stage()
    if var:
        st.GetPrimAtPath(var[0]).GetVariantSet(var[1]).SetVariantSelection(var[2])
        for _ in range(30):
            app.update()
    anns = []
    for n, e, t in views:
        cam = UsdGeom.Camera.Define(st, f"/World/Cam_{Tf.MakeValidIdentifier(n)}")
        cam.CreateFocalLengthAttr(14.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500))
        UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*e), Gf.Vec3d(*t), Gf.Vec3d(0, 0, 1)).GetInverse())
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        rp = rep.create.render_product(str(cam.GetPath()), (960, 540))
        a.attach([rp])
        anns.append((n, a, rp))
    for n, a, rp in anns:
        for _ in range(3):
            rep.orchestrator.step(rt_subframes=16)
        imgs[(src, n)] = Image.fromarray(np.asarray(a.get_data())[..., :3])
    for n, a, rp in anns:
        a.detach()
        rp.destroy()
for (src_, n_), im_ in imgs.items():
    im_.save(f"{USD}/DX_Hub_Building_no_obj_v1/validation/_raw_{tag}_{src_}_{Tf.MakeValidIdentifier(n_)}.png")
W = Image.new("RGB", (2 * 640, len(views) * 386), "white")
d = ImageDraw.Draw(W)
for r, (n, _, _) in enumerate(views):
    for c, src in enumerate([s_[0] for s_ in SOURCES]):
        W.paste(imgs[(src, n)].resize((640, 360)), (c * 640, r * 386 + 26))
        d.text((c * 640 + 8, r * 386 + 6), f"{n} - {src}", fill=(0, 0, 0))
os.makedirs(f"{USD}/DX_Hub_Building_no_obj_v1/validation", exist_ok=True)
W.save(f"{USD}/DX_Hub_Building_no_obj_v1/validation/views_{tag}.png")
app.close()
