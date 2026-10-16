"""Stage 22 (add_obj_v1): Isaac Sim 6.1 check of the placed furniture (GPU physics).

  * settle 4 s: no rigid furniture body may drift (every RigidBodyAPI prim under /Furniture is checked); pillow /
    blanket nodal positions finite, pillow on the mattress, blanket draped
  * articulations: every DOF of every articulated piece is opened (doors 90 deg, drawers / sliding doors to 85 % of
    travel) and must hold for 1 s; wall-anchored bodies must not move. Microwave: the push button is pressed through
    its spring drive and a latch controller (README) opens the door once the button is in by > press_to_open_m
  * renders at the photo / video viewpoints of config validation_views -> validation/add_obj_vs_photos_<room>.png
Results -> DX_Hub_Building_add_obj_v1/validation/add_obj_checks.json
"""
import json
import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import Articulation, DeformablePrim  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

USD = "/home/cai/P2&DX_project/P2&DX/USD"
DST = f"{USD}/DX_Hub_Building_add_obj_v1"
VAL = f"{DST}/validation"
WORK = f"{USD}/work/add_obj_v1"
os.makedirs(VAL, exist_ok=True)
CFG = json.load(open(f"{USD}/config/add_obj_v1_furniture.json"))
FU = "/World/DX_Hub_Building_add_obj/Furniture"
ctx = omni.usd.get_context()
ctx.open_stage(f"{DST}/DX_Hub_Building_add_obj_v1_scene.usda")
for _ in range(40):
    app.update()
stage = ctx.get_stage()
stage.GetPrimAtPath("/World/DX_Hub_Building_add_obj/Building").GetVariantSet("lighting").SetVariantSelection("day")
for _ in range(20):
    app.update()

VIEWS = {k: v for k, v in CFG["validation_views"].items() if not k.startswith("_")}
anns = {}
for k, v in VIEWS.items():
    cam = UsdGeom.Camera.Define(stage, f"/World/Cam_{k}")
    cam.CreateFocalLengthAttr(13.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * 3 / 4)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.03, 200))
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*v["eye"]), Gf.Vec3d(*v["target"]), Gf.Vec3d(0, 0, 1)).GetInverse())
    a = rep.AnnotatorRegistry.get_annotator("rgb")
    rp = rep.create.render_product(str(cam.GetPath()), (800, 600))
    a.attach([rp])
    anns[k] = a


def render(tag):
    out = {}
    for _ in range(3):
        rep.orchestrator.step(rt_subframes=16, pause_timeline=False)
    for k, a in anns.items():
        out[k] = Image.fromarray(np.asarray(a.get_data())[..., :3])
        out[k].save(f"{VAL}/_raw_{tag}_{k}.png")
    return out


bodies = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(FU)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
arts = [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(FU)) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
SimulationManager.set_physics_sim_device("cuda:0")
tl = omni.timeline.get_timeline_interface()
px = omni.physx.get_physx_interface()
tl.play()
for _ in range(5):
    app.update()
p0 = {b: np.array(px.get_rigidbody_transformation(b)["position"]) for b in bodies}
for _ in range(240):
    app.update()
p1 = {b: np.array(px.get_rigidbody_transformation(b)["position"]) for b in bodies}
drift = {b.replace(FU + "/", ""): round(float(np.linalg.norm(p1[b] - p0[b])), 4) for b in bodies}
rep_ = {"rigid_bodies": len(bodies), "max_drift_m": max(drift.values()), "drift_over_5mm": {k: v for k, v in drift.items() if v > 0.005}}
bed = CFG["placements"]["CareRoom"]["Bed"]
mat_top = CFG["assets"]["Bed"]["base_top_z"] + CFG["assets"]["Bed"]["mattress"]["size"][2]
for name, path in (("pillow", f"{FU}/CareRoom/Bed/Pillow"), ("blanket", f"{FU}/CareRoom/Bed/Blanket")):
    pos = DeformablePrim(path).get_nodal_positions()[0].numpy().reshape(-1, 3)
    on_top = (np.abs(pos[:, 0] - bed["x"]) < 0.3) & (np.abs(pos[:, 1] - bed["y"]) < 0.6)
    rep_[name] = {"finite": bool(np.isfinite(pos).all()), "z_min": round(float(pos[:, 2].min()), 3), "z_max": round(float(pos[:, 2].max()), 3),
                  "z_median_over_mattress_centre": round(float(np.median(pos[on_top, 2])), 4) if on_top.any() else None, "mattress_top_z": mat_top}
ts = CFG["placements"]["CareRoom"].get("TShirt")
if ts:                                                                # T-shirt on the care-room desk
    pos = DeformablePrim(f"{FU}/CareRoom/TShirt/Shirt").get_nodal_positions()[0].numpy().reshape(-1, 3)
    dk = CFG["placements"]["CareRoom"]["CareDesk"]
    top = CFG["assets"]["CareDesk"]["size"][2]
    dw, dd = CFG["assets"]["CareDesk"]["size"][:2]                     # desk along the wall (world y) x depth (world x)
    on_desk = (np.abs(pos[:, 0] - dk["x"]) < dd / 2 - 0.02) & (np.abs(pos[:, 1] - dk["y"]) < dw / 2 - 0.02)
    rep_["tshirt"] = {"finite": bool(np.isfinite(pos).all()), "z_min": round(float(pos[:, 2].min()), 3), "z_max": round(float(pos[:, 2].max()), 3),
                      "share_over_desk": round(float(on_desk.mean()), 3), "z_median_over_desk": round(float(np.median(pos[on_desk, 2])), 4),
                      "desk_top_z": top, "xy_extent": [round(float(np.ptp(pos[:, 0])), 3), round(float(np.ptp(pos[:, 1])), 3)]}
settled = render("settled")

# ------------------------------------------------------------------ articulations
rep_["articulations"] = {}
handles = {}
for path in arts:
    art = Articulation(path)
    names = list(art.dof_names)
    jp = {str(p.GetName()): p for p in Usd.PrimRange(stage.GetPrimAtPath(path)) if p.IsA(UsdPhysics.Joint)}
    tgt = []
    for d in names:
        j = jp[d]
        lo, hi = j.GetAttribute("physics:lowerLimit").Get(), j.GetAttribute("physics:upperLimit").Get()
        if d == "button":
            tgt.append(0.0)
        elif j.IsA(UsdPhysics.RevoluteJoint):
            tgt.append(math.radians(90.0) * (1 if hi > 0 else -1))
        else:
            tgt.append(0.85 * (hi if abs(hi) > abs(lo) else lo))
    tgt = np.array([tgt], np.float32)
    handles[path] = [art, names, tgt]
    if "button" in names:                                             # microwave: press the button through its spring drive
        art.set_dof_position_targets(np.array([[-0.006]], np.float32), dof_indices=[names.index("button")])
    else:
        art.set_dof_positions(tgt)
for _ in range(30):
    app.update()
for path, h in handles.items():                                       # latch controller: button in -> open the door
    art, names = h[0], h[1]
    if "button" in names:
        q = art.get_dof_positions().numpy().reshape(-1)
        latch = stage.GetPrimAtPath(path).GetCustomDataByKey("latch")
        pressed = float(q[names.index("button")])
        if pressed < -latch["press_to_open_m"]:
            art.set_dof_positions(np.array([[math.radians(90.0) * latch.get("door_open_sign", 1)]], np.float32), dof_indices=[names.index("door")])
        art.set_dof_position_targets(np.array([[0.0]], np.float32), dof_indices=[names.index("button")])
        h.append(pressed)
for _ in range(60):
    app.update()
for path, h in handles.items():
    art, names, tgt = h[:3]
    q = art.get_dof_positions().numpy().reshape(-1)
    body = f"{path}/Body"
    r = {"dofs": names, "target": [round(float(v), 3) for v in tgt[0]], "after_1s": [round(float(v), 3) for v in q],
         "body_moved_m": round(float(np.linalg.norm(np.array(px.get_rigidbody_transformation(body)["position"]) - p0[body]))
                               if body in p0 else 0.0, 4)}
    if len(h) == 4:
        r["button_pressed_m"] = round(h[3], 4)
        r["door_after_press_rad"] = round(float(q[names.index("door")]), 3)
        r["button_after_release_m"] = round(float(q[names.index("button")]), 4)
    rep_["articulations"][path.replace(FU + "/", "")] = r
opened = render("open")
tl.stop()
print(json.dumps(rep_, indent=1))
json.dump(rep_, open(f"{VAL}/add_obj_checks.json", "w"), indent=1)

# ------------------------------------------------------------------ comparison sheets per room
groups = {"care_room": [k for k in VIEWS if (VIEWS[k]["ref"] or "ref/").startswith("ref/")],
          "kitchen": [k for k in VIEWS if (VIEWS[k]["ref"] or "").startswith("kitchen_ref/")],
          "hall": [k for k in VIEWS if (VIEWS[k]["ref"] or "").startswith("hall_ref/")],
          "coffee_sink": [k for k in VIEWS if (VIEWS[k]["ref"] or "").startswith("coffee_ref/")]}
for g, keys in groups.items():
    rows = []
    for k in keys:
        im = (ImageOps.fit(ImageOps.exif_transpose(Image.open(f"{WORK}/{VIEWS[k]['ref']}")).convert("RGB"), (800, 600))
              if VIEWS[k]["ref"] else Image.new("RGB", (800, 600), (225, 225, 225)))    # no photo for this view
        W = Image.new("RGB", (2400, 626), "white")
        W.paste(im, (0, 26))
        W.paste(settled[k], (800, 26))
        W.paste(opened[k], (1600, 26))
        d = ImageDraw.Draw(W)
        d.text((8, 6), f"{k} - real", fill=(0, 0, 0))
        d.text((808, 6), f"{k} - USD (settled 4 s)", fill=(0, 0, 0))
        d.text((1608, 6), f"{k} - USD (doors / drawers opened)", fill=(0, 0, 0))
        rows.append(W)
    S = Image.new("RGB", (2400, 626 * len(rows)), "white")
    for i, r_ in enumerate(rows):
        S.paste(r_, (0, 626 * i))
    S.save(f"{VAL}/add_obj_vs_photos_{g}.png")
app.close()
