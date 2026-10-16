"""Stage 25 (add_obj_v1): how the microwave push button works - simulated sequence + figure + time series.

The Microwave asset is an articulation with two DOFs:
  door   : revolute, hinge on the viewer's LEFT front edge, 0 .. -100 deg (negative = opening), damping only (no spring)
  button : prismatic, push direction into the body (-y local), 0 .. -6 mm, spring drive (stiffness 3000 N/m) holding it at 0
The latch itself is not physics. A small controller (customData 'latch' on the asset root) does what the real latch does:
  when button position < -press_to_open_m (3 mm) and the door is closed -> release: drive the door open (velocity target)

Sequence (GPU physics, 120 Hz): settle 1 s -> finger presses the button (drive target -6 mm) for 0.3 s -> controller
releases the door -> finger lets go (button springs back) -> door swings open.
Outputs (DX_Hub_Building_add_obj_v1/validation/): microwave_button_demo.png (4 states x 2 cameras),
microwave_button_timeseries.csv, microwave_button_demo.json
"""
import csv
import json
import math
import os

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.experimental.prims import Articulation  # noqa: E402
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

USD = "/home/cai/P2&DX_project/P2&DX/USD"
DST = f"{USD}/DX_Hub_Building_add_obj_v1"
VAL = f"{DST}/validation"
CFG = json.load(open(f"{USD}/config/add_obj_v1_furniture.json"))
MW = "/World/DX_Hub_Building_add_obj/Furniture/Kitchen/Microwave"
pl = CFG["placements"]["Kitchen"]["Microwave"]
c = CFG["assets"]["Microwave"]
W_, D_, H_ = c["size"]

ctx = omni.usd.get_context()
ctx.open_stage(f"{DST}/DX_Hub_Building_add_obj_v1_scene.usda")
for _ in range(40):
    app.update()
stage = ctx.get_stage()
stage.GetPrimAtPath("/World/DX_Hub_Building_add_obj/Building").GetVariantSet("lighting").SetVariantSelection("day")
for _ in range(20):
    app.update()


def world(lx, ly, lz):
    """Microwave-local point -> world (yaw about z, then translate)."""
    a = math.radians(pl["yaw"])
    return (pl["x"] + lx * math.cos(a) - ly * math.sin(a), pl["y"] + lx * math.sin(a) + ly * math.cos(a), pl["z"] + lz)


xc = -W_ / 2 + (W_ - c["door_w"]) / 2                                   # control column centre (viewer's right)
VIEWS = {"front": (world(0.15, 0.75, 0.3), world(0.0, 0.0, H_ / 2)),
         "button": (world(xc + 0.06, 0.32, 0.12), world(xc, D_ / 2, 0.045))}
anns = {}
for k, (eye, tgt) in VIEWS.items():
    cam = UsdGeom.Camera.Define(stage, f"/World/MwCam_{k}")
    cam.CreateFocalLengthAttr(18.0 if k == "front" else 30.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 50))
    UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*tgt), Gf.Vec3d(0, 0, 1)).GetInverse())
    a_ = rep.AnnotatorRegistry.get_annotator("rgb")
    a_.attach([rep.create.render_product(str(cam.GetPath()), (640, 480))])
    anns[k] = a_

SimulationManager.set_physics_sim_device("cuda:0")
tl = omni.timeline.get_timeline_interface()
tl.play()
for _ in range(5):
    app.update()
art = Articulation(MW)
names = list(art.dof_names)
bi, di = names.index("button"), names.index("door")
latch = stage.GetPrimAtPath(MW).GetCustomDataByKey("latch")
dt = 1 / 60
rows, snaps, released, t = [], {}, None, 0.0
PRESS = (1.0, 1.3)                                                   # finger on the button
SNAP = {"1_closed": 0.9, "2_pressed": 1.12, "3_opening": 1.45, "4_open": 3.2}


def capture(tag):
    """Render while the timeline runs (a paused GPU sim is not reflected in the render); keep the joint state of the frame."""
    rep.orchestrator.step(rt_subframes=2, pause_timeline=False)
    q_ = art.get_dof_positions().numpy().reshape(-1)
    snaps[tag] = {k: Image.fromarray(np.asarray(a_.get_data())[..., :3]) for k, a_ in anns.items()}
    snaps[tag]["state"] = f"button {q_[bi] * 1000:.1f} mm  door {math.degrees(q_[di]):.0f}°"


while t < 3.4:
    press = PRESS[0] <= t < PRESS[1]
    art.set_dof_position_targets(np.array([[-0.006 if press else 0.0]], np.float32), dof_indices=[bi])
    q = art.get_dof_positions().numpy().reshape(-1)
    if released is None and q[bi] < -latch["press_to_open_m"] and abs(q[di]) < math.radians(3):
        released = t                                                  # latch controller: release the door
    if released is not None:
        art.set_dof_velocity_targets(np.array([[2.5 * latch.get("door_open_sign", 1)]], np.float32), dof_indices=[di])
    app.update()
    t += dt
    q = art.get_dof_positions().numpy().reshape(-1)
    rows.append({"t_s": round(t, 4), "finger_pressing": int(press), "button_mm": round(float(q[bi]) * 1000, 3),
                 "door_deg": round(math.degrees(float(q[di])), 2), "latch_released": int(released is not None)})
    for tag, ts in SNAP.items():
        if tag not in snaps and t >= ts:
            capture(tag)
tl.stop()

with open(f"{VAL}/microwave_button_timeseries.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
btn_min = min(r["button_mm"] for r in rows)
summary = {"dofs": names, "press_window_s": PRESS, "latch_released_at_s": round(released, 3) if released else None,
           "button_min_mm": btn_min, "button_after_release_mm": rows[-1]["button_mm"], "door_final_deg": rows[-1]["door_deg"],
           "door_limit_deg": -c["open_deg"], "press_to_open_mm": latch["press_to_open_m"] * 1000}
json.dump(summary, open(f"{VAL}/microwave_button_demo.json", "w"), indent=1)
print(json.dumps(summary, indent=1))

# ------------------------------------------------------------------ figure: 4 states x 2 cameras with captions
cap = {"1_closed": "1. 닫힘 - 버튼 0 mm, 문 0°", "2_pressed": "2. 버튼 누름 (-6 mm) → 3 mm 넘으면 걸쇠 해제",
       "3_opening": "3. 손 뗌 → 버튼 스프링 복귀, 문이 열리는 중", "4_open": f"4. 열림 - 문 {rows[-1]['door_deg']:.0f}° (한계 -{c['open_deg']}°)"}
def state_at(ts):
    r = min(rows, key=lambda r: abs(r["t_s"] - ts))
    return f"t={r['t_s']:.2f}s  button {r['button_mm']:.1f} mm  door {r['door_deg']:.0f}°"
font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 20, index=1)
small = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 16, index=1)
S = Image.new("RGB", (4 * 640, 2 * 480 + 80), "white")
d = ImageDraw.Draw(S)
for i, tag in enumerate(SNAP):
    d.text((i * 640 + 10, 8), cap[tag], font=font, fill=(0, 0, 0))
    d.text((i * 640 + 10, 40), f"t≈{SNAP[tag]:.2f}s  " + snaps[tag]["state"], font=small, fill=(90, 90, 90))
    for j, k in enumerate(("front", "button")):
        S.paste(snaps[tag][k], (i * 640, 80 + j * 480))
    snaps[tag].pop("state")
bx, by = 640 * 1 + 320, 80 + 480 + 240                                # mark the button in the pressed close-up
d.ellipse((bx - 70, by - 50, bx + 70, by + 50), outline=(230, 40, 40), width=4)
S.save(f"{VAL}/microwave_button_demo.png")
print("wrote microwave_button_demo.png")
app.close()
