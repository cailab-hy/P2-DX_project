"""Open a stage in its own Isaac Sim window (view only, no simulation) next to the running bridge.

Every second the viewer writes STATE_JSON: the active camera's world position / looking direction and the selected
prims with their world positions, so a spot the user looks at or clicks can be found without asking for coordinates.
    python view_stage.py STAGE.usda [camera_prim] [STATE_JSON]
"""
import json
import sys
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": False, "width": 1600, "height": 900, "renderer": "RaytracedLighting"})

import omni.usd  # noqa: E402
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

stage_path = sys.argv[1]
state_path = sys.argv[3] if len(sys.argv) > 3 else None
ctx = omni.usd.get_context()
ctx.open_stage(stage_path)
for _ in range(30):
    app.update()
if len(sys.argv) > 2 and sys.argv[2]:
    get_active_viewport().camera_path = sys.argv[2]
print(f"[view] {stage_path} open; close the window to quit", flush=True)


def world(prim):
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return m


last = 0.0
while app.is_running():
    app.update()
    if state_path and time.time() - last > 1.0:
        last = time.time()
        try:
            stage = ctx.get_stage()
            cam_path = str(get_active_viewport().camera_path)
            m = world(stage.GetPrimAtPath(cam_path))
            pos = m.ExtractTranslation()
            fwd = m.TransformDir((0, 0, -1))
            sel = []
            for p in ctx.get_selection().get_selected_prim_paths():
                prim = stage.GetPrimAtPath(p)
                if prim.IsValid() and prim.IsA(UsdGeom.Xformable):
                    t = world(prim).ExtractTranslation()
                    sel.append({"path": p, "pos": [round(t[0], 3), round(t[1], 3), round(t[2], 3)]})
            json.dump({"time": time.strftime("%H:%M:%S"), "camera": cam_path, "camera_pos": [round(v, 3) for v in pos],
                       "camera_dir": [round(v, 3) for v in fwd], "selected": sel}, open(state_path, "w"), indent=1)
        except Exception as exc:  # keep the viewer alive
            print(f"[view] state: {exc}", flush=True)
app.close()
