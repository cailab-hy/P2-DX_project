"""Nav2 occupancy map from a DX Hub USD at the virtual LiDAR's plane, by the same PhysX scene the /scan rays hit.

The old dx_hub map came from SLAM and bends up to 0.4 m against the world, so new furniture can't simply be drawn on it. This keeps its grid (origin,
resolution, size) and the start spot's map coordinates, with one rigid world -> map transform:
  map = R(YAW) (world - SPAWN_WORLD) + SPAWN_MAP
Occupied = a collider at z = HEIGHT in the cell. Free = reachable from the spawn over unoccupied cells; the rest is unknown.
    python rasterize_map.py STAGE.usda OUT_PREFIX   ->  OUT_PREFIX.pgm, .yaml, .npz
"""
import sys
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, UsdPhysics  # noqa: E402
from scipy.ndimage import label  # noqa: E402

stage_path, out = sys.argv[1:3]
RES, OX, OY, W, H = 0.05, -6.628, -31.947, 547, 867
HEIGHT = 0.25
YAW = -0.0611
SPAWN_WORLD, SPAWN_MAP = np.array((-3.527, 13.406)), np.array((4.32, 3.19))
c, s = np.cos(YAW), np.sin(YAW)

omni.usd.get_context().open_stage(stage_path)
for _ in range(40):
    app.update()
st = omni.usd.get_context().get_stage()
st.SetEditTarget(st.GetSessionLayer())
for prim in st.Traverse():      # the bridge's default (--doors open): the robot passes through inner doors
    if prim.GetVariantSets().HasVariantSet('doors'):
        prim.GetVariantSets().GetVariantSet('doors').SetVariantSelection('open')
        print('[map] doors open', prim.GetPath(), flush=True)
        break
for _ in range(10):
    app.update()
n = 0
for prim in st.Traverse():      # like the bridge: furniture must not move while the map is drawn
    if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not prim.IsInstanceProxy():
        UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
        n += 1
print(f'[map] {n} rigid bodies kinematic', flush=True)
sc = UsdPhysics.Scene.Define(st, '/MapScene')
sc.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
sc.CreateGravityMagnitudeAttr(9.81)
a = PhysxSchema.PhysxSceneAPI.Apply(sc.GetPrim())
a.CreateEnableGPUDynamicsAttr(False)
a.CreateBroadphaseTypeAttr('MBP')
tl = omni.timeline.get_timeline_interface()
tl.play()
for _ in range(3):
    app.update()
q = omni.physx.get_physx_scene_query_interface()
ext, rot = carb.Float3(RES / 2, RES / 2, 0.02), carb.Float4(0, 0, 0, 1)
occ = np.zeros((H, W), bool)
t0 = time.time()
for r in range(H):
    my = OY + (H - 1 - r + 0.5) * RES
    for col in range(W):
        mx = OX + (col + 0.5) * RES
        dx, dy = mx - SPAWN_MAP[0], my - SPAWN_MAP[1]
        wx, wy = SPAWN_WORLD[0] + c * dx + s * dy, SPAWN_WORLD[1] - s * dx + c * dy
        occ[r, col] = q.overlap_box(ext, carb.Float3(float(wx), float(wy), HEIGHT), rot, None, True) > 0
    if r % 100 == 0:
        print(f'[map] row {r}/{H}, {time.time() - t0:.0f} s, occupied {occ.sum()}', flush=True)
tl.stop()
seed = ((SPAWN_MAP - (OX, OY)) / RES).astype(int)
sr, sc_ = H - 1 - seed[1], seed[0]
lab, _ = label(~occ)
print('[map] spawn cell occupied', occ[sr, sc_])
near = lab[sr - 8:sr + 9, sc_ - 8:sc_ + 9]
k = np.bincount(near[near > 0]).argmax()
free = lab == k
pix = np.full((H, W), 205, np.uint8)
pix[free] = 254
pix[occ] = 0
open(f'{out}.pgm', 'wb').write(f'P5\n{W} {H}\n255\n'.encode() + pix.tobytes())
name = out.rsplit('/', 1)[-1]
open(f'{out}.yaml', 'w').write(f'image: {name}.pgm\nmode: trinary\nresolution: {RES:.3f}\norigin: [{OX}, {OY}, 0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.196\n')
np.savez_compressed(f'{out}.npz', occ=occ, free=free)
print(f'[map] done {time.time() - t0:.0f} s occupied {occ.sum()} free {free.sum()}', flush=True)
app.close()
