"""Stage 12 (v3): load DX_Hub_Building_no_obj_v1 through the two Isaac Lab entry points and check physics + variants.

  --case terrain : TerrainImporterCfg(terrain_type="usd")             (default variants)
  --case asset   : AssetBaseCfg + UsdFileCfg(variants={doors: open, fixtures: none, lighting: night})

0.2 m cubes are dropped from 1.8 m onto the hall floor, the corridor floor and the south U counter; after 3 s the
cube centres must rest at 0.1 m (floor) or counter top + 0.1 m. With fixtures=none the counter cube falls to the
floor. Results -> DX_Hub_Building_no_obj_v1/validation/isaaclab_<case>.json
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["terrain", "asset"], required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

V3 = "/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_no_obj_v1"
ASSET = f"{V3}/DX_Hub_Building_no_obj_v1.usda"
VARIANTS = {"doors": "open", "fixtures": "none", "lighting": "night"}
DROPS = {"hall": (-8.0, 10.0), "corridor": (5.0, 0.0), "counter_U_south": (1.5, -0.4)}


def cube(x, y):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube_" + f"{int(x * 10)}_{int(y * 10)}".replace("-", "m"),
        spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                  mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                  collision_props=sim_utils.CollisionPropertiesCfg()),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, 1.8)))


@configclass
class SceneCfg(InteractiveSceneCfg):
    cube_hall = cube(*DROPS["hall"])
    cube_corridor = cube(*DROPS["corridor"])
    cube_counter = cube(*DROPS["counter_U_south"])


if args.case == "terrain":
    SceneCfg.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="usd", usd_path=ASSET)
    root = "/World/ground/terrain"
else:
    SceneCfg.building = AssetBaseCfg(prim_path="/World/DX_Hub_Building",
                                     spawn=sim_utils.UsdFileCfg(usd_path=ASSET, variants=VARIANTS))
    root = "/World/DX_Hub_Building"

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=0.0))
sim.reset()
for _ in range(360):
    scene.write_data_to_sim()
    sim.step()
    scene.update(1 / 120)

stage = omni.usd.get_context().get_stage()
rp = stage.GetPrimAtPath(root)
rest = {k: round(float(scene[f"cube_{k.split('_')[0]}"].data.root_pos_w.torch[0, 2]), 3) for k in DROPS}
n = {"mesh": 0, "collider": 0, "light": 0}
for p in Usd.PrimRange(rp):
    n["mesh"] += p.IsA(UsdGeom.Mesh)
    n["collider"] += p.HasAPI(UsdPhysics.CollisionAPI)
    n["light"] += p.GetTypeName().endswith("Light")
report = {"case": args.case, "root": root, "loaded": rp.IsValid(),
          "variants": {vs: rp.GetVariantSet(vs).GetVariantSelection() for vs in rp.GetVariantSets().GetNames()},
          "active_prims": n, "fixtures_prim_active": stage.GetPrimAtPath(f"{root}/Fixtures").IsActive(),
          "cube_rest_z": rest}
print(json.dumps(report, indent=1))
json.dump(report, open(f"{V3}/validation/isaaclab_{args.case}.json", "w"), indent=1)
app.close()
