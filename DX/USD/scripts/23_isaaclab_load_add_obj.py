"""Stage 23 (add_obj_v1): load DX_Hub_Building_add_obj_v1 in Isaac Lab (AssetBaseCfg + UsdFileCfg, GPU sim) and run 3 s.

Checks that the deformable pillow / blanket and the articulated pieces load and simulate under Isaac Lab, and that a
0.2 m cube dropped on the mattress, on the sofa table and into the kitchen sink comes to rest on them.
Result -> DX_Hub_Building_add_obj_v1/validation/isaaclab_add_obj.json
"""
import argparse
import json

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import omni.usd  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

DST = "/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_add_obj_v1"
CFG = json.load(open("/home/cai/P2&DX_project/P2&DX/USD/config/add_obj_v1_furniture.json"))
bed, tab = CFG["placements"]["CareRoom"]["Bed"], CFG["placements"]["Hall"]["SofaTable_C"]      # hall sofa table (care-room one removed)
sink = CFG["placements"]["Kitchen"]["Sink_1"]


def cube(x, y, z):
    return RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Cube_" + f"{int(x * 10)}_{int(y * 10)}".replace("-", "m"),
                          spawn=sim_utils.CuboidCfg(size=(0.2, 0.2, 0.2), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                                                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                                                    collision_props=sim_utils.CollisionPropertiesCfg()),
                          init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, z)))


@configclass
class SceneCfg(InteractiveSceneCfg):
    world = AssetBaseCfg(prim_path="/World/DX_Hub_Building_add_obj",
                         spawn=sim_utils.UsdFileCfg(usd_path=f"{DST}/DX_Hub_Building_add_obj_v1.usda"))
    cube_bed = cube(bed["x"], bed["y"] - 0.5, 1.2)
    cube_table = cube(tab["x"], tab["y"], 1.0)
    cube_sink = cube(sink["x"], sink["y"], 1.5)


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1 / 120, device="cuda:0"))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=0.0))
sim.reset()
for _ in range(360):
    scene.write_data_to_sim()
    sim.step()
    scene.update(1 / 120)
stage = omni.usd.get_context().get_stage()
root = stage.GetPrimAtPath("/World/DX_Hub_Building_add_obj")
n = {"rigid_bodies": 0, "joints": 0, "deformables": 0}
for p in Usd.PrimRange(root):
    n["rigid_bodies"] += p.HasAPI(UsdPhysics.RigidBodyAPI)
    n["joints"] += p.IsA(UsdPhysics.Joint)
    n["deformables"] += "OmniPhysicsDeformableBodyAPI" in p.GetAppliedSchemas()
rep = {"loaded": root.IsValid(), "counts": n, "device": str(sim.device),
       "cube_on_mattress_z": round(float(scene["cube_bed"].data.root_pos_w.torch[0, 2]), 3),
       "cube_on_table_z": round(float(scene["cube_table"].data.root_pos_w.torch[0, 2]), 3),
       "cube_in_sink_z": round(float(scene["cube_sink"].data.root_pos_w.torch[0, 2]), 3),
       "expected": {"mattress_collider_top+0.1": round(CFG["assets"]["Bed"]["base_top_z"] + CFG["assets"]["Bed"]["mattress"]["size"][2]
                                                    + CFG["assets"]["Bed"]["mattress"]["collider_lift"] + 0.1, 3),
                    "table_top+0.1": round(CFG["assets"]["SofaTable"]["size"][2] + 0.1, 3),
                    "sink_bowl_bottom+0.1": round(sink["z"] - CFG["assets"]["Sink"]["depth_inner"] + 0.1, 3)}}
print(json.dumps(rep, indent=1))
json.dump(rep, open(f"{DST}/validation/isaaclab_add_obj.json", "w"), indent=1)
app.close()
