"""Stage 8b (v3): asset root with variant sets + ready-to-open scene.

DX_Hub_Building_no_obj_v1.usda  (defaultPrim /DX_Hub_Building, payload ./payloads/building.usdc)
  variantSet doors    : closed (default) | open   - open removes leaves (and their colliders) of doors that join
                                                      two scanned spaces; doors to private / unscanned rooms stay
  variantSet blinds   : down (default) | half | up
  variantSet fixtures : counters (default) | none
  variantSet lighting : day (default: overcast sky HDRI + interior lights) | day_sunny (partly cloudy + sun)
                        | night (interior lights only) | none
DX_Hub_Building_no_obj_v1_scene.usda: PhysicsScene + the asset (all lights are inside the asset).
"""
import json

from pxr import Gf, Kind, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSemantics  # noqa

from common import OUT

DST = OUT.parent / "DX_Hub_Building_no_obj_v1"
CFG = OUT.parent / "config"
ZONES = json.load(open(CFG / "dx_hub_no_obj_v1_zones.json"))
payload = Usd.Stage.Open(str(DST / "payloads" / "building.usdc"))
R = "/DX_Hub_Building"


def cls_of(p):
    api = UsdSemantics.LabelsAPI(p, "class")
    v = api.GetLabelsAttr().Get() if api.GetLabelsAttr() else None
    return list(v)[0] if v else None


openable_leaves, openable_colliders, blinds, counter_colliders = [], [], {"Down": [], "Half": [], "Up": []}, []
for p in payload.Traverse():
    path = str(p.GetPath())
    if p.GetTypeName() == "Mesh" and cls_of(p) == "door" and not p.GetCustomDataByKey("always_closed"):
        openable_leaves.append(path)
    for st in blinds:
        if p.GetName() == f"Blind_{st}":
            blinds[st].append(path)
    if path.startswith(f"{R}/Collision/") and cls_of(p) == "counter":
        counter_colliders.append(path)
# colliders of openable leaves: boxes named door_<k>_<q>_* (opaque doors) and glass_<line>_<i>_* (glass doors)
leaf_keys = set()
for lp in openable_leaves:
    parts = lp.split("/")
    if "Doors" in parts:
        k = int(parts[parts.index("Doors") + 1].split("_")[1])
        q = parts[-1].replace("Leaf", "") or "0"
        leaf_keys.add(f"door_{k}_{q}_")
    else:
        line = parts[parts.index("Glass") + 1]
        i = int(parts[-1].split("_")[1])
        leaf_keys.add(f"glass_{line}_{i}_")
for p in payload.GetPrimAtPath(f"{R}/Collision").GetChildren():
    if any(p.GetName().startswith(k) for k in leaf_keys):
        openable_colliders.append(str(p.GetPath()))
print(f"openable leaves {len(openable_leaves)} (colliders {len(openable_colliders)}), blinds {len(blinds['Down'])}, "
      f"counter colliders {len(counter_colliders)}")

stage = Usd.Stage.CreateNew(str(DST / "DX_Hub_Building_no_obj_v1.usda"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
layer = stage.GetRootLayer()
layer.documentation = ("DX Hub 11F (CARE robot vocational training centre) - empty building no_obj_v1: floor, walls, ceiling, "
                       "windows, doors, glass partitions, built-in counters. Measured from the 2026-06-22 LiDAR scan.")
plan = json.load(open(CFG / "dx_hub_no_obj_v1_plan.json"))
layer.customLayerData = {"frame": "Z-up, metres, floor z=0, walls along X/Y (same frame as DX_Hub / DX_Hub_Building v2)",
                         "T_las_to_world": str(plan["frame"]["T_las_to_world"]),
                         "T_gsplat_to_world": str(plan["frame"]["T_gsplat_to_world"])}
root = UsdGeom.Xform.Define(stage, R)
stage.SetDefaultPrim(root.GetPrim())
Usd.ModelAPI(root).SetKind(Kind.Tokens.assembly)
root.GetPrim().GetPayloads().AddPayload("./payloads/building.usdc")
root.GetPrim().SetAssetInfo({"name": "DX_Hub_Building", "version": "3.0"})
vsets = root.GetPrim().GetVariantSets()


def variant(set_name, choices, default, edits):
    vs = vsets.AddVariantSet(set_name)
    for c in choices:
        vs.AddVariant(c)
        vs.SetVariantSelection(c)
        with vs.GetVariantEditContext():
            for path in edits.get(c, []):
                stage.OverridePrim(path).SetActive(False)
    vs.SetVariantSelection(default)


variant("doors", ["closed", "open"], "closed", {"open": openable_leaves + openable_colliders})
variant("blinds", ["down", "half", "up"], "down",
        {"down": blinds["Half"] + blinds["Up"], "half": blinds["Down"] + blinds["Up"], "up": blinds["Down"] + blinds["Half"]})
variant("fixtures", ["counters", "none"], "counters", {"none": [f"{R}/Fixtures"] + counter_colliders})
# daylight lives in the asset (sky dome with HDRI, optional sun); textures next to this layer
DL = ZONES["daylight"]
UsdGeom.Xform.Define(stage, f"{R}/Lights/Daylight")
for name_, spec in (("Overcast", DL["overcast"]), ("Sunny", DL["sunny"])):
    base_ = f"{R}/Lights/Daylight/{name_}"
    UsdGeom.Xform.Define(stage, base_)
    dome = UsdLux.DomeLight.Define(stage, f"{base_}/Sky")
    dome.CreateTextureFileAttr().Set(f"./textures/{spec['hdri']}")
    dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
    dome.CreateIntensityAttr(spec["dome_intensity"])
    if "sun_intensity" in spec:
        sun = UsdLux.DistantLight.Define(stage, f"{base_}/Sun")
        sun.CreateIntensityAttr(spec["sun_intensity"])
        sun.CreateAngleAttr(0.53)
        sun.CreateColorTemperatureAttr(spec["sun_cct"])
        sun.CreateEnableColorTemperatureAttr(True)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(*spec["sun_rotate_xyz"]))
interior = [f"{R}/Lights/Panels", f"{R}/Lights/Coves", f"{R}/Lights/Downlights"]
variant("lighting", ["day", "day_sunny", "night", "none"], "day", {
    "day": [f"{R}/Lights/Daylight/Sunny"],
    "day_sunny": [f"{R}/Lights/Daylight/Overcast"],
    "night": [f"{R}/Lights/Daylight"],
    "none": [f"{R}/Lights/Daylight"] + interior})
layer.Save()

scene = Usd.Stage.CreateNew(str(DST / "DX_Hub_Building_no_obj_v1_scene.usda"))
UsdGeom.SetStageUpAxis(scene, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(scene, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(scene, 1.0)
world = UsdGeom.Xform.Define(scene, "/World")
scene.SetDefaultPrim(world.GetPrim())
scene.DefinePrim("/World/DX_Hub_Building").GetReferences().AddReference("./DX_Hub_Building_no_obj_v1.usda")
ps = UsdPhysics.Scene.Define(scene, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(9.81)
scene.GetRootLayer().Save()
print("wrote", DST / "DX_Hub_Building_no_obj_v1.usda", "and the scene")
