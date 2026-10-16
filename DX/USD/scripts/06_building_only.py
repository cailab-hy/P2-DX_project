"""Stage 6: export a building-only USD: floor, walls, glass walls, ceiling (+ light panels) and the
two U-shaped tables in the middle of the hall.

Everything else is left out: scanned furniture, robots, desks, the Gaussian splats and the other
extruded fixtures (booths, sills, clutter). The result is a small, self-contained asset in
USD/DX_Hub_Building/.
"""
import json
import shutil

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from common import OUT, WORK

DST = OUT.parent / "DX_Hub_Building"
(DST / "textures").mkdir(parents=True, exist_ok=True)
shutil.copy(OUT / "payloads/textures/floor_tiles.png", DST / "textures/floor_tiles.png")
KEEP_FIXTURE_PREFIX = "UTable"          # the two U-shaped tables in the middle of the hall stay
fr = json.load(open(WORK / "frame.json"))

src = Sdf.Layer.FindOrOpen(str(OUT / "payloads/structure.usdc"))
lights_src = Sdf.Layer.FindOrOpen(str(OUT / "DX_Hub.usda"))
stage = Usd.Stage.CreateNew(str(DST / "DX_Hub_Building.usda"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
layer = stage.GetRootLayer()
layer.documentation = ("DX Hub building shell (floor, walls, glass walls, ceiling, ceiling lights) + the two "
                       "U-shaped tables - no other furniture/objects")
layer.customLayerData = {
    "source": "DX_Hub v2.0 clean structure (03_structure.py)",
    "frame": "Z-up, meters. Floor plane z=0, walls aligned to X/Y, origin = centre of floor footprint",
    "T_las_to_world": str(np.array(fr["T_las_to_world"]).round(6).tolist()),
}
root = UsdGeom.Xform.Define(stage, "/DX_Hub_Building")
stage.SetDefaultPrim(root.GetPrim())
UsdGeom.Xform.Define(stage, "/DX_Hub_Building/Visual")
UsdGeom.Scope.Define(stage, "/DX_Hub_Building/Looks")

# Copy the whole clean structure, re-point every connection/binding to the new location, then drop
# all fixtures except the U-shaped tables (and their materials).
OLD, NEW = Sdf.Path("/Structure"), Sdf.Path("/DX_Hub_Building/Visual")
Sdf.CopySpec(src, OLD, layer, NEW)
for prim in Usd.PrimRange(stage.GetPrimAtPath(NEW)):
    for attr in prim.GetAttributes():
        if attr.HasAuthoredConnections():
            attr.SetConnections([c.ReplacePrefix(OLD, NEW) for c in attr.GetConnections()])
    for rel in prim.GetRelationships():
        if rel.GetTargets():
            rel.SetTargets([t.ReplacePrefix(OLD, NEW) for t in rel.GetTargets()])
for name in [p.GetName() for p in stage.GetPrimAtPath(NEW.AppendChild("Fixtures")).GetChildren()]:
    if not name.startswith(KEEP_FIXTURE_PREFIX):
        stage.RemovePrim(NEW.AppendPath(f"Fixtures/{name}"))
        stage.RemovePrim(NEW.AppendPath(f"Looks/{name}"))

# Collision: the shell meshes merged into one static triangle mesh (light panels excluded).
pts, idx, off = [], [], 0
col_paths = ["Floor", "Walls", "Ceiling", "Glass/Clear", "Glass/Frosted", "Glass/Frames"] + \
    [f"Fixtures/{p.GetName()}" for p in stage.GetPrimAtPath(NEW.AppendChild("Fixtures")).GetChildren()]
for name in col_paths:
    m = UsdGeom.Mesh(stage.GetPrimAtPath(NEW.AppendPath(name)))
    p = np.asarray(m.GetPointsAttr().Get())
    pts.append(p)
    idx.append(np.asarray(m.GetFaceVertexIndicesAttr().Get()) + off)
    off += len(p)
P, F = np.vstack(pts), np.concatenate(idx)
col = UsdGeom.Mesh.Define(stage, "/DX_Hub_Building/Collision")
col.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(P.astype(np.float32)))
col.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F) // 3, 3, np.int32)))
col.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.astype(np.int32)))
col.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*P.min(0).tolist()), Gf.Vec3f(*P.max(0).tolist())]))
col.CreatePurposeAttr(UsdGeom.Tokens.guide)
UsdPhysics.CollisionAPI.Apply(col.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
pm = UsdShade.Material.Define(stage, "/DX_Hub_Building/Looks/PhysicsMaterial")
pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
pma.CreateStaticFrictionAttr(0.8)
pma.CreateDynamicFrictionAttr(0.6)
pma.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(col.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")

# Ceiling lights (same RectLights as the full asset).
Sdf.CopySpec(lights_src, "/DX_Hub/Lights", layer, "/DX_Hub_Building/Lights")
layer.Save()
n_lights = len(stage.GetPrimAtPath("/DX_Hub_Building/Lights").GetChildren())
print(f"building: {len(F) // 3:,} collision tris, {n_lights} lights, kept fixtures: "
      f"{[p.GetName() for p in stage.GetPrimAtPath(NEW.AppendChild('Fixtures')).GetChildren()]}")

# Ready-to-open scene.
scene = Usd.Stage.CreateNew(str(DST / "DX_Hub_Building_scene.usda"))
UsdGeom.SetStageUpAxis(scene, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(scene, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(scene, 1.0)
world = UsdGeom.Xform.Define(scene, "/World")
scene.SetDefaultPrim(world.GetPrim())
scene.DefinePrim("/World/DX_Hub_Building").GetReferences().AddReference("./DX_Hub_Building.usda")
ps = UsdPhysics.Scene.Define(scene, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(9.81)
UsdGeom.Xform.Define(scene, "/World/Lights")
UsdLux.DomeLight.Define(scene, "/World/Lights/Dome").CreateIntensityAttr(800)
scene.GetRootLayer().Save()
