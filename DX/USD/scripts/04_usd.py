"""Stage 4: assemble the Isaac Sim USD asset for the DX Hub.

Outputs (in OUT):
  DX_Hub.usda                    asset root (defaultPrim /DX_Hub, Z-up, meters) with "appearance" and "objects" variant sets
  DX_Hub_scene.usda              ready-to-open stage: /World/DX_Hub + PhysicsScene + lights
  payloads/structure.usdc        clean architectural model (written by 03_structure.py)
  payloads/objects.usdc          scanned furniture (written by 03_structure.py)
  payloads/visual_mesh.usdc      raw coloured scan mesh (Structure + Ceiling), UsdPreviewSurface reading displayColor
  payloads/collision_mesh.usdc   collider: clean shell + simplified furniture (purpose=guide)
  payloads/gaussian_splat.usdc   ParticleField3DGaussianSplat (native splat frame, cropped to the interior)
"""
import json
import sys

import numpy as np
import open3d as o3d
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

from common import GSPLAT, OUT, WORK, apply, read_gs_ply, write_gs_ply

sys.path.insert(0, "/home/cai/miniforge3/envs/env_isaaclab/lib/python3.12/site-packages/isaacsim/extscache/"
                   "omni.kit.converter.gsplat-0.1.16+110.1.0.lx64.r.cp312.u7f4/pip_prebundle")
from usd_convert_gsplat import read_ply, write_gaussian_splat_usd  # noqa: E402  (NVIDIA converter shipped with Isaac Sim)

fr = json.load(open(WORK / "frame.json"))
stats = json.load(open(WORK / "mesh_stats.json"))
T_gs2w = np.array(fr["T_gsplat_to_world"])
PAY = OUT / "payloads"
PAY.mkdir(parents=True, exist_ok=True)


def new_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    return stage


def define_mesh(stage, path, V, F, N=None, C=None):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.astype(np.int32).ravel()))
    m.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())]))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    if N is not None:
        m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(N.astype(np.float32)))
        m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if C is not None:
        pv = m.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(C.astype(np.float32)))
    return m


def submesh(V, F, sel):
    F = F[sel]
    used = np.unique(F)
    remap = np.full(len(V), -1, np.int64)
    remap[used] = np.arange(len(used))
    return used, remap[F]


# ---------------------------------------------------------------- visual mesh
vis = o3d.io.read_triangle_mesh(str(WORK / "mesh_visual.ply"))
V, F = np.asarray(vis.vertices), np.asarray(vis.triangles)
vis.compute_triangle_normals()
FN = np.asarray(vis.triangle_normals)
N, C = np.asarray(vis.vertex_normals), np.asarray(vis.vertex_colors)
ceiling = (FN[:, 2] < -0.6) & (V[F].mean(1)[:, 2] > 1.9)    # downward-facing surfaces up high
stage = new_stage(PAY / "visual_mesh.usdc")
root = UsdGeom.Xform.Define(stage, "/ScanMesh")
stage.SetDefaultPrim(root.GetPrim())
mat = UsdShade.Material.Define(stage, "/ScanMesh/Looks/ScanVertexColor")
sh = UsdShade.Shader.Define(stage, "/ScanMesh/Looks/ScanVertexColor/PreviewSurface")
sh.CreateIdAttr("UsdPreviewSurface")
rd = UsdShade.Shader.Define(stage, "/ScanMesh/Looks/ScanVertexColor/DisplayColorReader")
rd.CreateIdAttr("UsdPrimvarReader_float3")
rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
rd.CreateInput("fallback", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(0.5, 0.5, 0.5))
sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
for name, sel in [("Structure", ~ceiling), ("Ceiling", ceiling)]:
    used, Fi = submesh(V, F, sel)
    m = define_mesh(stage, f"/ScanMesh/{name}", V[used], Fi, N[used], C[used])
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    print(f"visual {name}: {len(Fi):,} tris")
stage.GetRootLayer().Save()

# ---------------------------------------------------------------- collision: clean shell + simplified furniture
shell = o3d.io.read_triangle_mesh(str(WORK / "shell.ply"))
objs_full = o3d.io.read_triangle_mesh(str(WORK / "objects.ply"))
objs = objs_full.simplify_quadric_decimation(150_000)
objs.remove_degenerate_triangles()
objs.remove_unreferenced_vertices()
stage = new_stage(PAY / "collision_mesh.usdc")
root = UsdGeom.Xform.Define(stage, "/Collision")
stage.SetDefaultPrim(root.GetPrim())
pm = UsdShade.Material.Define(stage, "/Collision/PhysicsMaterial")
pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
pma.CreateStaticFrictionAttr(0.8)
pma.CreateDynamicFrictionAttr(0.6)
pma.CreateRestitutionAttr(0.0)
for name, mm in [("Shell", shell), ("Objects", objs)]:
    m = define_mesh(stage, f"/Collision/{name}", np.asarray(mm.vertices), np.asarray(mm.triangles))
    m.CreatePurposeAttr(UsdGeom.Tokens.guide)                # not rendered, not seen by RTX sensors
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)  # static tri mesh
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
    print(f"collision {name}: {len(mm.triangles):,} tris")
stage.GetRootLayer().Save()

# ---------------------------------------------------------------- Gaussian splat (cropped)
hdr, props, gs = read_gs_ply(GSPLAT)
c = {p: i for i, p in enumerate(props)}
Pw = apply(T_gs2w, gs[:, [c["x"], c["y"], c["z"]]].astype(np.float64))
mask = np.load(WORK / "footprint_mask.npy")
mj = json.load(open(WORK / "footprint_mask.json"))
from scipy import ndimage  # noqa: E402
mask = ndimage.binary_dilation(mask, iterations=4)            # +1 m beyond the walls (window views)
ij = np.floor(Pw[:, :2] / mj["cell"]).astype(int) - np.array(mj["lo"])
inb = (ij >= 0).all(1) & (ij < mask.shape).all(1)
inside = np.zeros(len(Pw), bool)
inside[inb] = mask[ij[inb, 0], ij[inb, 1]]
smax = np.exp(gs[:, [c["scale_0"], c["scale_1"], c["scale_2"]]]).max(1) * fr["gsplat_scale"]
keep = inside & (Pw[:, 2] > -0.5) & (Pw[:, 2] < stats["zmax"] + 0.5) & (smax < 2.0)
cropped = WORK / "gaussian_splat_cropped.ply"
write_gs_ply(cropped, props, gs[keep])
print(f"splat crop: {keep.sum():,} / {len(gs):,} gaussians kept")
write_gaussian_splat_usd(read_ply(str(cropped)), str(PAY / "gaussian_splat.usdc"),
                         source_file=str(GSPLAT), prim_name="GaussianSplat", up_axis="Z")

# Furniture-only splat: Gaussians hugging the extracted furniture, away from floor/walls/ceiling.
from scipy.spatial import cKDTree  # noqa: E402
otree = cKDTree(np.asarray(objs_full.vertices))
d_obj, _ = otree.query(Pw, distance_upper_bound=0.08, workers=-1)
op_all = 1 / (1 + np.exp(-gs[:, c["opacity"]]))
keep_o = np.isfinite(d_obj) & (Pw[:, 2] > 0.02) & (Pw[:, 2] < 2.2) & (smax < 0.25) & (op_all > 0.02)
cropped_o = WORK / "gaussian_splat_objects.ply"
write_gs_ply(cropped_o, props, gs[keep_o])
print(f"furniture splat: {keep_o.sum():,} gaussians")
write_gaussian_splat_usd(read_ply(str(cropped_o)), str(PAY / "gaussian_splat_objects.usdc"),
                         source_file=str(GSPLAT), prim_name="FurnitureSplat", up_axis="Z")

# ---------------------------------------------------------------- asset root
stage = new_stage(OUT / "DX_Hub.usda")
layer = stage.GetRootLayer()
layer.documentation = "DX Hub indoor scan (FJD SLAM LiDAR + 3D Gaussian Splat), converted for NVIDIA Isaac Sim"
layer.customLayerData = {
    "source_scan": "260622_dx_scan01_2026-06-22-14-23-30 (FJD Trion, 92,760,841 pts)",
    "frame": "Z-up, meters. Floor plane z=0, walls aligned to X/Y, origin = centre of floor footprint",
    "T_las_to_world": str(np.array(fr["T_las_to_world"]).round(6).tolist()),
    "T_gsplat_to_world": str(T_gs2w.round(6).tolist()),
}
asset = UsdGeom.Xform.Define(stage, "/DX_Hub")
stage.SetDefaultPrim(asset.GetPrim())
Usd.ModelAPI(asset).SetKind(Kind.Tokens.component)
asset.GetPrim().SetAssetInfo({"name": "DX_Hub", "version": "2.0"})

for name, f in [("Structure", "structure.usdc"), ("Objects", "objects.usdc"), ("ScanMesh", "visual_mesh.usdc")]:
    stage.DefinePrim(f"/DX_Hub/{name}", "Xform").GetPayloads().AddPayload(f"./payloads/{f}")
stage.DefinePrim("/DX_Hub/Collision", "Xform").GetReferences().AddReference("./payloads/collision_mesh.usdc")
gs_x = UsdGeom.Xform.Define(stage, "/DX_Hub/GaussianSplat")
gs_x.AddTransformOp().Set(Gf.Matrix4d(T_gs2w.T.tolist()))     # USD uses row-vector convention
stage.DefinePrim("/DX_Hub/GaussianSplat/Splat").GetPayloads().AddPayload("./payloads/gaussian_splat.usdc")
stage.DefinePrim("/DX_Hub/GaussianSplat/FurnitureSplat").GetPayloads().AddPayload(
    "./payloads/gaussian_splat_objects.usdc")

# Ceiling lights at the recessed panels measured in the scan (RectLight emits along -Z = down).
LIGHT_RADIANCE = 30000.0
UsdGeom.Xform.Define(stage, "/DX_Hub/Lights")
for n_, pnl in enumerate(json.load(open(WORK / "lights.json"))["panels"]):
    lt = UsdLux.RectLight.Define(stage, f"/DX_Hub/Lights/Panel_{n_:03d}")
    lt.CreateWidthAttr(pnl["l"])
    lt.CreateHeightAttr(pnl["w"])
    lt.CreateIntensityAttr(LIGHT_RADIANCE)
    lt.CreateColorTemperatureAttr(5000.0)
    lt.CreateEnableColorTemperatureAttr(True)
    x = UsdGeom.Xformable(lt)
    x.AddTranslateOp().Set(Gf.Vec3d(pnl["x"], pnl["y"], pnl["z"] - 0.01))
    x.AddRotateZOp().Set(pnl["ang"])
print(f"ceiling lights: {n_ + 1}")

# Variant sets: "appearance" is listed first, so its opinions win over "objects".
vset = asset.GetPrim().GetVariantSets().AddVariantSet("appearance")
oset = asset.GetPrim().GetVariantSets().AddVariantSet("objects")
for name in ["clean", "scan", "gsplat"]:
    vset.AddVariant(name)
    vset.SetVariantSelection(name)
    with vset.GetVariantEditContext():
        if name != "scan":
            stage.OverridePrim("/DX_Hub/ScanMesh").SetActive(False)
        if name != "gsplat":
            stage.OverridePrim("/DX_Hub/GaussianSplat/Splat").SetActive(False)
        if name != "clean":
            for pth in ["/DX_Hub/Structure", "/DX_Hub/Objects"]:
                UsdGeom.Imageable(stage.OverridePrim(pth)).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            stage.OverridePrim("/DX_Hub/GaussianSplat/FurnitureSplat").SetActive(False)
vset.SetVariantSelection("clean")
for name in ["scan", "gsplat", "off"]:                       # how furniture is shown in the clean building
    oset.AddVariant(name)
    oset.SetVariantSelection(name)
    with oset.GetVariantEditContext():
        if name != "gsplat":
            stage.OverridePrim("/DX_Hub/GaussianSplat/FurnitureSplat").SetActive(False)
        if name == "gsplat":                                  # furniture mesh hidden but still collides
            UsdGeom.Imageable(stage.OverridePrim("/DX_Hub/Objects")).CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        if name == "off":                                     # empty building: no furniture visuals/colliders
            stage.OverridePrim("/DX_Hub/Objects").SetActive(False)
            stage.OverridePrim("/DX_Hub/Collision/Objects").SetActive(False)
oset.SetVariantSelection("gsplat")
layer.Save()

# ---------------------------------------------------------------- demo scene
scene = new_stage(OUT / "DX_Hub_scene.usda")
world = UsdGeom.Xform.Define(scene, "/World")
scene.SetDefaultPrim(world.GetPrim())
hub = scene.DefinePrim("/World/DX_Hub")
hub.GetReferences().AddReference("./DX_Hub.usda")
ps = UsdPhysics.Scene.Define(scene, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
ps.CreateGravityMagnitudeAttr(9.81)
UsdGeom.Xform.Define(scene, "/World/Lights")
dome = UsdLux.DomeLight.Define(scene, "/World/Lights/Dome")
dome.CreateIntensityAttr(800)
sun = UsdLux.DistantLight.Define(scene, "/World/Lights/Key")
sun.CreateIntensityAttr(1500)
sun.CreateAngleAttr(1.0)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35, 20, 0))
scene.GetRootLayer().Save()
print("wrote", OUT)
