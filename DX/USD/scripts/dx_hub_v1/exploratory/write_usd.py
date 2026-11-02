"""Write the rescan meshes as an Isaac Sim USD in the twin world frame (meters, Z up).

    /DX_Hub_Scan                  (defaultPrim, Xform)
        Visual      Mesh, vertex colours (displayColor), no collision
        Collision   Mesh, invisible, UsdPhysics.CollisionAPI + MeshCollisionAPI (triangle mesh, static)
customLayerData records the source scan and T_scan_to_world. Run with the Isaac Lab python (pxr):
    python write_usd.py WORKDIR OUT.usd
"""
import json
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

work, out = sys.argv[1:3]
reg = json.load(open(f'{work}/T_scan_to_world.json'))
stage = Usd.Stage.CreateNew(out)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
root = UsdGeom.Xform.Define(stage, '/DX_Hub_Scan')
stage.SetDefaultPrim(root.GetPrim())
stage.GetRootLayer().customLayerData = {
    'source_scan': '2026-10-30-15-38-13 (FJD, 4,857,827 pts, DX Hub rescan after the furniture moved)',
    'frame': 'twin world (DX_Hub_Building_no_obj_v1): Z up, m, floor z = 0',
    'T_scan_to_world': json.dumps(reg['T_scan_to_world']),
    'registration': f"yaw {reg['yaw_deg']:.2f} deg, ICP rmse {reg['icp_rmse_cm']:.1f} cm (5 cm inliers {reg['icp_fitness_5cm']:.2f})",
    'pipeline': 'outputs/scan_20261030/tools (quick version: Poisson mesh, ceiling cut at 2.55 m)',
}


def mesh(path, npz, colours):
    d = np.load(npz)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(d['vertices']))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(d['triangles']), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(d['triangles'].reshape(-1)))
    m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    m.CreateDoubleSidedAttr(True)
    lo, hi = d['vertices'].min(0), d['vertices'].max(0)
    m.CreateExtentAttr([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))])
    if colours and len(d['colors']):
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar('displayColor', Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy(d['colors']))
    print(f'{path}: {len(d["vertices"]):,} vertices, {len(d["triangles"]):,} triangles')
    return m


mesh('/DX_Hub_Scan/Visual', f'{work}/mesh_visual.npz', True)
col = mesh('/DX_Hub_Scan/Collision', f'{work}/mesh_collision.npz', False)
col.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
UsdPhysics.CollisionAPI.Apply(col.GetPrim())
UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.none)
stage.GetRootLayer().Save()
print('saved', out)
