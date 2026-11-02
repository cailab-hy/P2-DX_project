"""Override layer: the modelled DX Hub (add_obj_v1) with the Hall furniture moved per relocation.json.

The source asset is a sublayer and is not changed. Moved prims get new xformOp:translate / xformOp:rotateZ; prims with
action 'hide' are deactivated (no render, no physics). defaultPrim stays /DX_Hub_Building_add_obj, so the file drops in
wherever DX_Hub_Building_add_obj_v1.usda is referenced.
    python write_layout.py relocation.json SOURCE.usda OUT.usda
"""
import json
import os
import sys

from pxr import Gf, Sdf, Usd

reloc, src, out = sys.argv[1:4]
layer = Sdf.Layer.CreateNew(out)
layer.subLayerPaths.append(os.path.relpath(src, os.path.dirname(os.path.abspath(out))))
stage = Usd.Stage.Open(layer)
stage.SetEditTarget(Usd.EditTarget(layer))
stage.SetDefaultPrim(stage.GetPrimAtPath('/DX_Hub_Building_add_obj'))
layer.customLayerData = {'source': 'scan 2026-10-30-15-38-13 relocation (outputs/scan_20261030/tools/relocate.py)',
                         'note': 'first pass, to be checked by eye; hidden = deactivated'}
moved = hidden = 0
for r in json.load(open(reloc)):
    prim = stage.OverridePrim(r['path'])
    if r['action'] == 'move':
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(*r['new_pos']))
        prim.GetAttribute('xformOp:rotateZ').Set(float(r['new_yaw']))
        moved += 1
    else:
        prim.SetActive(False)
        hidden += 1
layer.Save()
print(f'{out}: moved {moved}, hidden {hidden}')
