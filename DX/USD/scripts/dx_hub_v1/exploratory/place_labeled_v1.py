"""Hall furniture per the user's blob labels (2026-10-30) -> override layer on the modelled DX Hub (add_obj_v1).

Each labelled blob (work/blobs.json bbox) gets its furniture, laid along the blob's long side; chairs face the desks
next to them. Existing Hall prims of the same asset are moved first, extra ones are added (same asset USD), unused
ones are deactivated. Kept as they are: the Desk6 row (blob 29) with its east chairs, CareRoom and Kitchen furniture
(blobs 34, 37-40, 41 already match), except the Kitchen plant (moved to blob 11). Robots / empty blobs get nothing.
Asset origin = floor centre, front = +y (assets README). Run with the Isaac Lab python (pxr):
    python place_labeled.py WORKDIR SOURCE.usda OUT.usda
"""
import json
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom

work, src, out = sys.argv[1:4]
blobs = {b['id']: b['bbox'] for b in json.load(open(f'{work}/blobs.json'))}
items = json.load(open(f'{work}/furniture_v1.json'))
HALL = '/DX_Hub_Building_add_obj/Furniture/Hall'
# native footprint (x = width, y = depth) of each asset, m
SIZE = {'OfficeDesk': (1.4, 0.7), 'OfficeChair': (0.69, 0.72), 'SofaTable': (1.2, 0.8), 'Armchair': (0.71, 0.75),
        'Chair6': (0.53, 0.47), 'Desk6': (1.8, 0.8), 'TVStand': (1.45, 0.71), 'Plant': (0.4, 0.4)}
place = []   # (asset, x, y, yaw, blob)


def bbox(k):
    x0, y0, x1, y1 = blobs[k]
    return x0, y0, x1, y1, (x0 + x1) / 2, (y0 + y1) / 2


def along(k, asset, n, yaw_x=0.0, yaw_y=90.0):
    """n objects evenly along the blob's long side; long side of the asset along it."""
    x0, y0, x1, y1, cx, cy = bbox(k)
    w, d = SIZE[asset]
    if x1 - x0 >= y1 - y0:
        step = max(w, (x1 - x0) / n)
        for i in range(n):
            place.append((asset, cx + (i - (n - 1) / 2) * step, cy, yaw_x, k))
    else:
        step = max(w, (y1 - y0) / n)
        for i in range(n):
            place.append((asset, cx, cy + (i - (n - 1) / 2) * step, yaw_y, k))


def desks_with_chairs(k, n):
    """n office desks in a column on the blob's west side, a chair east of each facing it (as the old office rows)."""
    x0, y0, x1, y1, cx, cy = bbox(k)
    dx = x0 + 0.35 + 0.02
    for i in range(n):
        y = cy + (i - (n - 1) / 2) * 1.4
        place.append(('OfficeDesk', dx, y, -90.0, k))     # front (+y) turned to +x: the chair side
        place.append(('OfficeChair', dx + 0.62, y, 90.0, k))  # front turned to -x: facing the desk


def grid(k, asset, nx, ny, yaw=0.0, gap=0.8):
    x0, y0, x1, y1, cx, cy = bbox(k)
    for i in range(nx):
        for j in range(ny):
            place.append((asset, cx + (i - (nx - 1) / 2) * gap, cy + (j - (ny - 1) / 2) * gap, yaw, k))


desks_with_chairs(1, 3)                       # 1,2,4: white desks + black chairs along the west wall
along(6, 'OfficeChair', 2, 180.0, 90.0)       # 6: black chairs
place.append(('OfficeChair', *bbox(16)[4:], 180.0, 16))
desks_with_chairs(9, 2)                       # 9: white desks + black chairs
for k in (10, 12, 13):                        # black low tables (no black desk model: sofa table)
    along(k, 'SofaTable', 1, 0.0, 90.0)
_, _, _, _, cx, cy = bbox(17)                 # 17: white desks, 2 x 2 back to back
for i in (-0.7, 0.7):
    place.append(('OfficeDesk', cx + i, cy + 0.35, 0.0, 17))
    place.append(('OfficeDesk', cx + i, cy - 0.35, 180.0, 17))
place.append(('Plant', *bbox(11)[4:], 0.0, 11))       # 11: plant (the Kitchen one, missing there)
place.append(('Armchair', *bbox(18)[4:], 0.0, 18))
place.append(('OfficeChair', *bbox(22)[4:], 180.0, 22))
grid(23, 'OfficeChair', 2, 2, 180.0)          # 22, 23: black chairs
place.append(('TVStand', *bbox(25)[4:], 90.0, 25))
along(26, 'Chair6', 2, 0.0, 90.0)             # 26, 27: small chairs
place.append(('Chair6', *bbox(27)[4:], 0.0, 27))
along(28, 'Desk6', 1, 0.0, 90.0)              # 28: white desk (29 = the Desk6 row, kept)
along(31, 'OfficeDesk', 1, 0.0, 90.0)         # 31: table
along(35, 'OfficeChair', 2, 0.0, 90.0)        # 35: black chairs

# prim pool per asset: Hall prims, minus the kept Desk6 row and its east chairs
keep = {i['path'] for i in items if i['room'] == 'Hall' and (i['asset'] == 'Desk6.usda' or ('Chair6_' in i['path'] and 'E' in i['path'].split('_')[-1]))}
pool = {}
for i in items:
    if i['room'] == 'Hall' and i['path'] not in keep:
        pool.setdefault(i['asset'][:-5], []).append(i['path'])
pool.setdefault('Plant', []).append('/DX_Hub_Building_add_obj/Furniture/Kitchen/Plant')

layer = Sdf.Layer.CreateNew(out)
layer.subLayerPaths.append(os.path.relpath(src, os.path.dirname(os.path.abspath(out))))
stage = Usd.Stage.Open(layer)
stage.SetEditTarget(Usd.EditTarget(layer))
stage.SetDefaultPrim(stage.GetPrimAtPath('/DX_Hub_Building_add_obj'))
layer.customLayerData = {'source': 'scan 2026-10-30-15-38-13 + user labels of out/blobs_hall.png (outputs/scan_20261030/tools/place_labeled.py)'}
asset_dir = os.path.join(os.path.dirname(src), 'assets')
log, added = [], {}
for asset, x, y, yaw, k in place:
    if pool.get(asset):
        path = pool[asset].pop(0)
        prim = stage.OverridePrim(path)
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(x, y, 0.0))
        prim.GetAttribute('xformOp:rotateZ').Set(float(yaw))
        log.append(('move', path.split('/')[-1], asset, round(x, 2), round(y, 2), yaw, k))
    else:
        added[asset] = added.get(asset, 0) + 1
        path = f'{HALL}/Scan_{asset}_{added[asset]}'
        prim = stage.DefinePrim(path, 'Xform')
        ref = os.path.relpath(os.path.join(asset_dir, asset, f'{asset}.usda'), os.path.dirname(os.path.abspath(out)))
        prim.GetReferences().AddReference(ref)
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
        xf.AddRotateZOp().Set(float(yaw))
        log.append(('add', path.split('/')[-1], asset, round(x, 2), round(y, 2), yaw, k))
for rest in pool.values():
    for path in rest:
        stage.OverridePrim(path).SetActive(False)
        log.append(('hide', path.split('/')[-1], '', '', '', '', ''))
layer.Save()
json.dump(log, open(f'{work}/placement_labeled.json', 'w'), indent=1)
from collections import Counter
print(out, Counter(a for a, *_ in log))
for row in log:
    print(*row)
