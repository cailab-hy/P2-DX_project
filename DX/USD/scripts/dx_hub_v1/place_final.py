"""Hall furniture from the merged 10/30 + 11/02 scans + the user's notes -> override layer on the modelled DX Hub.

Measured on the scans: sofa groups (fit.json, headings as modelled), plants, TV, desks. Unchanged per the user: the office
corner's modelled desks and chairs, the Desk6 row with the chairs on both sides, CareRoom, Kitchen (except its plant).
Existing Hall prims of the same asset are moved first; extras are added from the asset USD; unused ones deactivated.
    python place_final.py WORK_A WORK_B SOURCE.usda OUT.usda
"""
import json
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom

work_a, work_b, src, out = sys.argv[1:5]
items = json.load(open(f'{work_a}/furniture_v1.json'))
fit = json.load(open(f'{work_b}/fit.json'))
HALL = '/DX_Hub_Building_add_obj/Furniture/Hall'
place = []   # (asset, x, y, yaw, note)

# sofa groups as measured (order: S, C, N rows; west then east = the modelled prims' order)
for f in fit:
    if f.get('asset') in ('SofaTable', 'Armchair', 'Sofa3'):
        place.append((f['asset'], f['x'], f['y'], f['yaw'], f['why']))
# plants between the tables, seen in both scans
place.append(('Plant', -9.61, 5.16, 0.0, 'both scans'))
place.append(('Plant', -9.49, 7.89, 0.0, 'both scans'))
# TV: moved 37 cm between the scans, 11/02 wins; heading as before
place.append(('TVStand', -4.96, 14.49, 90.0, '11/02: top 1.95 m long, x -4.96'))
# blob 26 small chairs (user label, not confirmed by 11/02 yet)
place.append(('Chair6', -5.01, 13.62, 0.0, 'legs y 13.40..13.84, back on the south side'))
place.append(('Chair6', -4.33, 13.28, 0.0, 'label 26'))
place.append(('Chair6', -5.01, 15.385, 180.0, 'user: legs y 15.17..15.60, back on the north side'))
# fire extinguisher by the pillar (user, 2026-11-02): the red points, y 5.2..5.5, x -1.05..-0.8; the Kitchen one's asset
place.append(('FireExtinguisher', -0.90, 5.35, 90.0, 'red points'))
place.append(('FireExtinguisher', -4.53, 18.43, 90.0, 'user: arc x -4.62, y 18.33 at the wall end'))
# blob 28: top 1.15 x 1.74 m -> Desk6 along y; blob 31 table
place.append(('Desk6', -3.57, 14.50, 90.0, 'top y 13.63..15.37'))
place.append(('OfficeDesk', -3.17, 16.93, 0.0, 'label 31'))
# blob 35: labelled chairs, but both scans show a flat 1.41 x 1.30 m top at desk height -> two desks back to back
place.append(('OfficeDesk', -1.45, 14.94, 0.0, 'top x -2.15..-0.74 y 13.94..15.24'))
place.append(('OfficeDesk', -1.45, 14.24, 180.0, 'top x -2.15..-0.74 y 13.94..15.24'))
# chairs on both open sides of that pair (user, 2026-11-02; the east end is the wall). N sits on the seat points; S has no points in either scan
place.append(('OfficeChair', -1.05, 15.72, 180.0, 'seat x -1.2..-0.8 y 15.5..16.1, no back seen'))
place.append(('OfficeChair', -1.45, 13.62, 0.0, 'user: no scan points, centred on the desk'))
# office desks found on the scans outside the modelled office corner (blobs 1, 9), with their chairs
for dx, dy, dyaw, cx, cy, cyaw in ((-12.44, 16.90, -90.0, -11.82, 16.90, 90.0), (-12.44, 15.27, -90.0, -11.82, 15.27, 90.0)):
    place.append(('OfficeDesk', dx, dy, dyaw, 'scan top'))
    place.append(('OfficeChair', cx, cy, cyaw, 'scan seat / back'))
# desk with a chair on each long side (user, 2026-11-02): long side along y; chairs measured, both facing the desk
place.append(('OfficeDesk', -9.85, 16.33, 90.0, 'user: long side along y'))
place.append(('OfficeChair', -10.75, 16.50, -90.0, 'seat x -10.6, back x -11.07'))
place.append(('OfficeChair', -8.77, 16.20, 90.0, 'seat and back, 11/02'))

# Chair6: both sides of the Desk6 row stay as modelled (west side: user, 2026-11-02, though both scans show few points there)
# office desks shifted onto the desk-top points (user, 2026-11-02; 1.4 x 1.42 m double tops measured in both scans);
# each chair moves with its desk by the same offset (chairs not measured yet). name: new x, y
OFFICE_MOVES = {'OfficeDesk_3': (-10.10, 18.18), 'OfficeDesk_5': (-9.40, 18.18), 'OfficeDesk_7': (-7.73, 18.88),
                'OfficeDesk_9': (-7.03, 18.88), 'OfficeDesk_11': (-4.91, 20.30)}
keep = {i['path'] for i in items if i['room'] == 'Hall' and i['asset'] in ('Desk6.usda', 'OfficeDesk.usda', 'OfficeChair.usda', 'Chair6.usda')}
pool = {}
for i in items:
    if i['room'] == 'Hall' and i['path'] not in keep:
        pool.setdefault(i['asset'][:-5], []).append(i['path'])
pool.setdefault('Plant', []).append('/DX_Hub_Building_add_obj/Furniture/Kitchen/Plant')

layer = Sdf.Layer.CreateNew(out)
base = os.path.dirname(os.path.abspath(out))
layer.subLayerPaths.append(os.path.relpath(src, base))
stage = Usd.Stage.Open(layer)
stage.SetEditTarget(Usd.EditTarget(layer))
stage.SetDefaultPrim(stage.GetPrimAtPath('/DX_Hub_Building_add_obj'))
layer.customLayerData = {'source': 'scans 2026-10-30 + 2026-11-02 and user labels (DX/USD/scripts/place_final.py)'}
asset_dir = os.path.join(os.path.dirname(src), 'assets')
log, added = [], {}
for asset, x, y, yaw, note in place:
    if pool.get(asset):
        path = pool[asset].pop(0)
        prim = stage.OverridePrim(path)
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3d(float(x), float(y), 0.0))
        prim.GetAttribute('xformOp:rotateZ').Set(float(yaw))
        log.append(['move', path.split('/')[-1], asset, x, y, yaw, note])
    else:
        added[asset] = added.get(asset, 0) + 1
        path = f'{HALL}/Scan_{asset}_{added[asset]}'
        prim = stage.DefinePrim(path, 'Xform')
        prim.GetReferences().AddReference(os.path.relpath(os.path.join(asset_dir, asset, f'{asset}.usda'), base))
        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), 0.0))
        xf.AddRotateZOp().Set(float(yaw))
        log.append(['add', path.split('/')[-1], asset, x, y, yaw, note])
by_name = {i['path'].split('/')[-1]: i for i in items}
for desk, (nx_, ny_) in OFFICE_MOVES.items():
    old = by_name[desk]
    dx_, dy_ = nx_ - old['pos'][0], ny_ - old['pos'][1]
    for name, x_, y_ in ((desk, nx_, ny_),):
        stage.OverridePrim(old['path']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(x_, y_, 0.0))
        log.append(['move', name, 'OfficeDesk', x_, y_, old['yaw'], 'desk-top points'])
    chair = by_name[desk.replace('Desk', 'Chair')]
    cx_, cy_ = round(chair['pos'][0] + dx_, 3), round(chair['pos'][1] + dy_, 3)
    stage.OverridePrim(chair['path']).GetAttribute('xformOp:translate').Set(Gf.Vec3d(cx_, cy_, 0.0))
    log.append(['move', chair['path'].split('/')[-1], 'OfficeChair', cx_, cy_, chair['yaw'], 'with its desk'])
for rest in pool.values():
    for path in rest:
        stage.OverridePrim(path).SetActive(False)
        log.append(['hide', path.split('/')[-1], '', '', '', '', ''])
# BRS120M robots by the west window (user, 2026-11-02): modelled from primitives, see brs120m.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import brs120m  # noqa: E402
brs = json.load(open(f'{work_b}/brs120m_fit.json'))
brs120m.add(stage, HALL, brs)
for n, f in enumerate(brs, 1):
    log.append(['add', f'Scan_BRS120M_{n}', 'BRS120M', f['x'], f['y'], f['yaw'], 'scan body + arm pose'])
# LG CLOi ServeBots by the curved desk (user, 2026-11-02): lofted mesh fitted to the points, see cloi.py
import cloi  # noqa: E402
clois = json.load(open(f'{work_b}/cloi_fit.json'))
cloi.add(stage, HALL, clois)
for n, f in enumerate(clois, 1):
    log.append(['add', f'Scan_CLOi_{n}', 'CLOi', f['x'], f['y'], f['yaw'], 'outline fit'])
# maum.ai Aiden robots around the curved wall (user, 2026-11-02): product spec + scan profile, see maumai.py
import maumai  # noqa: E402
aidens = json.load(open(f'{work_b}/maumai_fit.json'))
maumai.add(stage, HALL, aidens)
for n, f in enumerate(aidens, 1):
    log.append(['add', f'Scan_Aiden_{n}', 'Aiden', f['x'], f['y'], f['yaw'], 'base edge fit, spec size'])
# round table, round stools, potted plants (user's product photos, 2026-11-02): see round_items.py, monstera.py
import monstera  # noqa: E402
import round_items  # noqa: E402
rounds = json.load(open(f'{work_b}/round_items.json'))
round_items.add(stage, HALL, rounds)
log.append(['add', 'Scan_RoundTable_1', 'RoundTable', rounds['table']['x'], rounds['table']['y'], 0.0, 'circle fit'])
for n, f in enumerate(rounds['stools'], 1):
    log.append(['add', f'Scan_RoundStool_{n}', 'RoundStool', f['x'], f['y'], 0.0, 'circle fit'])
plants = json.load(open(f'{work_b}/monstera.json'))
monstera.add(stage, HALL, plants)
for n, f in enumerate(plants['pots'], 1):
    log.append(['add', f'Scan_Monstera_{n}', 'Monstera', f['x'], f['y'], 0.0, f"pot + {len(f['leaves'])} leaves from the points"])
for n, (x_, y_, _, _) in enumerate(monstera.FLOWERS, 1):
    log.append(['add', f'Scan_FlowerPot_{n}', 'FlowerPot', x_, y_, 0.0, 'points on the counter'])
# RBM-S100a robots beside the counter's end (user, 2026-11-02), see rbm_s100a.py
import rbm_s100a  # noqa: E402
rbm_s100a.add(stage, HALL)
for n, (x_, y_, yaw_) in enumerate(rbm_s100a.ROBOTS, 1):
    log.append(['add', f'Scan_RBM_{n}', 'RBM-S100a', x_, y_, yaw_, 'slices, page size'])
# RB-Y1 beside the T-shirt desk (user, 2026-11-02), see rby1.py
import rby1  # noqa: E402
rby1.add(stage, HALL)
log.append(['add', 'Scan_RBY1_1', 'RB-Y1', -2.56, 18.90, 0.0, 'slices, product photo'])
# chest of drawers in front of the office desks (user, 2026-11-02), see drawer_chest.py
import drawer_chest  # noqa: E402
drawer_chest.add(stage, HALL)
log.append(['add', 'Scan_DrawerChest_1', 'DrawerChest', -7.63, 17.86, 0.0, 'box on the slices'])
log.append(['add', 'Scan_DrawerChest_2', 'DrawerChest', -4.755, 19.01, 0.0, 'box on the slices'])
# pendant lamps, monitors (user, 2026-11-02), see pendant_lamps.py, monitors.py
import monitors  # noqa: E402
import pendant_lamps  # noqa: E402
lamps = json.load(open(f'{work_b}/pendant_lamps.json'))
pendant_lamps.add(stage, HALL, lamps)
for n, f in enumerate(lamps, 1):
    log.append(['add', f'Scan_PendantLamp_{n}', 'PendantLamp', f['x'], f['y'], 0.0, f"r {f['r']} rim {f['rim']}"])
mons = json.load(open(f'{work_b}/monitors.json'))
monitors.add(stage, HALL, mons)
for n, f in enumerate(mons, 1):
    log.append(['add', f'Scan_Monitor_{n}', 'Monitor', f['c'][0], f['c'][1], f['yaw'], f"run {f['seg']}"])
# ROBOTIS SG2 north of the Desk6 (user, 2026-11-02): pose fitted to the points (2.6 cm). Referenced, so a static display piece: no articulation, joints or moving bodies
SG2 = os.path.join(os.path.expanduser(os.environ.get('CYCLO_LAB', '~/Isaac/cyclo_lab')), 'source/cyclo_lab/data/robots/FFW/FFW_SG2.usd')
sg2 = stage.DefinePrim(f'{HALL}/Scan_SG2_1', 'Xform')
sg2.GetReferences().AddReference(os.path.relpath(SG2, base))
xf = UsdGeom.Xformable(sg2)
xf.AddTranslateOp().Set(Gf.Vec3d(-3.47, 15.37, 0.0))
xf.AddRotateZOp().Set(-90.0)
from pxr import UsdPhysics  # noqa: E402
for q in Usd.PrimRange(sg2, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
    if q.IsInstanceProxy():
        continue
    if q.HasAPI(UsdPhysics.ArticulationRootAPI):
        q.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if q.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(q).GetRigidBodyEnabledAttr().Set(False)
    if q.IsA(UsdPhysics.Joint):
        q.SetActive(False)
log.append(['add', 'Scan_SG2_1', 'FFW_SG2', -3.47, 15.37, -90.0, 'pose fit 2.6 cm, static'])
layer.Save()
json.dump(log, open(f'{work_b}/placement_final.json', 'w'), indent=1, ensure_ascii=False)
from collections import Counter
print(out, Counter(r[0] for r in log))
for r in log:
    print(*r[:6])
