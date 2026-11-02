"""Which modelled furniture still stands where the rescan sees something? (top-down, furniture height band)

Each object's footprint (world AABB of the modelled prim) is checked against the registered scan points between
0.15 and 0.80 m: support = share of the footprint's 5 cm cells with scan points. Writes work/furniture_check.json and
out/furniture_check.png (scan band = grey, footprints green = supported >= 0.30, orange 0.10-0.30, red < 0.10).
    python furniture_check.py LAS WORKDIR OUTDIR
"""
import json
import sys

import numpy as np
import open3d as o3d

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from scanio import read_las  # noqa: E402

las, work, out = sys.argv[1:4]
T = np.load(f'{work}/T_scan_to_world.npy')
items = json.load(open(f'{work}/furniture_v1.json'))
xyz, _ = read_las(las)
W = xyz @ T[:3, :3].T + T[:3, 3]
band = W[(W[:, 2] > 0.15) & (W[:, 2] < 0.80)]
RES = 0.05
lo = np.array([-15.0, -25.0]); hi = np.array([15.0, 25.0])
shape = tuple(np.ceil((hi - lo) / RES).astype(int))
occ = np.zeros(shape, np.int32)
ij = np.floor((band[:, :2] - lo) / RES).astype(int)
ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
np.add.at(occ, (ij[ok, 0], ij[ok, 1]), 1)
hit = occ >= 2

res = []
for it in items:
    a = np.floor((np.array(it['min'][:2]) - lo) / RES).astype(int)
    b = np.ceil((np.array(it['max'][:2]) - lo) / RES).astype(int)
    cell = hit[a[0]:b[0], a[1]:b[1]]
    support = float(cell.mean()) if cell.size else 0.0
    res.append({**it, 'support': round(support, 3),
                'status': 'kept' if support >= 0.30 else ('unclear' if support >= 0.10 else 'missing')})
json.dump(res, open(f'{work}/furniture_check.json', 'w'), indent=1, ensure_ascii=False)

img = np.full(shape + (3,), 255, np.uint8)
img[hit] = (120, 120, 120)
col = {'kept': (40, 170, 60), 'unclear': (240, 150, 20), 'missing': (220, 30, 30)}
for r in res:
    a = np.floor((np.array(r['min'][:2]) - lo) / RES).astype(int)
    b = np.ceil((np.array(r['max'][:2]) - lo) / RES).astype(int) - 1
    c = col[r['status']]
    img[a[0]:b[0] + 1, [a[1], b[1]]] = c
    img[[a[0], b[0]], a[1]:b[1] + 1] = c
crop = img[int((-14 - lo[0]) / RES):int((1 - lo[0]) / RES), int((3 - lo[1]) / RES):int((23 - lo[1]) / RES)]
o3d.io.write_image(f'{out}/furniture_check_hall.png', o3d.geometry.Image(np.ascontiguousarray(np.rot90(crop))))
from collections import Counter
print(Counter((r['room'], r['status']) for r in res))
for r in res:
    if r['room'] == 'Hall' and r['status'] != 'kept':
        print(f"{r['path'].split('/')[-1]:16s} {r['status']:8s} support {r['support']:.2f} at {r['pos'][:2]}")
