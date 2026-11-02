"""First-pass relocation of the modelled Hall furniture onto the 2026-10-30 rescan (proposal, to be checked by eye).

Free scan cells = rescan points 0.15-0.80 m (>= 2 per 5 cm cell), minus walls (building points in that band, 10 cm
around), minus the footprints of furniture still in place (furniture_check: kept). Objects to place: Hall furniture
that is 'missing' (searched over the whole Hall) or 'unclear' (searched within 1.5 m of its old spot), largest first.
A placement = the object's footprint (its size, at its old yaw or turned 90 deg) where the share of free cells is
highest; that area is then used up. Score = free-cell share x height agreement (p90 of the scan tops in the footprint vs the object's height);
below MIN_SCORE -> not placed (proposed hidden).
Writes work/relocation.json and out/relocation_hall.png (scan grey, old red, new blue, kept green).
    python relocate.py LAS WORKDIR OUTDIR
"""
import json
import sys

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from scanio import read_las  # noqa: E402

las, work, out = sys.argv[1:4]
RES, MIN_FILL, LOCAL = 0.05, 0.35, 1.5
MIN_SCORE = 0.25   # fill x height agreement
HALL = (np.array([-13.6, 3.5]), np.array([0.6, 22.6]))
lo, hi = np.array([-15.0, -25.0]), np.array([15.0, 25.0])
shape = tuple(np.ceil((hi - lo) / RES).astype(int))


def cells(points):
    m = np.zeros(shape, np.int32)
    ij = np.floor((points[:, :2] - lo) / RES).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.add.at(m, (ij[ok, 0], ij[ok, 1]), 1)
    return m


def rect(cx, cy, sx, sy):
    a = np.floor((np.array([cx - sx / 2, cy - sy / 2]) - lo) / RES).astype(int)
    b = np.ceil((np.array([cx + sx / 2, cy + sy / 2]) - lo) / RES).astype(int)
    return slice(max(a[0], 0), b[0]), slice(max(a[1], 0), b[1])


T = np.load(f'{work}/T_scan_to_world.npy')
xyz, _ = read_las(las)
W = xyz @ T[:3, :3].T + T[:3, 3]
free = cells(W[(W[:, 2] > 0.15) & (W[:, 2] < 0.80)]) >= 2
# top of whatever stands in each cell (max scan z up to 2.3 m): furniture types differ in height
Hm = np.zeros(shape, np.float32)
_t = W[(W[:, 2] > 0.05) & (W[:, 2] < 2.3)]
_ij = np.floor((_t[:, :2] - lo) / RES).astype(int)
_ok = (_ij >= 0).all(1) & (_ij[:, 0] < shape[0]) & (_ij[:, 1] < shape[1])
np.maximum.at(Hm, (_ij[_ok, 0], _ij[_ok, 1]), _t[_ok, 2].astype(np.float32))
B = np.load(f'{work}/building_points.npy')
walls = binary_dilation(cells(B[(B[:, 2] > 0.15) & (B[:, 2] < 0.80)]) > 0, iterations=2)
free &= ~walls
items = json.load(open(f'{work}/furniture_check.json'))
for it in items:
    if it['status'] == 'kept':
        c = np.add(it['min'][:2], it['max'][:2]) / 2
        free[rect(*c, *(np.subtract(it['max'][:2], it['min'][:2]) + 0.1))] = False
hall = np.zeros(shape, bool)
hall[rect(*(HALL[0] + HALL[1]) / 2, *(HALL[1] - HALL[0]))] = True
free &= hall

todo = [it for it in items if it['room'] == 'Hall' and it['status'] != 'kept']
todo.sort(key=lambda it: -np.prod(np.subtract(it['max'][:2], it['min'][:2])))
used = free.copy()
result = []
for it in todo:
    size = np.subtract(it['max'][:2], it['min'][:2])
    old = np.add(it['min'][:2], it['max'][:2]) / 2
    best = (0.0, None)
    for turn, (sx, sy) in ((0, size), (90, size[::-1])):
        nx, ny = max(int(round(sx / RES)), 1), max(int(round(sy / RES)), 1)
        integ = np.pad(used.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        box = integ[nx:, ny:] - integ[:-nx, ny:] - integ[nx:, :-ny] + integ[:-nx, :-ny]
        fill = box / float(nx * ny)                       # fill[i, j]: footprint with corner cell (i, j)
        centre_x = lo[0] + (np.arange(fill.shape[0]) + nx / 2) * RES
        centre_y = lo[1] + (np.arange(fill.shape[1]) + ny / 2) * RES
        if it['status'] == 'unclear':                     # only near the old spot
            mask = (np.abs(centre_x - old[0])[:, None] <= LOCAL) & (np.abs(centre_y - old[1])[None, :] <= LOCAL)
            fill = np.where(mask, fill, 0.0)
        # height agreement at the best few footprints: p90 of the cell tops inside vs the object's height
        height = it['max'][2] - it['min'][2]
        flat = np.argsort(fill, axis=None)[::-1][:400]
        for f in flat:
            i, j = np.unravel_index(f, fill.shape)
            if fill[i, j] < MIN_FILL:
                break
            top = Hm[i:i + nx, j:j + ny]
            top = top[top > 0]
            if top.size < 5:
                continue
            agree = float(np.exp(-((np.percentile(top, 90) - height) / 0.25) ** 2))
            sc = float(fill[i, j]) * agree
            if sc > best[0]:
                best = (sc, (centre_x[i], centre_y[j], turn, sx, sy))
    entry = {k: it[k] for k in ('path', 'asset', 'status', 'support', 'pos', 'yaw')}
    if best[1] is None or best[0] < MIN_SCORE:
        entry.update(action='hide', fill=round(best[0], 2))
    else:
        cx, cy, turn, sx, sy = best[1]
        # translate moves the prim origin: keep the origin's offset from the footprint centre (turned with the yaw)
        off = np.array(it['pos'][:2]) - old
        if turn:
            off = np.array([-off[1], off[0]])
        entry.update(action='move', fill=round(best[0], 2), new_pos=[round(cx + off[0], 3), round(cy + off[1], 3), it['pos'][2]],
                     new_yaw=it['yaw'] + turn, footprint=[round(cx - sx / 2, 3), round(cy - sy / 2, 3), round(cx + sx / 2, 3), round(cy + sy / 2, 3)])
        used[rect(cx, cy, sx, sy)] = False
    result.append(entry)
json.dump(result, open(f'{work}/relocation.json', 'w'), indent=1, ensure_ascii=False)

img = np.full(shape + (3,), 255, np.uint8)
img[cells(W[(W[:, 2] > 0.15) & (W[:, 2] < 0.80)]) >= 2] = (150, 150, 150)
def box(x0, y0, x1, y1, colour):
    a = np.floor((np.array([x0, y0]) - lo) / RES).astype(int); b = np.ceil((np.array([x1, y1]) - lo) / RES).astype(int) - 1
    img[a[0]:b[0] + 1, [a[1], b[1]]] = colour; img[[a[0], b[0]], a[1]:b[1] + 1] = colour
for it in items:
    if it['room'] == 'Hall' and it['status'] == 'kept':
        box(*it['min'][:2], *it['max'][:2], (40, 170, 60))
for r, it in zip(result, todo):
    box(*it['min'][:2], *it['max'][:2], (240, 160, 160))
for r in result:
    if r['action'] == 'move':
        box(*r['footprint'], (30, 60, 220))
crop = img[int((-14 - lo[0]) / RES):int((1 - lo[0]) / RES), int((3 - lo[1]) / RES):int((23 - lo[1]) / RES)]
big = np.repeat(np.repeat(np.rot90(crop), 3, 0), 3, 1)
o3d.io.write_image(f'{out}/relocation_hall.png', o3d.geometry.Image(np.ascontiguousarray(big)))
moved = [r for r in result if r['action'] == 'move']
print(f'{len(todo)} Hall objects to place: moved {len(moved)}, hidden {len(result) - len(moved)}')
for r in result:
    tag = f"-> {r['new_pos'][:2]} yaw {r['new_yaw']}" if r['action'] == 'move' else 'hide'
    print(f"{r['path'].split('/')[-1]:16s} {r['status']:8s} fill {r['fill']:.2f} {tag}")
