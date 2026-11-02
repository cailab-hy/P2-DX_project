"""Measure furniture poses on the merged scans (10/30 + 11/02) from height slices; proposal JSON + overlay.

seat 0.20-0.60 m, top 0.66-0.80 m, back 0.90-1.35 m, tall 1.45-2.30 m (3 cm cells).
- sofa groups (x -11.6..-7.4, y 2.2..10): in each row around a low table, the west / east seat outline's extent; length < 1.6 m
  = two armchairs (1.42 m), longer = a 3-seat sofa (1.75 m). Headings as modelled: west -90, east +90, tables 90.
- office desks (1.4 x 0.7 tops, both orientations, greedy, fill >= MIN_TOP) outside the kept Desk6 row; a modelled desk within
  KEEP_DIST and the same orientation stays where it is. Each desk's chair = the long side with more seat / back cells.
Writes work/fit.json and out/fit_overlay.png.
    python fit_from_slices.py WORK_B OUT_B WORK_A
"""
import json
import sys

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation

work, out, work_a = sys.argv[1:4]
RES, S, MIN_TOP, KEEP_DIST = 0.03, 3, 0.55, 0.20
X0, X1, Y0, Y1 = -14.0, 1.0, 1.0, 23.0
nx, ny = int(np.ceil((X1 - X0) / RES)), int(np.ceil((Y1 - Y0) / RES))
P = np.load(f'{work}/merged_points.npz')['points']
P = P[(P[:, 0] > X0) & (P[:, 0] < X1) & (P[:, 1] > Y0) & (P[:, 1] < Y1)]
ij = np.minimum(np.floor((P[:, :2] - [X0, Y0]) / RES).astype(int), [nx - 1, ny - 1])


def mask(z0, z1, minc=1):
    c = np.zeros((nx, ny), np.int32)
    k = (P[:, 2] > z0) & (P[:, 2] < z1)
    np.add.at(c, (ij[k, 0], ij[k, 1]), 1)
    return c >= minc


seat, top, back, tall = mask(0.20, 0.60), mask(0.66, 0.80, 2), mask(0.90, 1.35), mask(1.45, 2.30)
B = np.load(f'{work_a}/building_points.npy')
B = B[(B[:, 2] > 0.1) & (B[:, 2] < 2.3) & (B[:, 0] > X0) & (B[:, 0] < X1) & (B[:, 1] > Y0) & (B[:, 1] < Y1)]
wall = np.zeros((nx, ny), bool)
bij = np.minimum(np.floor((B[:, :2] - [X0, Y0]) / RES).astype(int), [nx - 1, ny - 1])
wall[bij[:, 0], bij[:, 1]] = True
wall = binary_dilation(wall, iterations=3)
seat &= ~wall; top &= ~wall; back &= ~wall
cx = lambda x: int((x - X0) / RES)
cy = lambda y: int((y - Y0) / RES)
items = json.load(open(f'{work_a}/furniture_v1.json'))
fit = []


def extent(m, x0, x1, y0, y1, frac=0.02):
    """Robust bbox (world) of the cells of m in a window: trims sparse tails."""
    sub = m[cx(x0):cx(x1), cy(y0):cy(y1)]
    if sub.sum() < 20:
        return None
    px, py = sub.sum(1), sub.sum(0)
    def span(p, base):
        c = np.cumsum(p) / p.sum()
        a, b = np.searchsorted(c, frac), np.searchsorted(c, 1 - frac)
        return base + a * RES, base + (b + 1) * RES
    return (*span(px, x0), *span(py, y0)), py


# --- sofa groups: tables first (solid low rectangles at x ~ -9.5)
tables = []
col = seat[cx(-10.0):cx(-9.0)].sum(0).astype(float)
col = np.convolve(col, np.ones(5) / 5, mode='same')
inrow = col > 8
j = cy(2.2)
while j < cy(10.0):
    if inrow[j]:
        k = j
        while k < cy(10.0) and inrow[k]:
            k += 1
        if (k - j) * RES > 0.7:
            tables.append((Y0 + j * RES, Y0 + k * RES))
        j = k
    j += 1
print('tables (y):', [(round(a, 2), round(b, 2)) for a, b in tables])
for y0, y1 in tables:
    e = extent(seat, -10.1, -8.95, y0, y1)
    tx = (e[0][0] + e[0][1]) / 2
    ty = (y0 + y1) / 2
    fit.append({'asset': 'SofaTable', 'x': round(tx, 3), 'y': round(ty, 3), 'yaw': 90.0, 'why': f'table {e[0][1] - e[0][0]:.2f} x {y1 - y0:.2f} m'})
    for side, (xa, xb), yaw in (('W', (-11.6, -10.15), -90.0), ('E', (-8.9, -7.4), 90.0)):
        e = extent(seat, xa, xb, ty - 1.15, ty + 1.15)
        if e is None:
            fit.append({'asset': None, 'why': f'no seats {side} of table y {ty:.2f}'})
            continue
        (a, b, c, d), prof = e
        length = d - c
        ymid = (c + d) / 2
        p = np.convolve(prof.astype(float), np.ones(3) / 3, mode='same')
        off = cy(ty - 1.15)
        mid = p[cy(ymid - 0.12) - off:cy(ymid + 0.12) - off].min()
        ends = np.median(p[cy(c) - off:cy(d) - off])
        two = mid < 0.45 * ends
        xc = (a + b) / 2
        two = length < 1.60          # 2 armchairs = 1.42 m, a 3-seat sofa = 1.75 m (the ring outlines hide the gap)
        if length > 1.25 and two:
            for dy in (-0.37, 0.37):
                fit.append({'asset': 'Armchair', 'x': round(xc, 3), 'y': round(ymid + dy, 3), 'yaw': yaw,
                            'why': f'{side} pair, outline {b - a:.2f} x {length:.2f} m, gap {mid:.0f}/{ends:.0f}'})
        elif length > 1.25:
            fit.append({'asset': 'Sofa3', 'x': round(xc, 3), 'y': round(ymid, 3), 'yaw': yaw,
                        'why': f'{side} sofa, outline {b - a:.2f} x {length:.2f} m, no gap ({mid:.0f}/{ends:.0f})'})
        else:
            fit.append({'asset': 'Armchair', 'x': round(xc, 3), 'y': round(ymid, 3), 'yaw': yaw, 'why': f'{side} single, {b - a:.2f} x {length:.2f} m'})

# --- office desks: tops 1.4 x 0.7 in the office region (not the Desk6 row x -4.3..-3 y 5..11.6)
region = np.zeros((nx, ny), bool)
region[cx(-13.2):cx(-4.4), cy(14.2):cy(21.3)] = True
free = binary_closing(top, iterations=1) & region
desks = []
for _ in range(40):
    integ = np.pad(free.astype(np.int32).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best = (0, None)
    for (w, h), turn in (((1.4, 0.7), 0.0), ((0.7, 1.4), 90.0)):
        a, b = int(round(w / RES)), int(round(h / RES))
        box = (integ[a:, b:] - integ[:-a, b:] - integ[a:, :-b] + integ[:-a, :-b]) / float(a * b)
        k = np.unravel_index(np.argmax(box), box.shape)
        if box[k] > best[0]:
            best = (float(box[k]), (k[0], k[1], a, b, turn))
    if best[0] < MIN_TOP:
        break
    i, j, a, b, turn = best[1]
    free[i:i + a, j:j + b] = False
    desks.append({'x': X0 + (i + a / 2) * RES, 'y': Y0 + (j + b / 2) * RES, 'along_x': turn == 0.0, 'fill': best[0]})
print(f'{len(desks)} desk tops found')
model = [i for i in items if i['asset'] == 'OfficeDesk.usda']
chair_model = [i for i in items if i['asset'] == 'OfficeChair.usda']
used = set()
furn = seat | back
for dk in desks:
    # which long side has the chair?
    ox, oy = (0.0, 0.62) if dk['along_x'] else (0.62, 0.0)
    score = {}
    for sgn in (1, -1):
        qx, qy = dk['x'] + sgn * ox, dk['y'] + sgn * oy
        win = furn[cx(qx - 0.3):cx(qx + 0.3), cy(qy - 0.3):cy(qy + 0.3)]
        score[sgn] = float(win.mean()) if win.size else 0.0
    sgn = 1 if score[1] >= score[-1] else -1
    # front (+y of the asset) towards the chair side; chair faces the desk
    if dk['along_x']:
        d_yaw, c_yaw = (0.0, 180.0) if sgn > 0 else (180.0, 0.0)
    else:
        d_yaw, c_yaw = (-90.0, 90.0) if sgn > 0 else (90.0, -90.0)
    near = [m for m in model if m['path'] not in used and abs(((m['yaw'] % 180) == 0) - dk['along_x']) == 0
            and np.hypot(m['pos'][0] - dk['x'], m['pos'][1] - dk['y']) < KEEP_DIST]
    entry = {'asset': 'OfficeDesk', 'x': round(dk['x'], 3), 'y': round(dk['y'], 3), 'yaw': d_yaw, 'why': f"top fill {dk['fill']:.2f}"}
    if near:
        used.add(near[0]['path'])
        entry.update(keep=near[0]['path'], x=near[0]['pos'][0], y=near[0]['pos'][1], yaw=near[0]['yaw'])
    fit.append(entry)
    if score[sgn] > 0.08:
        fit.append({'asset': 'OfficeChair', 'x': round(entry['x'] + sgn * ox, 3), 'y': round(entry['y'] + sgn * oy, 3), 'yaw': c_yaw,
                    'why': f'seat/back share {score[sgn]:.2f} vs {score[-sgn]:.2f}'})
json.dump(fit, open(f'{work}/fit.json', 'w'), indent=1, ensure_ascii=False)

# overlay
img = np.full((nx, ny, 3), 255, np.uint8)
img[seat] = (230, 170, 80); img[top] = (120, 200, 120); img[back] = (120, 120, 240); img[tall] = (70, 70, 70)
view = cv2.resize(np.ascontiguousarray(np.rot90(img)), (nx * S, ny * S), interpolation=cv2.INTER_NEAREST)
SIZE = {'OfficeDesk': (1.4, 0.7), 'OfficeChair': (0.69, 0.72), 'SofaTable': (1.2, 0.8), 'Armchair': (0.71, 0.75), 'Sofa3': (1.75, 0.75)}
px = lambda x, y: (int((x - X0) / RES * S), int((Y1 - y) / RES * S))
for f in fit:
    if not f.get('asset'):
        continue
    w, d = SIZE[f['asset']]; c, s = np.cos(np.radians(f['yaw'])), np.sin(np.radians(f['yaw']))
    pts = [px(f['x'] + c * u - s * v, f['y'] + s * u + c * v) for u, v in ((-w/2, -d/2), (w/2, -d/2), (w/2, d/2), (-w/2, d/2))]
    colour = (0, 130, 0) if f.get('keep') else (200, 40, 0)
    cv2.polylines(view, [np.array(pts)], True, colour, 2)
    fx, fy = f['x'] - s * d / 2, f['y'] + c * d / 2
    cv2.circle(view, px(fx, fy), 4, colour, -1)
for xm in range(int(X0), int(X1) + 1):
    cv2.putText(view, str(xm), (px(xm, Y1)[0] + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.line(view, (px(xm, Y1)[0], 0), (px(xm, Y1)[0], view.shape[0]), (225, 225, 225), 1)
for ym in range(int(Y0), int(Y1) + 1):
    cv2.putText(view, str(ym), (2, px(X0, ym)[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1)
    cv2.line(view, (0, px(X0, ym)[1]), (view.shape[1], px(X0, ym)[1]), (225, 225, 225), 1)
cv2.imwrite(f'{out}/fit_overlay.png', view)
from collections import Counter
print(Counter((f.get('asset'), bool(f.get('keep'))) for f in fit))
for f in fit:
    if f.get('asset') in ('SofaTable', 'Armchair', 'Sofa3', None):
        print(f)
