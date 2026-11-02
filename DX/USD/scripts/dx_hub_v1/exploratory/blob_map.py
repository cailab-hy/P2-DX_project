"""Numbered map of the furniture-height blobs in the Hall (for the user to say which furniture each one is).

Blobs = connected scan cells 0.15-0.80 m (>= 2 points per 5 cm cell, walls removed, closed by 1 cell), area >= 0.08 m^2,
inside the Hall. Background = the scan's own colours seen from above (highest point below 2.3 m). Writes
out/blobs_hall.png (3x, 1 m grid, numbers) and work/blobs.json (id, centre, size, bbox x0 y0 x1 y1, top height).
    python blob_map.py LAS WORKDIR OUTDIR
"""
import json
import sys

import cv2
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, label

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from scanio import read_las  # noqa: E402

las, work, out = sys.argv[1:4]
RES, S = 0.05, 3
X0, X1, Y0, Y1 = -14.0, 1.0, 3.0, 23.0
nx, ny = int((X1 - X0) / RES), int((Y1 - Y0) / RES)
T = np.load(f'{work}/T_scan_to_world.npy')
xyz, rgb = read_las(las)
W = xyz @ T[:3, :3].T + T[:3, 3]
m = (W[:, 0] > X0) & (W[:, 0] < X1) & (W[:, 1] > Y0) & (W[:, 1] < Y1)
W, rgb = W[m], rgb[m]
ij = np.floor((W[:, :2] - [X0, Y0]) / RES).astype(int)

band = (W[:, 2] > 0.15) & (W[:, 2] < 0.80)
cnt = np.zeros((nx, ny), np.int32)
np.add.at(cnt, (ij[band, 0], ij[band, 1]), 1)
B = np.load(f'{work}/building_points.npy')
B = B[(B[:, 2] > 0.15) & (B[:, 2] < 0.80) & (B[:, 0] > X0) & (B[:, 0] < X1) & (B[:, 1] > Y0) & (B[:, 1] < Y1)]
wall = np.zeros((nx, ny), bool)
bij = np.floor((B[:, :2] - [X0, Y0]) / RES).astype(int)
wall[bij[:, 0], bij[:, 1]] = True
occ = binary_closing((cnt >= 2) & ~binary_dilation(wall, iterations=2), iterations=1)
lab, n = label(occ)

top = np.zeros((nx, ny), np.float32)
low = W[:, 2] < 2.3
np.maximum.at(top, (ij[low, 0], ij[low, 1]), W[low, 2].astype(np.float32))
# colour from above: per cell the colour of the highest point below 2.3 m
order = np.argsort(W[low, 2])
img = np.full((nx, ny, 3), 255, np.uint8)
cij = ij[low][order]
img[cij[:, 0], cij[:, 1]] = (np.clip(rgb[low][order], 0, 1) * 255).astype(np.uint8)

blobs = []
for k in range(1, n + 1):
    cells = np.argwhere(lab == k)
    area = len(cells) * RES * RES
    if area < 0.08:
        continue
    (i0, j0), (i1, j1) = cells.min(0), cells.max(0)
    blobs.append({'id': len(blobs) + 1, 'centre': [round(X0 + (cells[:, 0].mean() + 0.5) * RES, 2), round(Y0 + (cells[:, 1].mean() + 0.5) * RES, 2)],
                  'size': [round((i1 - i0 + 1) * RES, 2), round((j1 - j0 + 1) * RES, 2)], 'area_m2': round(area, 2),
                  'top_m': round(float(np.percentile(top[lab == k], 90)), 2), 'cells': (i0, j0, i1, j1),
                  'bbox': [round(X0 + i0 * RES, 3), round(Y0 + j0 * RES, 3), round(X0 + (i1 + 1) * RES, 3), round(Y0 + (j1 + 1) * RES, 3)]})

view = np.ascontiguousarray(np.rot90(img))                       # x to the right, y up
view = cv2.resize(view, (view.shape[1] * S, view.shape[0] * S), interpolation=cv2.INTER_NEAREST)
view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
H = view.shape[0]
for xm in range(int(X0), int(X1) + 1):
    c = int((xm - X0) / RES * S)
    cv2.line(view, (c, 0), (c, H), (215, 215, 215), 1)
    cv2.putText(view, f'{xm}', (c + 2, H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
for ym in range(int(Y0), int(Y1) + 1):
    r = int((Y1 - ym) / RES * S)
    cv2.line(view, (0, r), (view.shape[1], r), (215, 215, 215), 1)
    cv2.putText(view, f'{ym}', (2, r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)
for b in blobs:
    i0, j0, i1, j1 = b['cells']
    p0 = (int(i0 * S), int((ny - 1 - j1) * S)); p1 = (int((i1 + 1) * S), int((ny - j0) * S))
    cv2.rectangle(view, p0, p1, (40, 40, 220), 1)
    cv2.putText(view, str(b['id']), (p0[0], max(p0[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 200), 2)
cv2.imwrite(f'{out}/blobs_hall.png', view)
for b in blobs:
    b.pop('cells')
json.dump(blobs, open(f'{work}/blobs.json', 'w'), indent=1)
print(f'{len(blobs)} blobs; image {view.shape[1]}x{view.shape[0]}')
for b in blobs:
    print(f"{b['id']:3d}  centre ({b['centre'][0]:6.2f}, {b['centre'][1]:6.2f})  {b['size'][0]:.2f} x {b['size'][1]:.2f} m  top {b['top_m']:.2f} m")
