"""Merge the 10/30 and 11/02 DX Hub scans in the twin frame and show what changed (11/02 = the newer, wins).

1. refine scan B onto scan A (point-to-plane ICP, 0.10 -> 0.03 m, 5 cm voxels)
2. per scan and 10 cm cell, furniture band 0.15-0.80 m: persistent = seen in >= MIN_WINDOWS different 5 s windows
   (walking people leave only a few windows); floor seen = points below 0.05 m in the cell (nothing stands there
   when that scan looked)
3. classes (Hall, walls removed): both / only B, A saw floor = new / only B = new or unseen before /
   only A, B saw floor = removed / only A, B did not see the floor = unknown, A kept
Writes WORK_B/T_scan_to_world_refined.npy, merged points (z < 2.55) work_b/merged_points.npz, out_b/scan_diff.png.
    python compare_scans.py LAS_A WORK_A LAS_B WORK_B OUT_B
"""
import json
import sys

import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from scanio import read_las  # noqa: E402
import struct  # noqa: E402

las_a, work_a, las_b, work_b, out_b = sys.argv[1:6]
X0, X1, Y0, Y1, RES, MIN_WINDOWS = -14.0, 1.0, 3.0, 23.0, 0.10, 3
nx, ny = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))
BAND = (0.15, 0.80)


def times(path):
    h = open(path, 'rb').read(400)
    off, rl, n = struct.unpack('<I', h[96:100])[0], struct.unpack('<H', h[105:107])[0], struct.unpack('<I', h[107:111])[0]
    t = np.memmap(path, dtype=np.dtype([('p', f'V20'), ('t', '<f8'), ('q', f'V{rl - 28}')]), mode='r', offset=off, shape=(n,))['t']
    return np.asarray(t) - float(np.min(t))


def world(las, T):
    xyz, rgb = read_las(las)
    return xyz @ T[:3, :3].T + T[:3, 3], rgb, times(las)


def analyse(P, t):
    m = (P[:, 0] > X0) & (P[:, 0] < X1) & (P[:, 1] > Y0) & (P[:, 1] < Y1)
    Q, tq = P[m], t[m]
    ij = np.floor((Q[:, :2] - [X0, Y0]) / RES).astype(int)
    band = (Q[:, 2] > BAND[0]) & (Q[:, 2] < BAND[1])
    key = ij[band, 0] * ny + ij[band, 1]
    pairs = np.unique(np.stack([key, (tq[band] // 5).astype(np.int64)], 1), axis=0)
    cells, counts = np.unique(pairs[:, 0], return_counts=True)
    win = np.zeros((nx, ny), np.int32)
    win[cells // ny, cells % ny] = counts
    floor = np.zeros((nx, ny), bool)
    low = Q[:, 2] < 0.05
    floor[ij[low, 0], ij[low, 1]] = True
    return win >= MIN_WINDOWS, (win > 0) & (win < MIN_WINDOWS), floor


Ta = np.load(f'{work_a}/T_scan_to_world.npy')
Tb = np.load(f'{work_b}/T_scan_to_world.npy')
Pa, Ca, ta = world(las_a, Ta)
Pb, Cb, tb = world(las_b, Tb)

# 1) refine B onto A on the walls / structure seen by both (z 0.3-2.5)
def cloud(P):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[(P[:, 2] > 0.3) & (P[:, 2] < 2.5)]))
    return pc.voxel_down_sample(0.05)


src, dst = cloud(Pb), cloud(Pa)
dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
d = np.eye(4)
for dist in (0.10, 0.05, 0.03):
    r = o3d.pipelines.registration.registration_icp(src, dst, dist, d, o3d.pipelines.registration.TransformationEstimationPointToPlane())
    d = r.transformation
    print(f'ICP B->A {dist:.2f} m: fitness {r.fitness:.3f}, rmse {r.inlier_rmse * 100:.1f} cm', flush=True)
print(f'B correction: {np.round(d[:3, 3] * 100, 1)} cm, yaw {np.degrees(np.arctan2(d[1, 0], d[0, 0])):.2f} deg')
Pb = Pb @ d[:3, :3].T + d[:3, 3]
np.save(f'{work_b}/T_scan_to_world_refined.npy', d @ Tb)

# 2-3) classes
Ap, At, Af = analyse(Pa, ta)
Bp, Bt, Bf = analyse(Pb, tb)
B_ = np.load(f'{work_a}/building_points.npy')
B_ = B_[(B_[:, 2] > 0.15) & (B_[:, 2] < 0.80) & (B_[:, 0] > X0) & (B_[:, 0] < X1) & (B_[:, 1] > Y0) & (B_[:, 1] < Y1)]
wall = np.zeros((nx, ny), bool)
ij = np.floor((B_[:, :2] - [X0, Y0]) / RES).astype(int)
wall[ij[:, 0], ij[:, 1]] = True
wall = binary_dilation(wall, iterations=1)
near_a, near_b = binary_dilation(Ap, iterations=1), binary_dilation(Bp, iterations=1)
both = Bp & near_a & ~wall
new_sure = Bp & ~near_a & Af & ~wall
new_maybe = Bp & ~near_a & ~Af & ~wall
removed = Ap & ~near_b & Bf & ~wall
unknown = Ap & ~near_b & ~Bf & ~wall
area = lambda m: round(float(m.sum()) * RES * RES, 1)
stats = {'both_m2': area(both), 'new_floor_seen_m2': area(new_sure), 'only_B_m2': area(new_maybe), 'removed_m2': area(removed),
         'only_A_unseen_m2': area(unknown), 'transient_B_cells': int(Bt.sum()), 'transient_A_cells': int(At.sum())}
print(stats)
json.dump(stats, open(f'{work_b}/scan_diff.json', 'w'), indent=1)
np.savez_compressed(f'{work_b}/scan_diff.npz', both=both, new_sure=new_sure, new_maybe=new_maybe, removed=removed, unknown=unknown,
                    Bp=Bp, Bf=Bf, Ap=Ap, Af=Af, origin=[X0, Y0], res=RES)

# merged points: B everywhere it looked, A only where B saw nothing in that 10 cm cell (and never the removed furniture)
def keep_cells(P):
    q = np.floor((P[:, :2] - [X0, Y0]) / RES).astype(int)
    ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < nx) & (q[:, 1] < ny)
    return q, ok


qa, oka = keep_cells(Pa)
drop = np.zeros(len(Pa), bool)
hit = binary_dilation(removed, iterations=1)
drop[oka] = hit[qa[oka, 0], qa[oka, 1]]
Pm = np.concatenate([Pb, Pa[~drop]])
Cm = np.concatenate([Cb, Ca[~drop]])
z = (Pm[:, 2] > -0.05) & (Pm[:, 2] < 2.55)
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pm[z]))
pc.colors = o3d.utility.Vector3dVector(np.clip(Cm[z], 0, 1))
pc = pc.voxel_down_sample(0.015)
np.savez_compressed(f'{work_b}/merged_points.npz', points=np.asarray(pc.points, np.float32), colors=np.asarray(pc.colors, np.float32))
print(f'merged: {len(pc.points):,} points (1.5 cm), {int(drop.sum()):,} A points of removed things dropped')

# picture: 3x scale, y up
img = np.full((nx, ny, 3), 255, np.uint8)
img[Bf | Af] = (240, 240, 240)
for mask, colour in ((both, (120, 120, 120)), (new_sure, (40, 170, 60)), (new_maybe, (40, 120, 220)), (removed, (220, 40, 40)), (unknown, (250, 190, 90))):
    img[mask] = colour
S = 6
view = cv2.cvtColor(np.ascontiguousarray(np.rot90(img)), cv2.COLOR_RGB2BGR)
view = cv2.resize(view, (view.shape[1] * S, view.shape[0] * S), interpolation=cv2.INTER_NEAREST)
H = view.shape[0]
for xm in range(int(X0), int(X1) + 1):
    c = int((xm - X0) / RES * S)
    cv2.line(view, (c, 0), (c, H), (215, 215, 215), 1)
    cv2.putText(view, str(xm), (c + 2, H - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
for ym in range(int(Y0), int(Y1) + 1):
    r = int((Y1 - ym) / RES * S)
    cv2.line(view, (0, r), (view.shape[1], r), (215, 215, 215), 1)
    cv2.putText(view, str(ym), (2, r - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
# current furniture
SIZE = {'OfficeDesk': (1.4, 0.7), 'OfficeChair': (0.69, 0.72), 'SofaTable': (1.2, 0.8), 'Armchair': (0.71, 0.75), 'Chair6': (0.53, 0.47),
        'Desk6': (1.8, 0.8), 'TVStand': (1.45, 0.71), 'Plant': (0.4, 0.4)}
px = lambda x, y: (int((x - X0) / RES * S), int((Y1 - y) / RES * S))
place = json.load(open(f'{work_a}/placement_labeled.json'))
changed = {row[1]: row for row in place}
for it in json.load(open(f'{work_a}/furniture_v1.json')):
    name = it['path'].split('/')[-1]
    if it['room'] != 'Hall' or name in changed:
        continue
    cv2.rectangle(view, px(it['min'][0], it['max'][1]), px(it['max'][0], it['min'][1]), (160, 80, 160), 1)
for act, name, asset, x, y, yaw, k in place:
    if act == 'hide':
        continue
    w, dd = SIZE[asset]; c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    pts = [px(x + c * u - s * v, y + s * u + c * v) for u, v in ((-w/2, -dd/2), (w/2, -dd/2), (w/2, dd/2), (-w/2, dd/2))]
    cv2.polylines(view, [np.array(pts)], True, (200, 60, 0), 2)
    cv2.putText(view, name.replace('Scan_', '+'), (pts[0][0], pts[0][1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 60, 0), 1)
cv2.imwrite(f'{out_b}/scan_diff.png', view)
print('done')
