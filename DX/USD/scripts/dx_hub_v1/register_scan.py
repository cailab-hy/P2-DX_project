"""Register the 2026-10-30 DX Hub rescan (FJD LAS) to the twin world frame (the empty building USD, floor z = 0).

1. read the LAS (38-byte records: point format 3 + 4 extra bytes), 5 cm voxels
2. level: RANSAC floor plane -> normal to +Z, floor at z = 0
3. 2D: occupancy of the 1.2-2.2 m band (walls, columns, glass; above most furniture) for scan and building,
   yaw search 0..360 deg with FFT cross-correlation for the shift, refined to 0.1 deg
4. 3D point-to-plane ICP on the 0.3-2.5 m band (0.3 -> 0.1 -> 0.05 m)
Writes work/T_scan_to_world.npy (+ json) and a top-down overlay png.
    python register_scan.py LAS building_points.npy WORKDIR
"""
import json
import struct
import sys

import numpy as np
import open3d as o3d
from scipy.signal import fftconvolve

from scanio import read_las

las, bpath, work = sys.argv[1:4]
RES = 0.10


def grid(points, lo, shape):
    g = np.zeros(shape, np.float32)
    ij = np.floor((points[:, :2] - lo) / RES).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.add.at(g, (ij[ok, 0], ij[ok, 1]), 1.0)
    return np.minimum(g, 3.0)


def rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


xyz, rgb = read_las(las)
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
pc.colors = o3d.utility.Vector3dVector(rgb)
pc = pc.voxel_down_sample(0.05)
P = np.asarray(pc.points)
print(f'scan {len(xyz):,} -> {len(P):,} points (5 cm)')

# 2) level on the floor (lowest dense layer)
low = P[P[:, 2] < np.percentile(P[:, 2], 25)]
plane, inl = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(low)).segment_plane(0.03, 3, 2000)
n = np.array(plane[:3]); n = n if n[2] > 0 else -n
v = np.cross(n, [0, 0, 1.0]); s, c = np.linalg.norm(v), n[2]
K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
R_level = np.eye(3) + K + K @ K * ((1 - c) / s ** 2) if s > 1e-9 else np.eye(3)
L = P @ R_level.T
floor_z = np.median(L[np.abs(L @ np.array([0, 0, 1.0]) - np.median((low[inl] @ R_level.T)[:, 2])) < 0.05][:, 2])
L[:, 2] -= floor_z
print(f'floor tilt {np.degrees(np.arccos(c)):.2f} deg, floor inliers {len(inl):,}')

# 3) 2D yaw + shift on the wall band
B = np.load(bpath).astype(np.float64)
band = lambda Q, lo, hi: Q[(Q[:, 2] > lo) & (Q[:, 2] < hi)]
Bb, Sb = band(B, 1.2, 2.2), band(L, 1.2, 2.2)
blo = Bb[:, :2].min(0) - 2.0
bshape = tuple(np.ceil((Bb[:, :2].max(0) + 2.0 - blo) / RES).astype(int))
Gb = grid(Bb, blo, bshape)
Sc = Sb[:, :2].mean(0)


def score(yaw):
    S = (Sb - np.r_[Sc, 0]) @ rz(yaw).T
    slo = S[:, :2].min(0)
    sshape = tuple(np.ceil((S[:, :2].max(0) - slo) / RES).astype(int) + 1)
    Gs = grid(S, slo, sshape)
    corr = fftconvolve(Gb, Gs[::-1, ::-1], mode='full')
    k = np.unravel_index(np.argmax(corr), corr.shape)
    shift = (np.array(k) - (np.array(sshape) - 1)) * RES + blo - slo   # world xy of the rotated, centred scan
    return corr[k], shift


best = max(((score(y), y) for y in range(0, 360, 2)), key=lambda t: t[0][0])
for step in (0.5, 0.1):
    y0 = best[1]
    best = max(((score(y), y) for y in np.arange(y0 - 4 * step, y0 + 4.01 * step, step)), key=lambda t: t[0][0])
(peak, shift), yaw = best
T2 = np.eye(4); T2[:3, :3] = rz(yaw); T2[:3, 3] = np.r_[shift - rz(yaw)[:2, :2] @ Sc, 0]
print(f'2D: yaw {yaw:.1f} deg, shift {shift.round(2)}, peak {peak:.0f}')

# 4) ICP
T_level = np.eye(4); T_level[:3, :3] = R_level; T_level[2, 3] = -floor_z
src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(band(L, 0.3, 2.5)))
dst = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(band(B, 0.3, 2.5)))
dst.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
T = T2
for d in (0.3, 0.1, 0.05):
    r = o3d.pipelines.registration.registration_icp(src, dst, d, T,
                                                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    T = r.transformation
    print(f'ICP {d:.2f} m: fitness {r.fitness:.3f}, rmse {r.inlier_rmse * 100:.1f} cm')
T_full = T @ T_level
np.save(f'{work}/T_scan_to_world.npy', T_full)
yaw_final = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
json.dump({'T_scan_to_world': T_full.tolist(), 'yaw_deg': yaw_final, 'icp_fitness_5cm': r.fitness,
           'icp_rmse_cm': r.inlier_rmse * 100, 'floor_tilt_deg': float(np.degrees(np.arccos(c)))},
          open(f'{work}/T_scan_to_world.json', 'w'), indent=1)

# overlay png (building band grey, aligned scan band red)
W = (np.c_[P, np.ones(len(P))] @ T_full.T)[:, :3]
img_lo = Bb[:, :2].min(0) - 3; img_hi = Bb[:, :2].max(0) + 3
shape = tuple(np.ceil((img_hi - img_lo) / 0.05).astype(int))
def mask(Q):
    m = np.zeros(shape, bool); ij = np.floor((Q[:, :2] - img_lo) / 0.05).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1]); m[ij[ok, 0], ij[ok, 1]] = True; return m
mb, ms = mask(Bb), mask(band(W, 1.2, 2.2))
img = np.full(shape + (3,), 255, np.uint8); img[mb] = (150, 150, 150); img[ms] = (220, 30, 30); img[mb & ms] = (40, 40, 40)
o3d.io.write_image(f'{work}/register_overlay.png', o3d.geometry.Image(np.ascontiguousarray(np.rot90(img))))
print('done')
