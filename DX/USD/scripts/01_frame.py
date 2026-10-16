"""Stage 1: define the DX Hub world frame and register the Gaussian splat to it.

World frame: Z-up, meters, floor plane at z=0, walls aligned to X/Y, XY origin at
the centre of the scanned floor footprint.
"""
import json

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

from common import GSPLAT, LAS, TRAJ, WORK, apply, floor_plane, level_rotation, read_gs_ply, read_las, rot_z

WORK.mkdir(parents=True, exist_ok=True)
xyz, rec = read_las(LAS)
sub = xyz[::10].copy()
print(f"LAS points: {len(xyz):,}")

# 1) Level the LAS using its floor.
coef, res, ncell = floor_plane(sub)
R_level = np.eye(4)
R_level[:3, :3] = level_rotation(coef)
print(f"LAS floor tilt {np.degrees(np.arctan(np.hypot(*coef[:2]))):.3f} deg (residual {res*100:.1f} cm, {ncell} cells)")
P = apply(R_level, sub)

# 2) Yaw: align dominant wall normals with the X/Y axes.
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[::3])).voxel_down_sample(0.05)
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
n = np.asarray(pc.normals)
v = n[np.abs(n[:, 2]) < 0.1]
ang = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
# circular mean on 4*theta gives the axis direction mod 90
theta = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(4 * ang))))) / 4
# Pick the multiple of 90 deg closest to the splat's native yaw (+12.4 deg) so both frames agree.
cands = [-theta + k * 90 for k in range(-2, 3)]
yaw = min(cands, key=lambda c: abs(c - 12.45))
P = apply(rot_z(yaw), P)
R1 = rot_z(yaw) @ R_level

# 3) Floor to z=0, XY origin at footprint centre.
coef2, res2, _ = floor_plane(P)
floor_pts = P[np.abs(P[:, 2] - coef2[2]) < 0.08]
cell = 0.25
gi = np.floor(floor_pts[:, :2] / cell).astype(int)
lo = gi.min(0)
H = np.zeros(gi.max(0) - lo + 1, np.int32)
np.add.at(H, tuple((gi - lo).T), 1)
dense = H >= 15
lab, nl = ndimage.label(dense)
sizes = ndimage.sum(dense, lab, range(1, nl + 1))
main = lab == (np.argmax(sizes) + 1)
ij = np.argwhere(main)
xy_min = (ij.min(0) + lo) * cell
xy_max = (ij.max(0) + lo + 1) * cell
centre = np.round((xy_min + xy_max) / 2, 1)
T_shift = np.eye(4)
T_shift[:2, 3] = -centre
T_shift[2, 3] = -coef2[2]
T_las2w = T_shift @ R1
print(f"yaw {yaw:.3f} deg, floor offset {coef2[2]:.4f} m, residual tilt "
      f"{np.degrees(np.arctan(np.hypot(*coef2[:2]))):.4f} deg, footprint {xy_max - xy_min} m")

# 4) Register the Gaussian splat: 2D occupancy search -> robust 6-DOF ICP -> scale search.
#    (The splat's own floor is too fuzzy to level it independently; walls/ceiling constrain the tilt.)
_, props, gs = read_gs_ply(GSPLAT)
c = {p: i for i, p in enumerate(props)}
G = gs[:, [c["x"], c["y"], c["z"]]].astype(np.float64)
op = 1 / (1 + np.exp(-gs[:, c["opacity"]]))
smax = np.exp(gs[:, [c["scale_0"], c["scale_1"], c["scale_2"]]]).max(1)
src = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(G[(op > 0.8) & (smax < 0.03)]))
                 .voxel_down_sample(0.03).points)

W = apply(T_las2w, sub)
tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W)).voxel_down_sample(0.03)
tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
Q, Nq = np.asarray(tgt.points), np.asarray(tgt.normals)
tree = cKDTree(Q)


def occupancy(P2, res=0.2, size=512):
    q = np.floor(P2 / res).astype(int) + size // 2
    q = q[(q >= 0).all(1) & (q < size).all(1)]
    im = np.zeros((size, size))
    np.add.at(im, (q[:, 0], q[:, 1]), 1)
    return np.minimum(im, 5)


Ws = W[np.abs(W[:, 2] - 1.0) < 0.8]
F_w = np.fft.fft2(occupancy(Ws[:, :2]))
gs_ctr = np.median(src, 0)
best = None
for yaw_g in np.arange(-180, 180, 0.5):
    S2 = (src[np.abs(src[:, 2] - np.median(src[:, 2])) < 0.8] - gs_ctr)[:, :2] @ rot_z(yaw_g)[:2, :2].T
    corr = np.real(np.fft.ifft2(F_w * np.conj(np.fft.fft2(occupancy(S2)))))
    k = np.unravel_index(corr.argmax(), corr.shape)
    if best is None or corr[k] > best[0]:
        best = (corr[k], yaw_g, k)
_, yaw_g, k = best
shift = np.array([(i + 256) % 512 - 256 for i in k]) * 0.2
T_init = rot_z(yaw_g)
T_init[:2, 3] = shift - T_init[:2, :2] @ gs_ctr[:2]
T_init[2, 3] = np.percentile(W[:, 2], 1) - np.percentile(src[:, 2], 1)   # match floors (LAS median is ceiling-heavy)
print(f"splat coarse yaw {yaw_g:.1f} deg, shift {shift}")


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def exp_so3(w):
    t = np.linalg.norm(w)
    if t < 1e-12:
        return np.eye(3)
    K = skew(w / t)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def icp6(S, T, dmax, iters=50):
    """Point-to-plane ICP with IRLS (L1-like) weights."""
    for _ in range(iters):
        X = apply(T, S)
        d, k = tree.query(X, distance_upper_bound=dmax)
        ok = np.isfinite(d)
        x, q, n = X[ok], Q[k[ok]], Nq[k[ok]]
        r = np.einsum("ij,ij->i", n, x - q)
        w = 1 / np.maximum(np.abs(r), 0.01)
        J = np.c_[np.cross(x, n), n]
        delta = np.linalg.solve((J * w[:, None]).T @ J + 1e-9 * np.eye(6), -(J * w[:, None]).T @ r)
        dT = np.eye(4)
        dT[:3, :3] = exp_so3(delta[:3])
        dT[:3, 3] = delta[3:]
        T = dT @ T
        if np.abs(delta).max() < 1e-7:
            break
    return T


def inliers(T, S):
    d, _ = tree.query(apply(T, S))
    return np.mean(d < 0.05), np.median(d)


T = T_init
for dmax in [1.0, 0.5, 0.3, 0.15, 0.08, 0.05]:
    T = icp6(src, T, dmax)
print("splat 6-DOF ICP: inliers<5cm %.3f, median %.2f cm" % (lambda a: (a[0], a[1] * 100))(inliers(T, src)))

# The splat is reconstructed at a slightly different scale than the LiDAR map; search it.
results = []
for s in np.arange(0.990, 1.0061, 0.001):
    S_mat = np.eye(4)                         # uniform scale about the splat centroid
    S_mat[:3, :3] *= s
    S_mat[:3, 3] = gs_ctr * (1 - s)
    src_s = apply(S_mat, src)
    Ts = T
    for dmax in [0.15, 0.08, 0.05]:
        Ts = icp6(src_s, Ts, dmax, iters=30)
    results.append((inliers(Ts, src_s)[0], s, Ts @ S_mat))
fit, scale, T_gs2w = max(results, key=lambda r: r[0])
print("scale search: " + ", ".join(f"{s:.3f}:{f:.3f}" for f, s, _ in results))
fit, med = inliers(T_gs2w, src)
R = T_gs2w[:3, :3] / scale
print(f"T_gs->world: scale {scale:.3f}, yaw {np.degrees(np.arctan2(R[1, 0], R[0, 0])):.3f} deg, "
      f"tilt {np.degrees(np.arccos(min(R[2, 2], 1.0))):.3f} deg, t {T_gs2w[:3, 3].round(3)}; "
      f"inliers<5cm {fit:.3f}, median {med * 100:.2f} cm")

traj_xyz, _ = read_las(TRAJ)
json.dump({
    "T_las_to_world": T_las2w.tolist(),
    "T_gsplat_to_world": T_gs2w.tolist(),
    "gsplat_scale": float(scale),
    "gsplat_inliers_5cm": float(fit),
    "gsplat_median_dist_m": float(med),
    "footprint_min": (xy_min - centre).tolist(),
    "footprint_max": (xy_max - centre).tolist(),
    "las_floor_tilt_deg": float(np.degrees(np.arctan(np.hypot(*coef[:2])))),
    "yaw_deg": float(yaw),
}, open(WORK / "frame.json", "w"), indent=2)
np.save(WORK / "traj_world.npy", apply(T_las2w, traj_xyz))
