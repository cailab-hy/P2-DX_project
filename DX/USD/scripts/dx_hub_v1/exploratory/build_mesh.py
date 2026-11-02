"""Coloured surface mesh of the registered DX Hub rescan, in the twin world frame (quick version, option 1).

points (2 cm) -> crop to the building + 1 m, z -0.05..CEIL -> normals facing the scanner trajectory -> Poisson
(depth 10) -> drop low-density / far-from-scan triangles -> visual mesh (~1.5 M tris) + collision mesh (~0.2 M tris)
-> work/mesh_visual.npz, work/mesh_collision.npz (vertices, triangles, colours) for write_usd.py.
    python build_mesh.py LAS TRAJ_LAS WORKDIR [ceiling_cut_m] [poisson_depth] [threads]
"""
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

sys.path.insert(0, __file__.rsplit('/', 1)[0])
from scanio import read_las  # noqa: E402

las, traj, work = sys.argv[1:4]
CEIL = float(sys.argv[4]) if len(sys.argv) > 4 else 2.55   # cut the ceiling: the twin keeps the building's
DEPTH = int(sys.argv[5]) if len(sys.argv) > 5 else 10       # depth 11 + all threads segfaulted (open3d 0.20, aarch64)
THREADS = int(sys.argv[6]) if len(sys.argv) > 6 else -1
t0 = time.time()
T = np.load(f'{work}/T_scan_to_world.npy')
B = np.load(f'{work}/building_points.npy')
lo, hi = B[:, :2].min(0) - 1.0, B[:, :2].max(0) + 1.0

xyz, rgb = read_las(las)
W = xyz @ T[:3, :3].T + T[:3, 3]
keep = (W[:, 0] > lo[0]) & (W[:, 0] < hi[0]) & (W[:, 1] > lo[1]) & (W[:, 1] < hi[1]) & (W[:, 2] > -0.05) & (W[:, 2] < CEIL)
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(W[keep]))
pc.colors = o3d.utility.Vector3dVector(rgb[keep])
pc = pc.voxel_down_sample(0.02)
pc, _ = pc.remove_statistical_outlier(20, 2.5)
print(f'{keep.sum():,} points in the building -> {len(pc.points):,} (2 cm, outliers removed)  {time.time() - t0:.0f} s', flush=True)

pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))
tx, _ = read_las(traj)
TW = tx @ T[:3, :3].T + T[:3, 3]
TW[:, 2] = np.maximum(TW[:, 2], 0.0) + 1.0                       # scanner head height above the trajectory points
P, N = np.asarray(pc.points), np.asarray(pc.normals)
_, k = cKDTree(TW).query(P)
flip = np.einsum('ij,ij->i', N, TW[k] - P) < 0
N[flip] *= -1
pc.normals = o3d.utility.Vector3dVector(N)
print(f'normals oriented to {len(TW):,} trajectory poses  {time.time() - t0:.0f} s', flush=True)

mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=DEPTH, n_threads=THREADS)
dens = np.asarray(dens)
mesh.remove_vertices_by_mask(dens < np.quantile(dens, 0.04))
d, _ = cKDTree(P).query(np.asarray(mesh.vertices))
mesh.remove_vertices_by_mask(d > 0.06)                             # Poisson bubbles where nothing was scanned
V = np.asarray(mesh.vertices)
mesh.remove_vertices_by_mask((V[:, 2] < -0.05) | (V[:, 2] > CEIL))
mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
print(f'poisson: {len(mesh.triangles):,} triangles  {time.time() - t0:.0f} s', flush=True)


def save(m, name):
    np.savez_compressed(f'{work}/{name}.npz', vertices=np.asarray(m.vertices, np.float32),
                        triangles=np.asarray(m.triangles, np.int32), colors=np.asarray(m.vertex_colors, np.float32))
    print(f'{name}: {len(m.vertices):,} vertices, {len(m.triangles):,} triangles', flush=True)


visual = mesh.simplify_quadric_decimation(1_500_000) if len(mesh.triangles) > 1_500_000 else mesh
save(visual, 'mesh_visual')
collision = mesh.simplify_quadric_decimation(200_000)
save(collision, 'mesh_collision')
o3d.io.write_triangle_mesh(f'{work}/mesh_visual.ply', visual)
print(f'done {time.time() - t0:.0f} s')
