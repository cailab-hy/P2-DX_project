"""Stage 2: reconstruct a coloured triangle mesh of the DX Hub from the LiDAR map.

LAS (world frame) -> interior footprint crop -> 2 cm voxel grid -> normals oriented toward the
scanner pose at capture time -> screened Poisson -> trim unsupported surface -> decimate (visual / collision).
"""
import json
import time

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

from common import LAS, TRAJ, WORK, apply, read_las

VOXEL = 0.02
POISSON_DEPTH = 12
VISUAL_TRIS = 3_000_000
COLLISION_TRIS = 400_000
t0 = time.time()
log = lambda m: print(f"[{time.time() - t0:7.1f}s] {m}", flush=True)

fr = json.load(open(WORK / "frame.json"))
T_las2w = np.array(fr["T_las_to_world"])
T_gs2w = np.array(fr["T_gsplat_to_world"])

xyz, rec = read_las(LAS)
P = apply(T_las2w, xyz).astype(np.float32)
t_pt = np.asarray(rec["t"])
del xyz
log(f"loaded {len(P):,} points")

# --- interior footprint mask (removes points seen through glass / doors) --------------------
cell = 0.25
floor = P[np.abs(P[:, 2]) < 0.08]
lo = np.floor(P[:, :2].min(0) / cell).astype(int)
shape = np.floor(P[:, :2].max(0) / cell).astype(int) - lo + 1
H = np.zeros(shape, np.int32)
np.add.at(H, tuple((np.floor(floor[:, :2] / cell).astype(int) - lo).T), 1)
dense = H >= 15
lab, nl = ndimage.label(dense)
main = lab == (np.argmax(ndimage.sum(dense, lab, range(1, nl + 1))) + 1)
main = ndimage.binary_closing(main, iterations=4)
main = ndimage.binary_fill_holes(main)
mask = ndimage.binary_dilation(main, iterations=2)          # +0.5 m to keep the walls
np.save(WORK / "interior_mask.npy", main)
ij = np.floor(P[:, :2] / cell).astype(int) - lo
inside = mask[ij[:, 0], ij[:, 1]]
zmax = np.percentile(P[inside, 2], 99.95) + 0.1
keep = inside & (P[:, 2] > -0.15) & (P[:, 2] < zmax)
P, t_pt = P[keep], t_pt[keep]
np.save(WORK / "footprint_mask.npy", mask)
json.dump({"cell": cell, "lo": lo.tolist()}, open(WORK / "footprint_mask.json", "w"))
log(f"interior crop: {keep.sum():,} pts ({keep.mean() * 100:.1f}%), z in [-0.15, {zmax:.2f}]")

# --- 2 cm voxel grid (mean position, first timestamp) ----------------------------------------
v = np.floor(P / VOXEL).astype(np.int64)
v -= v.min(0)
key = (v[:, 0] * (v[:, 1].max() + 1) + v[:, 1]) * (v[:, 2].max() + 1) + v[:, 2]
order = np.argsort(key, kind="stable")
_, first, inv_counts = np.unique(key[order], return_index=True, return_counts=True)
sums = np.add.reduceat(P[order].astype(np.float64), first, axis=0)
pts = sums / inv_counts[:, None]
tvox = t_pt[order][first]
del v, key, order, sums, P, t_pt
log(f"voxelised: {len(pts):,} pts")

pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
pc, ind = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
tvox = tvox[np.asarray(ind)]
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.08, max_nn=30))

# The handheld scanner's operator hides the floor right under the walked path, leaving holes.
# Fill the interior floor at z=0 (the fitted floor plane) wherever no real floor sample exists.
FLOOR_STEP = 0.04
real = np.asarray(pc.points)
real_floor = real[np.abs(real[:, 2]) < 0.06]
ij_in = np.argwhere(main)
sub_off = (np.arange(int(cell / FLOOR_STEP)) + 0.5) * FLOOR_STEP
gx, gy = np.meshgrid(sub_off, sub_off, indexing="ij")
cand = ((ij_in + lo)[:, None, :] * cell + np.c_[gx.ravel(), gy.ravel()][None]).reshape(-1, 2)
d, _ = cKDTree(real_floor[:, :2]).query(cand, distance_upper_bound=0.05, workers=-1)
fill = cand[~np.isfinite(d)]
fill3 = np.c_[fill, np.zeros(len(fill))]
pc_fill = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fill3))
pc_fill.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(fill), 1)))
log(f"floor fill: {len(fill):,} synthetic floor samples ({len(fill) * FLOOR_STEP ** 2:.1f} m2 of "
    f"{main.sum() * cell ** 2:.0f} m2 interior)")
traj_w = np.load(WORK / "traj_world.npy")
_, trec = read_las(TRAJ)
tt = np.asarray(trec["t"])
o = np.argsort(tt)
sensor = np.stack([np.interp(tvox, tt[o], traj_w[o, i]) for i in range(3)], 1)
pts = np.asarray(pc.points)
nrm = np.asarray(pc.normals)
flip = np.einsum("ij,ij->i", nrm, sensor - pts) < 0
nrm[flip] *= -1
pc.normals = o3d.utility.Vector3dVector(nrm)
log(f"normals oriented to scanner ({flip.mean() * 100:.1f}% flipped), {len(pts):,} pts after SOR")
pc += pc_fill
pts = np.asarray(pc.points)

# --- Poisson + trimming ----------------------------------------------------------------------
with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Warning):
    mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc, depth=POISSON_DEPTH, n_threads=-1)
dens = np.asarray(dens)
log(f"poisson: {len(mesh.triangles):,} tris")
tree = cKDTree(pts)
dist, _ = tree.query(np.asarray(mesh.vertices), workers=-1)
mesh.remove_vertices_by_mask((dist > 2.5 * VOXEL) | (dens < np.quantile(dens, 0.02)))
cl, n_tri, _ = mesh.cluster_connected_triangles()
cl, n_tri = np.asarray(cl), np.asarray(n_tri)
mesh.remove_triangles_by_mask(n_tri[cl] < 500)
mesh.remove_unreferenced_vertices()
mesh.remove_degenerate_triangles()
log(f"trimmed: {len(mesh.triangles):,} tris, {len(mesh.vertices):,} verts")
o3d.io.write_triangle_mesh(str(WORK / "mesh_full.ply"), mesh)

# --- decimate --------------------------------------------------------------------------------
vis = mesh.simplify_quadric_decimation(VISUAL_TRIS) if len(mesh.triangles) > VISUAL_TRIS else mesh
vis.remove_unreferenced_vertices()
vis.compute_vertex_normals()
o3d.io.write_triangle_mesh(str(WORK / "mesh_visual.ply"), vis)
log(f"visual mesh: {len(vis.triangles):,} tris (colours are added by 02b_colour.py)")

col = mesh.simplify_quadric_decimation(COLLISION_TRIS)
col.remove_unreferenced_vertices()
col.remove_degenerate_triangles()
col.compute_vertex_normals()
V = np.asarray(col.vertices).copy()
N = np.asarray(col.vertex_normals)
snap = (np.abs(V[:, 2]) < 0.04) & (N[:, 2] > 0.9)         # flatten the floor for stable contacts
V[snap, 2] = 0.0
col.vertices = o3d.utility.Vector3dVector(V)
o3d.io.write_triangle_mesh(str(WORK / "mesh_collision.ply"), col)
log(f"collision mesh: {len(col.triangles):,} tris, {snap.sum():,} floor verts snapped to z=0")
json.dump({"zmax": float(zmax), "visual_tris": len(vis.triangles), "collision_tris": len(col.triangles),
           "full_tris": len(mesh.triangles)},
          open(WORK / "mesh_stats.json", "w"), indent=2)
