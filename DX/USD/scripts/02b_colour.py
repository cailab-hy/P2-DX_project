"""Stage 2b: colour the visual mesh from the registered Gaussian splat.

Each vertex takes the colour (SH DC term) of nearby Gaussians weighted by their opacity times their
anisotropic density at the vertex, exp(-0.5 * Mahalanobis^2), so large flat floor/wall splats count
where they actually cover the surface. Where the splat gives little support (areas it was not
trained on) the colour is blended toward neutral grey instead of inheriting floater colours.
"""
import json

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from common import GSPLAT, WORK, apply, read_gs_ply

GREY = 0.55          # sRGB
SUPPORT = 0.3        # accumulated alpha at which a vertex is ~63% splat colour
K, RADIUS, CHUNK = 32, 0.6, 200_000

fr = json.load(open(WORK / "frame.json"))
T = np.array(fr["T_gsplat_to_world"])
s_g = fr["gsplat_scale"]
_, props, gs = read_gs_ply(GSPLAT)
c = {p: i for i, p in enumerate(props)}
op = 1 / (1 + np.exp(-gs[:, c["opacity"]]))
scale = np.exp(gs[:, [c["scale_0"], c["scale_1"], c["scale_2"]]]) * s_g
sel = (op > 0.1) & (scale.max(1) < 1.0)
op, scale = op[sel], np.maximum(scale[sel], 1e-3)
mu = apply(T, gs[sel][:, [c["x"], c["y"], c["z"]]].astype(np.float64))
srgb = np.clip(0.5 + 0.28209479177387814 * gs[sel][:, [c["f_dc_0"], c["f_dc_1"], c["f_dc_2"]]], 0, 1)
q = gs[sel][:, [c["rot_0"], c["rot_1"], c["rot_2"], c["rot_3"]]].astype(np.float64)
w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
              np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
              np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)
R = (T[:3, :3] / s_g) @ R                                  # rotate the kernels into the world frame
tree = cKDTree(mu)

mesh = o3d.io.read_triangle_mesh(str(WORK / "mesh_visual.ply"))
V = np.asarray(mesh.vertices)
col = np.empty((len(V), 3))
acc = np.empty(len(V))
for a in range(0, len(V), CHUNK):
    v = V[a:a + CHUNK]
    d, k = tree.query(v, k=K, distance_upper_bound=RADIUS, workers=-1)
    ok = np.isfinite(d)
    k = np.where(ok, k, 0)
    local = np.einsum("nkji,nkj->nki", R[k], v[:, None, :] - mu[k]) / scale[k]   # R^T (v - mu) / s
    alpha = np.where(ok, op[k] * np.exp(-0.5 * (local ** 2).sum(-1)), 0.0)
    acc[a:a + CHUNK] = alpha.sum(1)
    col[a:a + CHUNK] = (alpha[..., None] * srgb[k]).sum(1) / np.maximum(alpha.sum(1), 1e-12)[:, None]
conf = 1 - np.exp(-acc / SUPPORT)
col = conf[:, None] * col + (1 - conf[:, None]) * GREY
lin = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4)      # sRGB -> linear
mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(lin, 0, 1))
o3d.io.write_triangle_mesh(str(WORK / "mesh_visual.ply"), mesh)
print(f"coloured {len(V):,} vertices; confidence >0.5 on {(conf > 0.5).mean() * 100:.1f}%, "
      f"<0.1 on {(conf < 0.1).mean() * 100:.1f}%")
for name, (x0, x1, y0, y1) in {"hall": (-12, -2, 5, 18), "east_room": (7, 12, 9, 18),
                               "south_corridor": (4, 6, -20, -14)}.items():
    m = (V[:, 0] > x0) & (V[:, 0] < x1) & (V[:, 1] > y0) & (V[:, 1] < y1)
    print(f"  {name}: median confidence {np.median(conf[m]):.2f}")
