"""Sample the empty DX Hub building USD (walls / floor / columns) into a point cloud in the twin world frame."""
import sys
import numpy as np
from pxr import Usd, UsdGeom

usd, out = sys.argv[1], sys.argv[2]
stage = Usd.Stage.Open(usd)
cache = UsdGeom.XformCache()
rng = np.random.default_rng(0)
pts, names = [], {}
for prim in stage.Traverse():
    if not prim.IsA(UsdGeom.Mesh) or not UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.inherited:
        continue
    mesh = UsdGeom.Mesh(prim)
    v = np.asarray(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
    if len(v) == 0 or len(counts) == 0:
        continue
    m = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)   # row-vector convention
    v = v @ m[:3, :3] + m[3, :3]
    tris, start = [], 0
    for c in counts:                                   # fan-triangulate polygons
        f = idx[start:start + c]
        tris += [(f[0], f[k], f[k + 1]) for k in range(1, c - 1)]
        start += c
    t = np.asarray(tris)
    a, b, c3 = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c3 - a), axis=1)
    n = rng.poisson(area / 0.0025)                     # ~400 points / m^2 (5 cm spacing)
    if n.sum() == 0:
        continue
    k = np.repeat(np.arange(len(t)), n)
    r1, r2 = rng.random((2, len(k)))
    s = np.sqrt(r1)
    p = (1 - s)[:, None] * a[k] + (s * (1 - r2))[:, None] * b[k] + (s * r2)[:, None] * c3[k]
    pts.append(p)
    top = str(prim.GetPath()).split('/')[2] if str(prim.GetPath()).count('/') > 1 else str(prim.GetPath())
    names[top] = names.get(top, 0) + len(p)
P = np.concatenate(pts)
np.save(out, P.astype(np.float32))
print(f'{len(P):,} points  x {P[:,0].min():.2f}..{P[:,0].max():.2f}  y {P[:,1].min():.2f}..{P[:,1].max():.2f}  z {P[:,2].min():.2f}..{P[:,2].max():.2f}')
print(sorted(names.items(), key=lambda kv: -kv[1])[:12])
