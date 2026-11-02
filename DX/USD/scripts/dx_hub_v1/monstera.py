"""Potted monstera plants by the two ends of the diagonal wall (user's photo, 2026-11-02): pots fitted to the points, leaves placed on them.

Foliage points (building removed, z above the pots) are grouped with k-means; each group gives one leaf: centre, direction and size from its
principal axes, on a stem from the nearest pot. The leaf shape itself comes from the photo, not from the points.
    python monstera.py WORK_B WORK_A   ->  WORK_B/monstera.json
place_final.py calls add() with that file.
"""
import json
import sys

import numpy as np

# site: centre, foliage radius, pots (start x, y, top z read from the side views), highest foliage z, x limit (points behind a counter's face don't count)
SITES = [((-6.0, -0.2), 0.75, [(-5.93, -0.47, 0.60)], 1.75, 9.0), ((-3.42, -3.6), 0.62, [(-3.47, -3.60, 0.42)], 1.75, 9.0),
         ((-0.98, 5.0), 0.42, [(-0.95, 5.0, 0.55)], 1.02, -0.78), ((-0.95, 10.33), 0.45, [(-0.95, 10.33, 0.55)], 1.14, -0.78)]
TRUNC = 0.03


# small flower pot on the kitchen counter: x, y, counter top z, flower top z (points y 9.7..10.02, z 1.17..1.35)
FLOWERS = [(-0.61, 9.86, 1.16, 1.35)]


def measure(work_b, work_a):
    from scipy.cluster.vq import kmeans2
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree

    P = np.load(f'{work_b}/merged_points.npz')['points']
    B = np.load(f'{work_a}/building_points.npy')
    out = {'pots': []}
    np.random.seed(7)
    for (cx, cy), radius, pots, zmax, xlim in SITES:
        k = (np.hypot(P[:, 0] - cx, P[:, 1] - cy) < radius) & (P[:, 2] > 0.02) & (P[:, 2] < zmax) & (P[:, 0] < xlim)
        q = P[k]
        b = B[(np.hypot(B[:, 0] - cx, B[:, 1] - cy) < radius + 0.2)]
        d, _ = cKDTree(b).query(q, k=1)
        q = q[d > 0.06]
        fitted = []
        for x0, y0, top in pots:
            low = q[(q[:, 2] > 0.05) & (q[:, 2] < top - 0.03)]
            low = low[np.hypot(low[:, 0] - x0, low[:, 1] - y0) < 0.35]
            best = None
            for r in np.arange(0.10, 0.27, 0.01):
                def cost(p):
                    return np.minimum(np.abs(np.hypot(low[:, 0] - p[0], low[:, 1] - p[1]) - r), TRUNC).mean()
                s = minimize(cost, (x0, y0), method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-5})
                if best is None or s.fun < best[0]:
                    best = (s.fun, s.x[0], s.x[1], r)
            print(f'pot ({best[1]:.3f}, {best[2]:.3f}) r {best[3]:.2f} top {top}, {best[0] * 100:.2f} cm, {len(low)} points')
            fitted.append({'x': round(float(best[1]), 3), 'y': round(float(best[2]), 3), 'r': round(float(best[3]), 2), 'top': top, 'leaves': []})
        ztop = max(p[2] for p in pots)
        f = q[q[:, 2] > ztop + 0.12]
        n = int(np.clip(len(f) / 110, 4, 9))
        cent, lab = kmeans2(f.astype(np.float64), n, minit='++', iter=30, seed=7)
        for i in range(n):
            g = f[lab == i]
            if len(g) < 25:
                continue
            c = g.mean(0)
            w, v = np.linalg.eigh(np.cov((g - c).T))
            axis, side, normal = v[:, 2], v[:, 1], v[:, 0]
            if normal[2] < 0:
                normal = -normal
            s = np.sqrt(w)
            length, width = float(np.clip(3.2 * s[2], 0.16, 0.36)), float(np.clip(3.2 * s[1], 0.12, 0.30))
            pot = min(fitted, key=lambda p: np.hypot(p['x'] - c[0], p['y'] - c[1]))
            out_dir = np.array([c[0] - pot['x'], c[1] - pot['y'], 0.0])
            if axis @ out_dir < 0:      # leaf tip away from the pot
                axis = -axis
            pot['leaves'].append({'c': np.round(c, 3).tolist(), 'axis': np.round(axis, 3).tolist(), 'normal': np.round(normal, 3).tolist(), 'len': round(length, 3), 'width': round(width, 3), 'n': int(len(g))})
        for p in fitted:   # the groups can't tell the plants apart where the foliage mixes: at least 3 leaves per pot, the closest ones
            while len(p['leaves']) < 3 and len(fitted) > 1:
                others = [(o, lf) for o in fitted if o is not p and len(o['leaves']) > 3 for lf in o['leaves']]
                o, lf = min(others, key=lambda e: np.hypot(e[1]['c'][0] - p['x'], e[1]['c'][1] - p['y']))
                o['leaves'].remove(lf)
                p['leaves'].append(lf)
        for p in fitted:
            print(f"  pot at ({p['x']}, {p['y']}): {len(p['leaves'])} leaves")
        out['pots'] += fitted
    json.dump(out, open(f'{work_b}/monstera.json', 'w'), indent=1)


def leaf_mesh(c, axis, normal, length, width):
    """Split monstera leaf: midrib along axis, lobes with notches on both sides, drooping tip."""
    axis, normal = np.array(axis, float), np.array(normal, float)
    axis = axis - normal * (axis @ normal)
    axis /= np.linalg.norm(axis)
    side = np.cross(normal, axis)
    base = np.array(c, float) - axis * length * 0.45
    profile = lambda s: np.sin(np.pi * np.clip(s, 0, 1) ** 0.72) ** 0.8
    ss = np.linspace(0, 1, 25)
    pts, counts, idx = [], [], []

    def point(s, u):
        return base + axis * (s * length) + side * (u * width / 2) - normal * (0.22 * length * (s - 0.35) ** 2 + 0.05 * length * abs(u))

    for s in ss:
        pts.append(point(s, 0.0))
    for sgn in (1, -1):
        start = len(pts)
        for i, s in enumerate(ss):
            notch = 0.45 if i % 4 == 2 and 0.1 < s < 0.92 else 1.0
            pts.append(point(s, sgn * profile(s) * notch))
        for i in range(len(ss) - 1):
            a, b, c_, d = i, i + 1, start + i + 1, start + i
            counts.append(4)
            idx += [a, b, c_, d] if sgn > 0 else [b, a, d, c_]
    return pts, counts, idx, base


def add(stage, hall, data):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_Monstera_Looks')

    def material(name, rgb, rough=0.5):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    white, dark, green, stem = material('Pot', (0.9, 0.9, 0.87), 0.25), material('Soil', (0.03, 0.03, 0.03), 0.9), material('Leaf', (0.05, 0.26, 0.08), 0.45), material('Stem', (0.16, 0.34, 0.1), 0.5)

    def bind(prim, mat):
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)

    def rod(path, p0, p1, r, mat):
        p0, p1 = Gf.Vec3d(*map(float, p0)), Gf.Vec3d(*map(float, p1))
        d = p1 - p0
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateAxisAttr('Z')
        c.CreateRadiusAttr(r)
        c.CreateHeightAttr(d.GetLength())
        c.AddTranslateOp().Set((p0 + p1) / 2)
        c.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), d).GetQuat()))
        bind(c, mat)

    for n, p in enumerate(data['pots'], 1):
        root = f'{hall}/Scan_Monstera_{n}'
        UsdGeom.Xform.Define(stage, root)
        org = np.array([p['x'], p['y'], 0.0])
        r, top = p['r'], p['top']
        a = np.linspace(0, 2 * np.pi, 40, endpoint=False)
        secs = [(0.0, r * 0.74), (top - 0.012, r), (top, r), (top, r - 0.012), (top - 0.03, r - 0.016)]
        pts = [Gf.Vec3f(*map(float, (org[0] + q * np.cos(t), org[1] + q * np.sin(t), z))) for z, q in secs for t in a]
        m_ = 40
        last = m_ * (len(secs) - 1)
        counts, idx = [m_, m_], list(range(m_ - 1, -1, -1)) + list(range(last, last + m_))
        for s in range(len(secs) - 1):
            for i in range(m_):
                j = (i + 1) % m_
                counts.append(4)
                idx += [s * m_ + i, s * m_ + j, (s + 1) * m_ + j, (s + 1) * m_ + i]
        pot = UsdGeom.Mesh.Define(stage, f'{root}/pot')
        pot.CreatePointsAttr(pts)
        pot.CreateFaceVertexCountsAttr(counts)
        pot.CreateFaceVertexIndicesAttr(idx)
        pot.CreateSubdivisionSchemeAttr('none')
        bind(pot, white)
        b = np.linspace(0, 2 * np.pi, 16, endpoint=False)     # invisible 16-sided hull: 40-gon faces aren't GPU-compatible
        col = UsdGeom.Mesh.Define(stage, f'{root}/pot_collider')
        col.CreatePointsAttr([Gf.Vec3f(*map(float, (org[0] + q * np.cos(t), org[1] + q * np.sin(t), z))) for z, q in ((0.0, r * 0.74), (top, r)) for t in b])
        col.CreateFaceVertexCountsAttr([16])
        col.CreateFaceVertexIndicesAttr(list(range(16)))
        col.MakeInvisible()
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr('convexHull')
        soil = UsdGeom.Cylinder.Define(stage, f'{root}/soil')
        soil.CreateAxisAttr('Z')
        soil.CreateRadiusAttr(r - 0.015)
        soil.CreateHeightAttr(0.01)
        soil.AddTranslateOp().Set(Gf.Vec3d(org[0], org[1], top - 0.03))
        bind(soil, dark)
        for k, lf in enumerate(p['leaves']):
            pts, cts, idx, base = leaf_mesh(lf['c'], lf['axis'], lf['normal'], lf['len'], lf['width'])
            m = UsdGeom.Mesh.Define(stage, f'{root}/leaf_{k}')
            m.CreatePointsAttr([Gf.Vec3f(*map(float, q)) for q in pts])
            m.CreateFaceVertexCountsAttr(cts)
            m.CreateFaceVertexIndicesAttr(idx)
            m.CreateSubdivisionSchemeAttr('none')
            m.CreateDoubleSidedAttr(True)
            bind(m, green)
            # stem: curve from the soil, near the pot's centre, up to the leaf's base
            ang = k * 2.4
            s0 = org + np.array([np.cos(ang) * r * 0.4, np.sin(ang) * r * 0.4, top - 0.03])
            ctrl = np.array([s0[0], s0[1], base[2] * 0.75 + s0[2] * 0.25])
            prev = s0
            for i, t in enumerate(np.linspace(0, 1, 7)[1:]):
                pnt = (1 - t) ** 2 * s0 + 2 * (1 - t) * t * ctrl + t * t * base
                rod(f'{root}/stem_{k}_{i}', prev, pnt, 0.009, stem)
                prev = pnt
    pink = material('Flower', (0.85, 0.45, 0.55), 0.6)
    rng = np.random.default_rng(3)
    for n, (x, y, z0, z1) in enumerate(FLOWERS, 1):
        root = f'{hall}/Scan_FlowerPot_{n}'
        UsdGeom.Xform.Define(stage, root)
        pot = UsdGeom.Cylinder.Define(stage, f'{root}/pot')
        pot.CreateAxisAttr('Z')
        pot.CreateRadiusAttr(0.07)
        pot.CreateHeightAttr(0.09)
        pot.AddTranslateOp().Set(Gf.Vec3d(x, y, z0 + 0.045))
        bind(pot, white)
        for i in range(16):
            a, r = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.11)
            z = rng.uniform(z0 + 0.1, z1 - 0.02)
            s_ = UsdGeom.Sphere.Define(stage, f'{root}/flower_{i}')
            s_.CreateRadiusAttr(0.024)
            s_.AddTranslateOp().Set(Gf.Vec3d(x + r * np.cos(a), y + r * np.sin(a), float(z)))
            bind(s_, pink)


if __name__ == '__main__':
    measure(sys.argv[1], sys.argv[2])
