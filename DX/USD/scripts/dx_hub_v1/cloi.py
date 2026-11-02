"""Four LG CLOi ServeBots by the curved reception desk: a lofted mesh fitted to the merged scan points.

A rounded square is fitted to the body's outline points (z 0.3-1.15, truncated distance, so the desk's points nearby don't count): one size
for all four, each with its own centre and heading. The points don't show the front, so it is taken as facing away from the desk.
    python cloi.py WORK_B   ->  WORK_B/cloi_fit.json
place_final.py calls add() with that file; the parts' heights follow the scan's total height and the product photo's proportions.
"""
import json
import sys

import numpy as np

START = [(-1.10, -1.54), (-0.80, -0.72), (-0.35, -0.13), (0.31, 0.27)]
RADIUS, TRUNC, WIN = 0.13, 0.05, 0.50
DESK = (1.35, -1.6)     # centre of the curved desk's arc


def outline_distance(u, v, w, d):
    """Distance of local points to a rounded rectangle's outline (w across u, d across v)."""
    qx, qy = np.abs(u) - (w / 2 - RADIUS), np.abs(v) - (d / 2 - RADIUS)
    out = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - RADIUS
    return np.abs(out)


def fit_outline(q, c0, w, d):
    from scipy.optimize import minimize

    def cost(p):
        ca, sa = np.cos(p[2]), np.sin(p[2])
        dx, dy = q[:, 0] - p[0], q[:, 1] - p[1]
        return np.minimum(outline_distance(ca * dx + sa * dy, -sa * dx + ca * dy, w, d), TRUNC).mean()

    best = None
    for a0 in np.radians(np.arange(0, 90, 15)):
        r = minimize(cost, (*c0, a0), method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': 1000})
        if best is None or r.fun < best.fun:
            best = r
    return best


def measure(work):
    """All four are the same robot: one size (the sum over the robots decides it), each with its own centre and heading."""
    P = np.load(f'{work}/merged_points.npz')['points']
    bodies, near = [], []
    for c0 in START:
        q = P[np.hypot(P[:, 0] - c0[0], P[:, 1] - c0[1]) < WIN]
        near.append(q)
        b = q[(q[:, 2] > 0.3) & (q[:, 2] < 1.15)]
        bodies.append(b[::2])
    table = {}
    for size in (0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60):
        rs = [fit_outline(b, c0, size, size) for b, c0 in zip(bodies, START)]
        table[size] = (sum(r.fun for r in rs) / 4, rs)
        print(f'size {size:.2f}: mean truncated distance {table[size][0] * 100:.2f} cm', flush=True)
    size = min(table, key=lambda s: table[s][0])
    fit = []
    for n, (q, r) in enumerate(zip(near, table[size][1]), 1):
        x, y, a = r.x
        outward = np.arctan2(y - DESK[1], x - DESK[0])  # front unknown from the points: away from the desk
        k = np.round((outward - a) / (np.pi / 2))
        front = a + k * np.pi / 2
        c, s = np.cos(front), np.sin(front)
        dx, dy = q[:, 0] - x, q[:, 1] - y
        u, v = c * dx + s * dy, -s * dx + c * dy
        inside = (np.abs(u) < size / 2 + 0.03) & (np.abs(v) < size / 2 + 0.03) & (q[:, 2] < 1.6)
        on = inside & (outline_distance(u, v, size, size) < 0.03) & (q[:, 2] > 0.3) & (q[:, 2] < 1.15)
        print(f'R{n} centre ({x:.3f}, {y:.3f}) front {np.degrees(front):.1f}, {r.fun * 100:.2f} cm, on outline {on.sum()}/{len(bodies[n - 1]) * 2}, top z99 {np.percentile(q[inside, 2], 99):.3f}')
        fit.append({'x': round(float(x), 3), 'y': round(float(y), 3), 'yaw': round(float(np.degrees(front)), 1), 'size': size, 'top': round(float(np.percentile(q[inside, 2], 99)), 3)})
    json.dump(fit, open(f'{work}/cloi_fit.json', 'w'), indent=1)


def add(stage, hall, fit):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_CLOi_Looks')

    def material(name, rgb, rough=0.4, emit=None):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        if emit:
            sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emit))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    mats = {'white': material('White', (0.9, 0.9, 0.9), 0.3), 'dark': material('Dark', (0.03, 0.03, 0.035), 0.35), 'green': material('Green', (0.05, 0.45, 0.15), 0.4, (0.02, 0.5, 0.12)),
            'red': material('Red', (0.75, 0.08, 0.05)), 'eye': material('Eye', (0.9, 0.95, 0.95), 0.3, (0.6, 0.7, 0.7)), 'grey': material('Grey', (0.35, 0.36, 0.38))}

    def ring(w, d, r, z):
        a = np.linspace(0, np.pi / 2, 7)
        pts = []
        for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            for t in a:
                c_, s_ = np.cos(t + k * np.pi / 2), np.sin(t + k * np.pi / 2)
                pts.append((sx * (w / 2 - r) + r * c_, sy * (d / 2 - r) + r * s_, z))
        return pts

    def loft(path, sections, w, d, mat, collide=True):
        """Mesh through rounded-rectangle rings; sections = (z, inset)."""
        rings = [ring(w - 2 * i, d - 2 * i, max(RADIUS - i, 0.02), z) for z, i in sections]
        n = len(rings[0])
        pts = [Gf.Vec3f(*map(float, p)) for rg in rings for p in rg]
        last = n * (len(rings) - 1)
        counts, idx = [n, n], list(range(n - 1, -1, -1)) + list(range(last, last + n))
        for s in range(len(rings) - 1):
            for i in range(n):
                j = (i + 1) % n
                counts.append(4)
                idx += [s * n + i, s * n + j, (s + 1) * n + j, (s + 1) * n + i]
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mats[mat])
        if collide:
            UsdPhysics.CollisionAPI.Apply(m.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')

    def box(path, lo, hi, mat):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
        c.AddScaleOp().Set(Gf.Vec3f(*[float(b - a) for a, b in zip(lo, hi)]))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mats[mat])

    def disc(path, x, y, z, r, mat, depth=0.004, axis='X'):
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateAxisAttr(axis)
        c.CreateRadiusAttr(r)
        c.CreateHeightAttr(depth)
        c.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mats[mat])

    for n, f in enumerate(fit, 1):
        root = f'{hall}/Scan_CLOi_{n}'
        xf = UsdGeom.Xform.Define(stage, root)
        xf.AddTranslateOp().Set(Gf.Vec3d(f['x'], f['y'], 0.0))
        xf.AddRotateZOp().Set(float(f['yaw']))      # local +x = the robot's front
        s, top = f['size'], f['top']
        front = s / 2
        loft(f'{root}/base', [(0.0, 0.0), (0.12, 0.0)], s, s, 'dark')
        loft(f'{root}/lower', [(0.12, 0.0), (0.156, 0.0)], s, s, 'white', False)
        loft(f'{root}/led', [(0.156, 0.004), (0.204, 0.004)], s, s, 'green', False)
        loft(f'{root}/body', [(0.204, 0.0), (top - 0.09, 0.0), (top - 0.05, 0.008), (top - 0.02, 0.03), (top - 0.006, 0.06)], s, s, 'white')
        box(f'{root}/head', (front - 0.11, -0.17, top - 0.18), (front + 0.012, 0.17, top), 'dark')
        for k, y in enumerate((-0.07, 0.07)):
            disc(f'{root}/eye_{k}', front + 0.014, y, top - 0.09, 0.024, 'eye')
            disc(f'{root}/eye_{k}_in', front + 0.017, y, top - 0.09, 0.017, 'dark')
        box(f'{root}/pocket', (front - 0.004, -0.09, 0.29), (front + 0.004, 0.09, 0.43), 'grey')
        box(f'{root}/pocket_in', (front, -0.07, 0.31), (front + 0.006, 0.07, 0.40), 'dark')
        disc(f'{root}/stop', front - 0.05, 0.21, top - 0.005, 0.015, 'red', 0.012, 'Z')


if __name__ == '__main__':
    measure(sys.argv[1])
