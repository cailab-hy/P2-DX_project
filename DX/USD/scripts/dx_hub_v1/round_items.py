"""Round table and round fabric stools found on the scans (user's product photos, 2026-11-02): circles fitted to the points -> lofted meshes.

    python round_items.py WORK_B   ->  WORK_B/round_items.json
place_final.py calls add() with that file.
"""
import json
import sys

import numpy as np

TRUNC = 0.04
# table by the diagonal wall; stools: by the wall, then the rows by the south window
TABLE_START = (-4.0, -2.43)
STOOL_START = [(-3.57, -2.18), (-11.0, 0.58), (-10.52, 0.58), (-10.03, 0.58)]


def circle_fit(q, c0, radii):
    from scipy.optimize import minimize

    best = None
    for r in radii:
        def cost(p):
            return np.minimum(np.abs(np.hypot(q[:, 0] - p[0], q[:, 1] - p[1]) - r), TRUNC).mean()
        s = minimize(cost, c0, method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-5})
        if best is None or s.fun < best[0]:
            best = (s.fun, s.x[0], s.x[1], r)
    return best


def measure(work):
    Z = np.load(f'{work}/merged_points.npz')
    P, C = Z['points'], Z['colors']

    def near(c, r):
        k = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1]) < r
        return P[k], C[k]

    q, _ = near(TABLE_START, 0.45)
    top = q[(q[:, 2] > 0.68) & (q[:, 2] < 0.78)]
    f, x, y, r = circle_fit(top, TABLE_START, np.arange(0.20, 0.36, 0.01))
    # the top is thin: its height from the band's mode
    h, e = np.histogram(top[:, 2], bins=np.arange(0.68, 0.79, 0.01))
    ztop = float(e[np.argmax(h) + 1])
    d = np.hypot(top[:, 0] - x, top[:, 1] - y)
    r = float(np.percentile(d[d < 0.40], 97))     # the fit's r sees the scattered points on the disc: the edge is the outer limit
    print(f'table ({x:.3f}, {y:.3f}) r {r:.3f} top z {ztop:.2f}, {f * 100:.2f} cm')
    out = {'table': {'x': round(float(x), 3), 'y': round(float(y), 3), 'r': round(r, 3), 'top': round(ztop, 2)}, 'stools': []}
    bodies = []
    for c0 in STOOL_START:
        q, col = near(c0, 0.40)
        k = (q[:, 2] > 0.08) & (q[:, 2] < 0.40)
        bodies.append((q[k], col[k], q))
    table = {}
    for r in (0.23, 0.24, 0.25):       # the row's 0.50 m spacing limits r
        fs = [circle_fit(b[0], c0, [r]) for b, c0 in zip(bodies, STOOL_START)]
        table[r] = (sum(s[0] for s in fs) / len(fs), fs)
        print(f'stool radius {r:.2f}: mean {table[r][0] * 100:.2f} cm')
    r = min(table, key=lambda k: table[k][0])
    for n, ((b, col, q), s) in enumerate(zip(bodies, table[r][1]), 1):
        d = np.hypot(b[:, 0] - s[1], b[:, 1] - s[2])
        on = np.abs(d - r) < 0.03
        th = np.arctan2(b[on, 1] - s[2], b[on, 0] - s[1])
        cov = len(np.unique(th // (np.pi / 8))) / 16
        ins = d < r + 0.01
        rgb = col[ins].mean(0) if ins.sum() else np.zeros(3)
        # top height: highest band with many points inside the circle
        qd = np.hypot(q[:, 0] - s[1], q[:, 1] - s[2]) < r - 0.04
        hh, ee = np.histogram(q[qd, 2], bins=np.arange(0.3, 0.7, 0.02))
        ztop = float(ee[np.argmax(hh) + 1]) if qd.sum() else 0.0
        print(f'stool {n} ({s[1]:.3f}, {s[2]:.3f}) {s[0] * 100:.2f} cm, on circle {int(on.sum())}, coverage {cov:.2f}, inside {int(ins.sum())}, top z {ztop:.2f}, rgb {np.round(rgb, 2).tolist()}')
        out['stools'].append({'x': round(float(s[1]), 3), 'y': round(float(s[2]), 3), 'r': r, 'top': round(ztop, 2), 'rgb': np.round(rgb, 3).tolist(), 'coverage': round(cov, 2)})
    # second row by the window (user: six in all): the points there are too few to fit, so the first row's x, touching it; beige as in the photo
    row = out['stools'][1:4]
    for s in row:
        out['stools'].append({'x': s['x'], 'y': round(s['y'] - 2 * s['r'], 3), 'r': s['r'], 'top': s['top'], 'rgb': row[0]['rgb'], 'coverage': 0.0})
        print('second row', out['stools'][-1]['x'], out['stools'][-1]['y'])
    json.dump(out, open(f'{work}/round_items.json', 'w'), indent=1)


def add(stage, hall, data):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_Round_Looks')

    def material(name, rgb, rough=0.5, metal=0.0):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metal)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    def revolve(path, sections, mat, collide=True):
        """Mesh of revolution about z; sections = (z, radius), first and last closed."""
        n = 48
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = [Gf.Vec3f(float(r * np.cos(t_)), float(r * np.sin(t_)), float(z)) for z, r in sections for t_ in a]
        last = n * (len(sections) - 1)
        counts, idx = [n, n], list(range(n - 1, -1, -1)) + list(range(last, last + n))
        for s in range(len(sections) - 1):
            for i in range(n):
                j = (i + 1) % n
                counts.append(4)
                idx += [s * n + i, s * n + j, (s + 1) * n + j, (s + 1) * n + i]
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr('none')
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
        if collide:     # an invisible 16-sided hull: 48-gon faces aren't GPU-compatible
            k = 16
            b = np.linspace(0, 2 * np.pi, k, endpoint=False)
            c = UsdGeom.Mesh.Define(stage, f'{path}_collider')
            c.CreatePointsAttr([Gf.Vec3f(float(r * np.cos(t_)), float(r * np.sin(t_)), float(z)) for z, r in (sections[0], sections[-1], max(sections, key=lambda e: e[1])) for t_ in b])
            c.CreateFaceVertexCountsAttr([k])
            c.CreateFaceVertexIndicesAttr(list(range(k)))
            c.MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(c.GetPrim()).CreateApproximationAttr('convexHull')

    t_ = data['table']
    root = f'{hall}/Scan_RoundTable_1'
    UsdGeom.Xform.Define(stage, root).AddTranslateOp().Set(Gf.Vec3d(t_['x'], t_['y'], 0.0))
    wood, steel = material('TableTop', (0.74, 0.62, 0.47), 0.45), material('Steel', (0.72, 0.73, 0.74), 0.3, 1.0)
    r, top = t_['r'], t_['top']
    revolve(f'{root}/base', [(0.0, r * 0.46), (0.012, r * 0.46), (0.02, r * 0.43)], steel)
    revolve(f'{root}/post', [(0.02, 0.03), (top - 0.024, 0.03)], steel)
    revolve(f'{root}/top', [(top - 0.024, r - 0.004), (top - 0.004, r), (top, r - 0.004)], wood)
    feet = material('Feet', (0.03, 0.03, 0.03))
    for n, s in enumerate(data['stools'], 1):
        root = f'{hall}/Scan_RoundStool_{n}'
        UsdGeom.Xform.Define(stage, root).AddTranslateOp().Set(Gf.Vec3d(s['x'], s['y'], 0.0))
        r, h = s['r'], s['top']
        fabric = material(f'Fabric_{n}', np.clip(np.array(s['rgb']) * 1.2, 0, 1), 0.85)
        revolve(f'{root}/feet', [(0.0, r - 0.06), (0.02, r - 0.06)], feet, False)
        revolve(f'{root}/body', [(0.02, r - 0.02), (0.04, r - 0.004), (0.07, r), (h - 0.05, r), (h - 0.03, r - 0.004), (h - 0.012, r - 0.016), (h - 0.003, r - 0.04), (h, r - 0.09)], fabric)


if __name__ == '__main__':
    measure(sys.argv[1])
