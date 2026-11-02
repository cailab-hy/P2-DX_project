"""Three maum.ai Aiden robots around the curved wall: a lofted mesh from the product spec (455 x 460 x 1180 mm) and the scan's profile.

Each robot's front edge and sides are fitted to the base points (z 0.3-0.58, truncated distance). The side profile (slanted display tower,
leaning back from the base) was read from the height slices; its slope faces away from the wall, which gives the front. Something 0.25 m
deep stands behind each robot (top z 0.6, up to the wall) that the spec doesn't cover: not modelled.
    python maumai.py WORK_B   ->  WORK_B/maumai_fit.json
place_final.py calls add() with that file.
"""
import json
import sys

import numpy as np

W, D, H = 0.455, 0.46, 1.18     # product spec
TRUNC, WIN = 0.05, 0.62
# scan start values: centre, direction towards the wall (deg)
START = [(-0.87, 3.30, 28.0), (0.37, 2.22, 70.0), (2.62, 2.24, 110.0)]
# measured per height band in the robots' frames (x forward, 0 = footprint centre): z, width
BASE = [(0.07, W), (0.14, W), (0.30, 0.44), (0.39, 0.41), (0.51, 0.37), (0.585, 0.365)]
# z, front x, rear x, width
TOWER = [(0.585, 0.225, 0.03, 0.36), (0.67, 0.215, 0.0, 0.36), (0.77, 0.185, -0.01, 0.36), (0.87, 0.165, -0.02, 0.36), (0.97, 0.125, -0.03, 0.355),
         (1.06, 0.095, -0.06, 0.33), (1.10, 0.06, -0.10, 0.25)]
HEAD = [(1.10, 0.06, -0.10, 0.22), (1.14, 0.035, -0.10, 0.18), (H, 0.01, -0.085, 0.12)]


def edge_distance(u, v, w):
    """Distance to the U of a base seen from the front: front edge at u = 0, sides at v = +-w/2, going back along +u."""
    du = np.maximum(-u, 0)
    side = np.hypot(du, np.abs(np.abs(v) - w / 2))
    front = np.hypot(np.abs(u), np.maximum(np.abs(v) - w / 2, 0))
    return np.minimum(side, front)


def measure(work):
    from scipy.optimize import minimize

    P = np.load(f'{work}/merged_points.npz')['points']
    bases = []
    for x, y, a in START:
        q = P[np.hypot(P[:, 0] - x, P[:, 1] - y) < WIN]
        bases.append(q[(q[:, 2] > 0.3) & (q[:, 2] < 0.58)])

    def solve(q, s, w):
        th = np.radians(s[2])
        c0 = (s[0] - 0.2 * np.cos(th), s[1] - 0.2 * np.sin(th))

        def cost(p):
            ca, sa = np.cos(p[2]), np.sin(p[2])
            dx, dy = q[:, 0] - p[0], q[:, 1] - p[1]
            return np.minimum(edge_distance(ca * dx + sa * dy, -sa * dx + ca * dy, w), TRUNC).mean()

        return min((minimize(cost, (*c0, th + d), method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': 1000})
                    for d in np.radians((-12, 0, 12))), key=lambda r: r.fun)

    table = {}
    for w in (0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46):
        rs = [solve(q, s, w) for q, s in zip(bases, START)]
        table[w] = (sum(r.fun for r in rs) / len(rs), rs)
        print(f'base width at z 0.3-0.58 = {w:.2f}: mean truncated distance {table[w][0] * 100:.2f} cm', flush=True)
    w = min(table, key=lambda k: table[k][0])
    fit = []
    for n, r in enumerate(table[w][1], 1):
        ex, ey, a = r.x
        x, y = ex + D / 2 * np.cos(a), ey + D / 2 * np.sin(a)
        yaw = (np.degrees(a) + 180) % 360
        print(f'R{n} centre ({x:.3f}, {y:.3f}) facing {yaw:.1f}, {r.fun * 100:.2f} cm')
        fit.append({'x': round(float(x), 3), 'y': round(float(y), 3), 'yaw': round(float(yaw), 1), 'base_width': w})
    json.dump(fit, open(f'{work}/maumai_fit.json', 'w'), indent=1)


def add(stage, hall, fit):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_Aiden_Looks')

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

    mats = {'white': material('White', (0.88, 0.88, 0.88), 0.35), 'frame': material('Frame', (0.74, 0.74, 0.73), 0.4), 'dark': material('Dark', (0.03, 0.03, 0.035), 0.4),
            'screen': material('Screen', (0.62, 0.58, 0.85), 0.25, (0.1, 0.09, 0.16)), 'grey': material('Grey', (0.3, 0.3, 0.32))}

    def ring(xc, w, d, r, z):
        a = np.linspace(0, np.pi / 2, 7)
        pts = []
        for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            for t in a:
                c_, s_ = np.cos(t + k * np.pi / 2), np.sin(t + k * np.pi / 2)
                pts.append((xc + sx * (d / 2 - r) + r * c_, sy * (w / 2 - r) + r * s_, z))
        return pts

    def loft(path, rings, mat, collide=True):
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

    def front_x(z, off=0.0):
        """x of the tower's front face at height z."""
        return float(np.interp(z, [t[0] for t in TOWER], [t[1] for t in TOWER])) + off

    def on_face(path, z0, z1, hw0, hw1, off, mat):
        """Strip following the tower's front face, one row per measured section."""
        zs = [z0] + [s[0] for s in TOWER if z0 < s[0] < z1] + [z1]
        hw = np.interp(zs, [z0, z1], [hw0, hw1])
        pts, counts, idx = [], [], []
        for z, h in zip(zs, hw):
            pts += [Gf.Vec3f(front_x(z, off), -float(h), z), Gf.Vec3f(front_x(z, off), float(h), z)]
        for i in range(len(zs) - 1):
            counts.append(4)
            idx += [2 * i, 2 * i + 1, 2 * i + 3, 2 * i + 2]
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr('none')
        m.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mats[mat])

    def speaker(path, y, z):
        p0 = Gf.Vec3d(front_x(z, 0.002), y, z)
        slope = (front_x(z + 0.02) - front_x(z - 0.02)) / 0.04
        nrm = Gf.Vec3d(1.0, 0.0, -slope).GetNormalized()
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateAxisAttr('Z')
        c.CreateRadiusAttr(0.032)
        c.CreateHeightAttr(0.006)
        c.AddTranslateOp().Set(p0)
        c.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), nrm).GetQuat()))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(mats['dark'])

    for n, f in enumerate(fit, 1):
        root = f'{hall}/Scan_Aiden_{n}'
        xf = UsdGeom.Xform.Define(stage, root)
        xf.AddTranslateOp().Set(Gf.Vec3d(f['x'], f['y'], 0.0))
        xf.AddRotateZOp().Set(float(f['yaw']))      # local +x = the robot's front
        loft(f'{root}/skirt', [ring(0, W, D, 0.05, 0.0), ring(0, W, D, 0.05, 0.07)], 'dark')
        loft(f'{root}/base', [ring(0, w_, D, 0.05, z) for z, w_ in BASE], 'white')
        loft(f'{root}/tower', [ring((s[1] + s[2]) / 2, s[3], s[1] - s[2], 0.03, s[0]) for s in TOWER], 'frame')
        on_face(f'{root}/screen', 0.71, 1.05, 0.15, 0.135, 0.003, 'screen')
        for k, y in enumerate((-0.09, 0.09)):
            speaker(f'{root}/speaker_{k}', y, 0.645)
        loft(f'{root}/head', [ring((s[1] + s[2]) / 2, s[3], s[1] - s[2], 0.03, s[0]) for s in HEAD], 'dark')


if __name__ == '__main__':
    measure(sys.argv[1])
