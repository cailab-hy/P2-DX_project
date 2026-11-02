"""Four BRS120M mobile manipulators by the west window (blobs 5, 7), simple solids fitted to the merged scan points.

The body (rounded box, 0.60 x 0.70 x 0.665 m) is the same for all four; centre and heading come from the points (east edge line, side
edges), the arm poses were read per robot from the height slices (top / x-z / y-z views), so each arm stands as it did in the scan.
    python brs120m.py WORK_B   ->  WORK_B/brs120m_fit.json
place_final.py calls add() with that file.
"""
import json
import sys

import numpy as np

W, L, H = 0.60, 0.70, 0.665     # body along local y (across), local x (arm end = -x), top
R_CORNER = 0.07
# (y window, arm points in world coordinates: base, shoulder, shoulder side, elbow side, elbow, wrist)
ROBOTS = [
    ((10.05, 10.90), [(-11.87, 10.58), (-11.87, 10.58, 1.20), (-11.87, 10.42, 1.22), (-11.47, 10.45, 1.20), (-11.40, 10.60, 1.20), (-11.37, 10.60, 0.92)]),
    ((10.90, 11.68), [(-11.87, 11.36), (-11.87, 11.36, 1.20), (-11.86, 11.20, 1.22), (-11.47, 11.25, 1.17), (-11.40, 11.38, 1.15), (-11.36, 11.38, 0.92)]),
    ((11.68, 12.50), [(-11.76, 12.18), (-11.78, 12.20, 1.20), (-11.72, 12.08, 1.22), (-11.45, 11.86, 1.16), (-11.38, 11.96, 1.16), (-11.33, 11.95, 0.93)]),
    ((12.50, 13.40), [(-11.72, 13.03), (-11.68, 13.03, 1.20), (-11.66, 12.90, 1.18), (-11.32, 12.88, 1.10), (-11.28, 12.98, 1.12), (-11.05, 12.95, 1.00)]),
]


def measure(work):
    P = np.load(f'{work}/merged_points.npz')['points']
    fit = []
    for (y0, y1), arm in ROBOTS:
        q = P[(P[:, 1] > y0) & (P[:, 1] < y1) & (P[:, 0] > -12.05) & (P[:, 0] < -10.95) & (P[:, 2] > 0.20) & (P[:, 2] < 0.60)]
        east = np.percentile(q[:, 0], 97)
        for _ in range(3):      # line x = a y + b through the body's east face
            e = q[np.abs(q[:, 0] - east) < 0.04]
            a, b = np.polyfit(e[:, 1], e[:, 0], 1)
            east = a * q[:, 1] + b
        yaw = -np.degrees(np.arctan(a))
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        v = -s * q[:, 0] + c * q[:, 1]      # across the body
        lo, hi = np.percentile(v, [2, 98])
        vc = (lo + hi) / 2
        ymid = (y0 + y1) / 2
        ex = a * ymid + b
        ue = c * ex + s * ymid              # east face along the body
        ux = ue - L / 2
        cx, cy = c * ux - s * vc, s * ux + c * vc
        fit.append({'x': round(float(cx), 3), 'y': round(float(cy), 3), 'yaw': round(float(yaw), 1), 'width_measured': round(float(hi - lo), 2), 'arm': arm})
        print(fit[-1])
    json.dump(fit, open(f'{work}/brs120m_fit.json', 'w'), indent=1)


def add(stage, hall, fit):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_BRS120M_Looks')

    def material(name, rgb, rough=0.5):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    mats = {'white': material('White', (0.86, 0.87, 0.88)), 'dark': material('Dark', (0.05, 0.05, 0.06)), 'grey': material('Grey', (0.55, 0.57, 0.6)),
            'blue': material('Blue', (0.22, 0.27, 0.5)), 'link': material('Link', (0.8, 0.81, 0.82)), 'red': material('Red', (0.75, 0.08, 0.05))}

    def finish(prim, mat, collide=True):
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mats[mat])
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())

    def prism(path, z0, z1, w, ln, mat):
        """Rounded box: corners along z, centred on the robot's origin."""
        a = np.linspace(0, np.pi / 2, 7)
        arc = np.stack([np.cos(a), np.sin(a)], 1) * R_CORNER
        ring = []
        for sx, sy, k in ((1, 1, 0), (-1, 1, 1), (-1, -1, 2), (1, -1, 3)):
            q = arc[::-1] if k % 2 else arc
            ring += [((ln / 2 - R_CORNER + abs(u)) * sx, (w / 2 - R_CORNER + abs(v)) * sy) for u, v in q]
        n = len(ring)
        pts = [Gf.Vec3f(x, y, z0) for x, y in ring] + [Gf.Vec3f(x, y, z1) for x, y in ring]
        counts, idx = [n, n], list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            idx += [i, j, n + j, n + i]
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(pts)
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr('none')
        finish(m, mat)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')

    def rod(path, p0, p1, r, mat, collide=True):
        p0, p1 = Gf.Vec3d(*map(float, p0)), Gf.Vec3d(*map(float, p1))
        d = p1 - p0
        c = UsdGeom.Cylinder.Define(stage, path)
        c.CreateAxisAttr('Z')
        c.CreateRadiusAttr(r)
        c.CreateHeightAttr(d.GetLength())
        c.AddTranslateOp().Set((p0 + p1) / 2)
        c.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), d).GetQuat()))
        finish(c, mat, collide)

    for n, f in enumerate(fit, 1):
        root = f'{hall}/Scan_BRS120M_{n}'
        xf = UsdGeom.Xform.Define(stage, root)
        xf.AddTranslateOp().Set(Gf.Vec3d(f['x'], f['y'], 0.0))
        xf.AddRotateZOp().Set(float(f['yaw']))
        c, s = np.cos(np.radians(f['yaw'])), np.sin(np.radians(f['yaw']))
        loc = lambda p: (c * (p[0] - f['x']) + s * (p[1] - f['y']), -s * (p[0] - f['x']) + c * (p[1] - f['y']), *p[2:])
        prism(f'{root}/skirt', 0.0, 0.14, W - 0.01, L - 0.01, 'dark')
        prism(f'{root}/body', 0.14, H - 0.035, W, L, 'white')
        prism(f'{root}/lid', H - 0.035, H, W, L, 'grey')
        for k, y in enumerate((-W / 2 + 0.07, W / 2 - 0.07)):       # red stop buttons, as in the product photo
            rod(f'{root}/stop_{k}', (L / 2 - 0.005, y, 0.55), (L / 2 + 0.012, y, 0.55), 0.025, 'red', False)
        base, shoulder, side, eside, elbow, wrist = f['arm']
        b = (*loc(base), H)
        s0, s1, e0, e1, w_ = loc(shoulder), loc(side), loc(eside), loc(elbow), loc(wrist)
        rod(f'{root}/arm_base', b, (b[0], b[1], H + 0.13), 0.075, 'blue')
        rod(f'{root}/arm_pillar', (b[0], b[1], H + 0.13), s0, 0.05, 'link')
        rod(f'{root}/arm_shoulder', s0, s1, 0.065, 'blue')
        rod(f'{root}/arm_upper', s1, e0, 0.05, 'link')
        rod(f'{root}/arm_elbow', e0, e1, 0.058, 'blue')
        rod(f'{root}/arm_fore', e1, w_, 0.045, 'link')
        d = np.subtract(w_, e1)
        cam = np.add(w_, d / np.linalg.norm(d) * 0.07)
        rod(f'{root}/arm_wrist', w_, cam, 0.052, 'blue')
        rod(f'{root}/arm_camera', cam, np.add(cam, d / np.linalg.norm(d) * 0.05), 0.04, 'dark')


if __name__ == '__main__':
    measure(sys.argv[1])
