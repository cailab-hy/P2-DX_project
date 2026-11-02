"""Black dome pendant lamps under the DX Hub ceiling (user's product photo, 2026-11-02): lofted meshes on the points above z 1.5 of the 11/02 scan.

The merged cloud stops at 2.55 m, so WORK_B/high_points.npz holds the raw scan's upper part (1.5 cm grid, refined transform). Seeds come from the
z 1.7-2.3 map without the building's points; each lamp's centre and rim radius come from the outline's extent, rim and top heights from the height
profile. Lamps seen on less than COVER of their outline take the size of their row, and are flagged.
    python pendant_lamps.py WORK_B WORK_A   ->  WORK_B/pendant_lamps.json
place_final.py calls add() with that file.
"""
import json
import sys

import numpy as np

CEILING, COVER, LIGHT = 2.70, 0.6, 12000.0
# x, y, radius for a lamp seen only in part (its row's size)
SEEDS = [(-11.52, 0.64, .14), (-9.98, 0.64, .14), (-8.32, 0.63, .14), (-6.73, 0.63, .14), (-12.62, 3.12, .22), (-11.45, 4.24, .14), (-10.18, 4.23, .14), (-8.76, 4.22, .14),
         (-7.13, 4.21, .14), (-3.57, 6.57, .22), (-12.24, 6.84, .22), (-8.59, 6.92, .14), (-3.55, 8.38, .22), (-12.26, 9.59, .22), (-8.57, 9.59, .14), (-3.53, 11.39, .22),
         (-12.17, 12.26, .22), (-8.55, 12.25, .14), (-3.55, 13.20, .22), (-12.6, 15.0, .22), (-12.55, 17.0, .22), (-9.08, 17.46, .22), (-7.20, 17.45, .22), (-12.58, 19.0, .22),
         (-5.72, 20.40, .22), (-8.17, 20.42, .22), (-10.75, 20.53, .22)]


def measure(work_b, work_a):
    from scipy.spatial import cKDTree

    H = np.load(f'{work_b}/high_points.npz')['p']
    B = np.load(f'{work_a}/building_points.npy')
    b = B[(B[:, 2] > 1.5) & (B[:, 2] < 2.5)]
    q = H[(H[:, 2] > 1.7) & (H[:, 2] < 2.35)]
    d, _ = cKDTree(b).query(q, k=1)
    q = q[d > 0.06]
    out = []
    for x, y, r0 in SEEDS:
        for _ in range(4):
            p = q[np.hypot(q[:, 0] - x, q[:, 1] - y) < r0 + 0.08]
            (ax, ay), (bx, by) = np.percentile(p[:, :2], [2, 98], axis=0)
            x, y = (ax + bx) / 2, (ay + by) / 2
        dd = np.hypot(p[:, 0] - x, p[:, 1] - y)
        r = float((bx - ax + by - ay) / 4)
        th = np.arctan2(p[:, 1] - y, p[:, 0] - x)
        cover = len(np.unique(th // (np.pi / 12))) / 24
        ok = cover >= COVER and abs(r - r0) < 0.05
        h, e = np.histogram(p[:, 2], bins=np.arange(1.7, 2.3, 0.02))
        rim = float(e[np.argmax(h >= 0.15 * h.max())])
        dome = p[dd > 0.06]
        top = float(np.percentile(dome[:, 2], 99.5))
        if not ok:
            x, y = float(np.mean(p[:, 0])), float(np.mean(p[:, 1])) if cover < 0.3 else y
        out.append({'x': round(float(x), 3), 'y': round(float(y), 3), 'r': round(r if ok else r0, 3), 'rim': round(rim, 2), 'top': round(min(max(top, rim + 0.1), rim + 0.22), 2), 'cover': round(cover, 2), 'measured_r': round(r, 3), 'sized_by_row': not ok, 'n': len(p)})
        print(out[-1])
    json.dump(out, open(f'{work_b}/pendant_lamps.json', 'w'), indent=1)


def add(stage, hall, lamps):
    from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_Lamp_Looks')

    def material(name, rgb, rough, emit=None):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        if emit:
            sh.CreateInput('emissiveColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emit))
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    black, cream, cord = material('Shade', (0.035, 0.035, 0.04), 0.55), material('Inside', (0.9, 0.82, 0.55), 0.6, (1.4, 1.1, 0.62)), material('Cord', (0.03, 0.03, 0.03), 0.6)
    bulb = material('Bulb', (1.0, 0.92, 0.75), 0.3, (40.0, 31.0, 18.0))

    def revolve(path, sections, mat):
        n = 64
        a = np.linspace(0, 2 * np.pi, n, endpoint=False)
        pts = [Gf.Vec3f(float(r * np.cos(t)), float(r * np.sin(t)), float(z)) for z, r in sections for t in a]
        counts, idx = [], []
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
        m.CreateDoubleSidedAttr(True)
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

    for n, f in enumerate(lamps, 1):
        root = f'{hall}/Scan_PendantLamp_{n}'
        UsdGeom.Xform.Define(stage, root).AddTranslateOp().Set(Gf.Vec3d(f['x'], f['y'], 0.0))
        r, rim, top = f['r'], f['rim'], f['top']
        h = top - rim
        t = np.radians(np.linspace(0, 90, 17))
        revolve(f'{root}/shade', [(rim + h * np.sin(a), max(r * np.cos(a), 0.012)) for a in t], black)
        revolve(f'{root}/inside', [(rim + (h - 0.012) * np.sin(a), max((r - 0.004) * np.cos(a), 0.01)) for a in t], cream)
        revolve(f'{root}/rim', [(rim, r - 0.004), (rim, r)], cream)
        s = UsdGeom.Sphere.Define(stage, f'{root}/bulb')
        s.CreateRadiusAttr(min(0.3 * r, 0.05))
        s.AddTranslateOp().Set(Gf.Vec3d(0, 0, rim + 0.45 * h))
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(bulb)
        lamp = UsdLux.DiskLight.Define(stage, f'{root}/light')     # shines down (-z); delete these prims if a frame rate needs it
        lamp.CreateRadiusAttr(0.7 * r)
        lamp.CreateIntensityAttr(LIGHT)
        lamp.CreateColorAttr(Gf.Vec3f(1.0, 0.82, 0.58))
        lamp.AddTranslateOp().Set(Gf.Vec3d(0, 0, rim + 0.01))
        c = UsdGeom.Cylinder.Define(stage, f'{root}/cord')
        c.CreateAxisAttr('Z')
        c.CreateRadiusAttr(0.004)
        c.CreateHeightAttr(CEILING - top)
        c.AddTranslateOp().Set(Gf.Vec3d(0, 0, (CEILING + top) / 2))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(cord)


if __name__ == '__main__':
    measure(sys.argv[1], sys.argv[2])
