"""ThinkVision T27QD-4v monitors on the white desks (user's product page, 2026-11-02), simple solids on panels found in the scan.

WORK_B/monitor_segments.json: straight runs fitted to the points at z 0.98-1.14 over the desks. Clean ones are 0.57-0.69 m long and 0.35 m tall (z 0.87-1.22),
a 27 inch panel. USE keeps the runs that look like panels (not the robots, dividers or chest tops). Short runs are panels seen in part: they are anchored at the
end that meets another run. The points don't show which way a panel faces: it faces the nearest chair.
    python monitors.py WORK_B WORK_A   ->  WORK_B/monitors.json
place_final.py calls add() with that file.
"""
import json
import sys

import numpy as np

USE = [1, 10, 2, 32, 21, 4, 7, 24, 8, 26, 37, 3, 5, 20, 9, 6, 19, 11, 12, 13, 34, 40, 17, 25, 36, 14, 28, 41, 15, 16, 18, 22, 39]
W, BOTTOM, TOP, DESK, MIN_GAP = 0.615, 0.87, 1.225, 0.73, 0.40


def measure(work_b, work_a):
    segs = json.load(open(f'{work_b}/monitor_segments.json'))
    log = json.load(open(f'{work_b}/placement_final.json'))
    items = json.load(open(f'{work_a}/furniture_v1.json'))
    moved = {r[1]: r for r in log}
    chairs = [(moved[i['path'].split('/')[-1]][3:5] if i['path'].split('/')[-1] in moved else i['pos'][:2]) for i in items if i['asset'] == 'OfficeChair.usda']
    chairs += [r[3:5] for r in log if r[0] == 'add' and r[2] == 'OfficeChair']
    chairs = np.array(chairs, float)
    used = [(np.array(segs[k][0]), np.array(segs[k][1]), k) for k in USE]
    out = []
    for e0, e1, k in used:
        L = np.linalg.norm(e1 - e0)
        t = (e1 - e0) / L
        c = (e0 + e1) / 2
        if L < 0.55:
            ends = [min(np.linalg.norm(e - o) for a, b, j in used if j != k for o in (a, b)) for e in (e0, e1)]
            c = e0 + t * W / 2 if ends[0] <= ends[1] else e1 - t * W / 2
        n = np.array([-t[1], t[0]])
        chair = chairs[np.argmin(np.hypot(*(chairs - c).T))]
        if n @ (chair - c) < 0:
            n = -n
        yaw = float(np.degrees(np.arctan2(n[1], n[0])))
        # the same panel found twice; panels back to back stand as close but face apart
        if any(np.hypot(*(c - np.array(o['c']))) < MIN_GAP and np.cos(np.radians(yaw - o['yaw'])) > 0 for o in out):
            print('skip', k)
            continue
        out.append({'seg': k, 'c': np.round(c, 3).tolist(), 'yaw': round(yaw, 1), 'run': round(float(L), 2)})
        print(out[-1])
    json.dump(out, open(f'{work_b}/monitors.json', 'w'), indent=1)


def add(stage, hall, mons):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_Monitor_Looks')

    def material(name, rgb, rough, metal=0.0):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metal)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    body, screen, red, bar = material('Body', (0.03, 0.03, 0.035), 0.5), material('Screen', (0.01, 0.015, 0.03), 0.12), material('Red', (0.75, 0.05, 0.04), 0.4), material('Bar', (0.1, 0.1, 0.11), 0.6)

    def box(path, lo, hi, m, collide=False):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
        c.AddScaleOp().Set(Gf.Vec3f(*[float(b - a) for a, b in zip(lo, hi)]))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(m)
        if collide:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())

    hw = W / 2
    for n, f in enumerate(mons, 1):
        root = f'{hall}/Scan_Monitor_{n}'
        xf = UsdGeom.Xform.Define(stage, root)
        xf.AddTranslateOp().Set(Gf.Vec3d(f['c'][0], f['c'][1], 0.0))
        xf.AddRotateZOp().Set(f['yaw'])     # local +x = out of the screen
        box(f'{root}/back', (-0.035, -hw, BOTTOM), (0.0, hw, TOP), body, True)
        box(f'{root}/screen', (0.0, -hw + 0.006, BOTTOM + 0.05), (0.002, hw - 0.006, TOP - 0.008), screen)
        box(f'{root}/speaker_bar', (0.0, -hw, BOTTOM), (0.003, hw, BOTTOM + 0.045), bar)
        box(f'{root}/webcam', (-0.03, -0.1, TOP), (0.0, 0.1, TOP + 0.022), body)
        box(f'{root}/stand', (-0.075, -0.042, DESK + 0.012), (-0.035, 0.042, BOTTOM + 0.1), body)
        box(f'{root}/foot', (-0.17, -0.13, DESK), (0.05, 0.13, DESK + 0.012), body, True)
        box(f'{root}/foot_stripe', (0.04, 0.015, DESK + 0.012), (0.046, 0.115, DESK + 0.0135), red)


if __name__ == '__main__':
    measure(sys.argv[1], sys.argv[2])
