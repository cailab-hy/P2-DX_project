"""Two RBM-S100a mobile robots beside the end of the curved counter (user's product page, 2026-11-02), simple solids.

Overall size from the product page (510 x 510 x 1250 mm). Positions, tier heights (deck 0.25, sheets at 0.55 and 0.85 m), the bins' extent and the
rear column were read from the height slices; local +x = the robot's front (away from the column). Colours and details from the page.
place_final.py calls add().
"""
import numpy as np

W, D, H = 0.51, 0.51, 1.25
ROBOTS = [(3.94, -1.48, 2.0), (3.94, -2.40, 2.0)]   # x, y, yaw read from the slices
BIN_X = (-0.235, 0.115)
BIN_W = 0.40
TIERS = [(0.30, 0.55), (0.57, 0.85)]


def add(stage, hall):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    looks = UsdGeom.Scope.Define(stage, f'{hall}/Scan_RBM_Looks')

    def material(name, rgb, rough=0.45):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    white, dark, yellow, screen = material('White', (0.9, 0.9, 0.9), 0.3), material('Dark', (0.04, 0.04, 0.045), 0.4), material('Yellow', (0.92, 0.72, 0.08), 0.55), material('Screen', (0.02, 0.02, 0.03), 0.15)

    def bind(prim, mat, collide=False):
        UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
        if collide:
            UsdPhysics.CollisionAPI.Apply(prim.GetPrim())

    def rounded(path, z0, z1, w, d, r, mat):
        a = np.linspace(0, np.pi / 2, 7)
        ring = []
        for k, (sx, sy) in enumerate(((1, 1), (-1, 1), (-1, -1), (1, -1))):
            for t in a:
                ring.append((sx * (d / 2 - r) + r * np.cos(t + k * np.pi / 2), sy * (w / 2 - r) + r * np.sin(t + k * np.pi / 2)))
        n = len(ring)
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr([Gf.Vec3f(float(x), float(y), z0) for x, y in ring] + [Gf.Vec3f(float(x), float(y), z1) for x, y in ring])
        counts, idx = [n, n], list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            idx += [i, j, n + j, n + i]
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr('none')
        bind(m, mat, True)
        UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')

    def box(path, lo, hi, mat, collide=False):
        c = UsdGeom.Cube.Define(stage, path)
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
        c.AddScaleOp().Set(Gf.Vec3f(*[float(b - a) for a, b in zip(lo, hi)]))
        bind(c, mat, collide)

    def side_plate(path, y0, y1, z0, z1, mat):
        """Bin side: full height at the back, sloping down to 60 % at the front."""
        x0, x1 = BIN_X
        low, mid = z0 + 0.6 * (z1 - z0), x0 + 0.55 * (x1 - x0)
        prof = [(x0, z0), (x1, z0), (x1, low), (mid, z1), (x0, z1)]
        n = len(prof)
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr([Gf.Vec3f(x, y0, z) for x, z in prof] + [Gf.Vec3f(x, y1, z) for x, z in prof])
        counts, idx = [n, n], list(range(n - 1, -1, -1)) + list(range(n, 2 * n))
        for i in range(n):
            j = (i + 1) % n
            counts.append(4)
            idx += [i, j, n + j, n + i]
        m.CreateFaceVertexCountsAttr(counts)
        m.CreateFaceVertexIndicesAttr(idx)
        m.CreateSubdivisionSchemeAttr('none')
        m.CreateDoubleSidedAttr(True)
        bind(m, mat)

    for n, (x, y, yaw) in enumerate(ROBOTS, 1):
        root = f'{hall}/Scan_RBM_{n}'
        xf = UsdGeom.Xform.Define(stage, root)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
        xf.AddRotateZOp().Set(yaw)
        for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            c = UsdGeom.Cylinder.Define(stage, f'{root}/wheel_{k}')
            c.CreateAxisAttr('Y')
            c.CreateRadiusAttr(0.035)
            c.CreateHeightAttr(0.04)
            c.AddTranslateOp().Set(Gf.Vec3d(sx * 0.17, sy * 0.19, 0.035))
            bind(c, dark)
        rounded(f'{root}/chassis', 0.035, 0.17, W, D, 0.06, white)
        rounded(f'{root}/spacer', 0.17, 0.21, W - 0.1, D - 0.1, 0.05, dark)
        rounded(f'{root}/deck', 0.21, 0.30, W, D, 0.06, white)
        box(f'{root}/panel', (D / 2 - 0.004, 0.05, 0.225), (D / 2 + 0.004, 0.22, 0.285), dark)
        for t, (z0, z1) in enumerate(TIERS):
            b = f'{root}/bin_{t}'
            hw = BIN_W / 2
            box(f'{b}_floor', (BIN_X[0], -hw, z0), (BIN_X[1], hw, z0 + 0.012), yellow, True)
            box(f'{b}_back', (BIN_X[0], -hw, z0), (BIN_X[0] + 0.012, hw, z1), yellow)
            box(f'{b}_front', (BIN_X[1] - 0.012, -hw, z0), (BIN_X[1], hw, z0 + 0.6 * (z1 - z0)), yellow)
            side_plate(f'{b}_left', hw - 0.012, hw, z0, z1, yellow)
            side_plate(f'{b}_right', -hw, -hw + 0.012, z0, z1, yellow)
        box(f'{root}/tray', (BIN_X[0] - 0.005, -BIN_W / 2 - 0.01, 0.55), (BIN_X[1] + 0.01, BIN_W / 2 + 0.01, 0.57), dark)
        box(f'{root}/column', (-D / 2 + 0.005, -0.125, 0.30), (-D / 2 + 0.075, 0.125, H), white, True)
        d = UsdGeom.Cube.Define(stage, f'{root}/display')
        d.CreateSizeAttr(1.0)
        d.AddTranslateOp().Set(Gf.Vec3d(-D / 2 + 0.09, 0.0, 1.11))
        d.AddRotateYOp().Set(-10.0)
        d.AddScaleOp().Set(Gf.Vec3f(0.018, 0.17, 0.22))
        bind(d, screen)
