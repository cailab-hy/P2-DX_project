"""White chests of drawers (user's product photo, 2026-11-02), solids on the scan's boxes: one in front of the office desks, one by the care room wall.

Measured on the height slices: sharp rectangle x -8.03..-7.23, y 17.67..18.05, flat top at z 0.77. The densest face (y 17.67, towards the open
floor) is taken as the drawers' front. World coordinates; place_final.py calls add().
"""
X0, X1, Y0, Y1, TOP = -8.03, -7.23, 17.67, 18.05, 0.77
N, GAP, FRAME = 5, 0.012, 0.018
# second chest, measured the same way: x -5.07..-4.44, back y 19.22, top z 1.0. Its front steps back: y 18.80 up to z 0.62, then a slope to y 18.97 at z 0.75
B_X0, B_X1, B_BACK, B_TOP = -5.07, -4.44, 19.22, 1.0
B_PROFILE = [(18.80, 0.0), (B_BACK, 0.0), (B_BACK, B_TOP), (18.97, B_TOP), (18.97, 0.75), (18.80, 0.62)]


def add(stage, hall):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    root = f'{hall}/Scan_DrawerChest_1'
    UsdGeom.Xform.Define(stage, root)
    looks = UsdGeom.Scope.Define(stage, f'{root}/Looks')

    def material(name, rgb, rough=0.4):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    frame, face, inner = material('Frame', (0.93, 0.92, 0.89)), material('Face', (0.88, 0.86, 0.8), 0.35), material('Inner', (0.2, 0.2, 0.19), 0.8)

    def box(name, lo, hi, m, collide=False):
        c = UsdGeom.Cube.Define(stage, f'{root}/{name}')
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
        c.AddScaleOp().Set(Gf.Vec3f(*[float(b - a) for a, b in zip(lo, hi)]))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(m)
        if collide:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())

    d = 0.015       # the frame's front stands proud of the shadow panel behind the drawers
    box('body', (X0 + 0.002, Y0 + d, 0.002), (X1 - 0.002, Y1 - 0.002, TOP - 0.002), inner, True)
    box('back', (X0 + FRAME, Y1 - 0.012, FRAME), (X1 - FRAME, Y1, TOP - FRAME), frame)
    box('side_l', (X0, Y0, 0.0), (X0 + FRAME, Y1, TOP - FRAME), frame)
    box('side_r', (X1 - FRAME, Y0, 0.0), (X1, Y1, TOP - FRAME), frame)
    box('top', (X0, Y0, TOP - FRAME), (X1, Y1, TOP), frame)
    box('bottom', (X0 + FRAME, Y0, 0.0), (X1 - FRAME, Y1, FRAME), frame)
    z0, z1 = FRAME + GAP, TOP - FRAME - GAP
    h = (z1 - z0 - (N - 1) * GAP) / N
    for i in range(N):
        z = z0 + i * (h + GAP)
        box(f'drawer_{i}', (X0 + FRAME + GAP, Y0 + 0.002, z), (X1 - FRAME - GAP, Y0 + d + 0.002, z + h), face)

    root2 = f'{hall}/Scan_DrawerChest_2'
    UsdGeom.Xform.Define(stage, root2)
    pts = [Gf.Vec3f(B_X0, y, z) for y, z in B_PROFILE] + [Gf.Vec3f(B_X1, y, z) for y, z in B_PROFILE]
    n = len(B_PROFILE)
    counts, idx = [n, n], list(range(n)) + list(range(2 * n - 1, n - 1, -1))
    for i in range(n):
        j = (i + 1) % n
        counts.append(4)
        idx += [j, i, n + i, n + j]
    m = UsdGeom.Mesh.Define(stage, f'{root2}/body')
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexCountsAttr(counts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateSubdivisionSchemeAttr('none')
    m.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(frame)
    UsdPhysics.CollisionAPI.Apply(m.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(m.GetPrim()).CreateApproximationAttr('convexHull')

    def face_panels(name, y, z0, z1, count):
        h_ = (z1 - z0 - (count - 1) * GAP) / count
        for i in range(count):
            z = z0 + i * (h_ + GAP)
            c = UsdGeom.Cube.Define(stage, f'{root2}/{name}_{i}')
            c.CreateSizeAttr(1.0)
            c.AddTranslateOp().Set(Gf.Vec3d((B_X0 + B_X1) / 2, y - 0.004, z + h_ / 2))
            c.AddScaleOp().Set(Gf.Vec3f(B_X1 - B_X0 - 2 * FRAME, 0.012, float(h_)))
            UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(face)

    face_panels('drawer_low', 18.80, FRAME + GAP, 0.60, 3)
    face_panels('drawer_up', 18.97, 0.76, B_TOP - FRAME, 2)
