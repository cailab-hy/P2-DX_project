"""RB-Y1 beside the T-shirt desk (user's product page, 2026-11-02), simple solids on the scan's pose.

Read from the height slices: torso rear surface x -2.92, arms at y 18.68 / 19.18 down to z 0.62, head top z 1.45, leg from (-2.92, z 0.6) down to
(-2.68, z 0.1). Hands and the head's bulk lie on the +x side, so the front is +x. The black base hardly shows on the scan: its size comes from the
product photo's proportions. World coordinates; place_final.py calls add().
"""
Y = 18.93
ANKLE = (-2.68, 18.90, 0.22)
KNEE = (-2.88, 18.90, 0.50)
HIP = (-2.87, 18.92, 0.74)
TORSO = (-2.92, -2.75, 0.84, 1.24)
TOP = 1.45


def add(stage, hall):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    root = f'{hall}/Scan_RBY1_1'
    UsdGeom.Xform.Define(stage, root)
    looks = UsdGeom.Scope.Define(stage, f'{root}/Looks')

    def material(name, rgb, rough=0.4, metal=0.0):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        sh.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(metal)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    white, dark, silver = material('White', (0.88, 0.88, 0.88), 0.35), material('Dark', (0.03, 0.03, 0.035), 0.4), material('Silver', (0.72, 0.73, 0.75), 0.3, 0.8)

    def bind(p, m, collide=False):
        UsdShade.MaterialBindingAPI.Apply(p.GetPrim()).Bind(m)
        if collide:
            UsdPhysics.CollisionAPI.Apply(p.GetPrim())

    def box(name, lo, hi, m, collide=False):
        c = UsdGeom.Cube.Define(stage, f'{root}/{name}')
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
        c.AddScaleOp().Set(Gf.Vec3f(*[float(b - a) for a, b in zip(lo, hi)]))
        bind(c, m, collide)

    def rod(name, p0, p1, r, m, collide=False):
        p0, p1 = Gf.Vec3d(*map(float, p0)), Gf.Vec3d(*map(float, p1))
        d = p1 - p0
        c = UsdGeom.Cylinder.Define(stage, f'{root}/{name}')
        c.CreateAxisAttr('Z')
        c.CreateRadiusAttr(r)
        c.CreateHeightAttr(d.GetLength())
        c.AddTranslateOp().Set((p0 + p1) / 2)
        c.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 0, 1), d).GetQuat()))
        bind(c, m, collide)

    # base: 0.60 x 0.50 x 0.20 m, ankle on the rear half
    bx = ANKLE[0] + 0.12
    box('base', (bx - 0.30, ANKLE[1] - 0.25, 0.05), (bx + 0.30, ANKLE[1] + 0.25, 0.20), dark, True)
    for k, y in enumerate((-0.27, 0.27)):
        rod(f'wheel_{k}', (bx, ANKLE[1] + y - 0.03, 0.09), (bx, ANKLE[1] + y + 0.03, 0.09), 0.09, dark)
    rod('ankle', (ANKLE[0], ANKLE[1] - 0.09, ANKLE[2]), (ANKLE[0], ANKLE[1] + 0.09, ANKLE[2]), 0.075, dark, True)
    rod('leg_lower', ANKLE, KNEE, 0.055, silver, True)
    rod('knee', (KNEE[0], KNEE[1] - 0.07, KNEE[2]), (KNEE[0], KNEE[1] + 0.07, KNEE[2]), 0.06, dark)
    rod('leg_upper', KNEE, HIP, 0.055, silver, True)
    rod('hip', (HIP[0], HIP[1] - 0.08, HIP[2]), (HIP[0], HIP[1] + 0.08, HIP[2]), 0.06, dark)
    rod('waist', HIP, (HIP[0], HIP[1], TORSO[2]), 0.06, dark)
    x0, x1, z0, z1 = TORSO
    box('torso_lower', (x0, Y - 0.14, z0), (x1, Y + 0.14, z0 + 0.14), dark, True)
    box('torso', (x0, Y - 0.19, z0 + 0.14), (x1, Y + 0.19, z1), white, True)
    xm = (x0 + x1) / 2
    for k, s in enumerate((-1, 1)):
        ys = Y + s * 0.25
        rod(f'shoulder_{k}', (xm, Y + s * 0.19, z1 - 0.05), (xm, ys + s * 0.03, z1 - 0.05), 0.06, dark)
        rod(f'arm_upper_{k}', (xm, ys, z1 - 0.07), (xm - 0.02, ys, 0.92), 0.042, silver)
        rod(f'elbow_{k}', (xm - 0.05, ys, 0.92), (xm + 0.02, ys, 0.92), 0.05, dark)
        rod(f'arm_fore_{k}', (xm - 0.02, ys, 0.92), (xm, ys, 0.72), 0.036, silver)
        rod(f'wrist_{k}', (xm, ys, 0.72), (xm, ys, 0.66), 0.04, dark)
        for j, d in enumerate((-0.025, 0.025)):
            box(f'finger_{k}_{j}', (xm - 0.012, ys + d - 0.008, 0.61), (xm + 0.012, ys + d + 0.008, 0.66), dark)
    rod('neck', (x0 + 0.05, Y, z1), (x0 + 0.05, Y, z1 + 0.08), 0.03, silver)
    box('head', (x0, Y - 0.06, z1 + 0.08), (x0 + 0.13, Y + 0.06, TOP), dark)
