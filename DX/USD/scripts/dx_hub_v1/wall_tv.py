"""Wall display on the diagonal wall between the round table and the plant (user's product photo, 2026-11-02), solids on the scan's rectangle.

The points on the wall show a sharp 16:9 rectangle: 2.53 m wide along the wall, z 0.78-2.22 (frame edges stand 0.02-0.04 m off the wall). The product
photo's 65 inch set is 1.45 m wide; the scan's size wins. The screen is off (black). World coordinates; place_final.py calls add().
"""
X, Y, YAW = -5.103, -2.041, 39.2    # centre on the wall face; local +x = out of the wall
W, Z0, Z1, DEPTH = 2.53, 0.78, 2.22, 0.046


def add(stage, hall):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    root = f'{hall}/Scan_WallTV_1'
    xf = UsdGeom.Xform.Define(stage, root)
    xf.AddTranslateOp().Set(Gf.Vec3d(X, Y, 0.0))
    xf.AddRotateZOp().Set(YAW)
    looks = UsdGeom.Scope.Define(stage, f'{root}/Looks')

    def material(name, rgb, rough):
        m = UsdShade.Material.Define(stage, f'{looks.GetPath()}/{name}')
        sh = UsdShade.Shader.Define(stage, f'{m.GetPath()}/Shader')
        sh.CreateIdAttr('UsdPreviewSurface')
        sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(rough)
        m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')
        return m

    def box(name, lo, hi, m, collide=False):
        c = UsdGeom.Cube.Define(stage, f'{root}/{name}')
        c.CreateSizeAttr(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*[(a + b) / 2 for a, b in zip(lo, hi)]))
        c.AddScaleOp().Set(Gf.Vec3f(*[float(b - a) for a, b in zip(lo, hi)]))
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(m)
        if collide:
            UsdPhysics.CollisionAPI.Apply(c.GetPrim())

    hw = W / 2
    box('body', (0.0, -hw, Z0), (DEPTH, hw, Z1), material('Body', (0.035, 0.035, 0.04), 0.5), True)
    box('screen', (DEPTH, -hw + 0.012, Z0 + 0.012), (DEPTH + 0.001, hw - 0.012, Z1 - 0.012), material('Screen', (0.006, 0.008, 0.012), 0.1))
