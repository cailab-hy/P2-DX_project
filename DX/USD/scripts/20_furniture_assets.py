"""Stage 20 (add_obj_v1): care-room (bedroom) furniture assets -> DX_Hub_Building_add_obj_v1/assets/<Name>/<Name>.usda

All sizes, colours and physics parameters come from config/add_obj_v1_furniture.json. Every asset has its local origin at
the floor centre of its footprint, z up, front facing +y, and is a single USD file (no external dependencies).

  Bed          rigid frame + rigid mattress (fixed joint), pillow = volume deformable (auto hex mesh, cooked here),
               blanket = surface deformable (triangle sheet)
  Sofa3        one rigid body (legs, base, arms, back, 3 seat + 3 back cushions)
  SofaTable    one rigid body (top, two side panels, bottom shelf, casters)
  Plant        one rigid body (tray, tapered pot, soil, trunk, leaves; leaves are visual only)
  Wardrobe800  articulation: body + 2 hinged doors (revolute, 0-110 deg) + 1 drawer (prismatic)   (장롱)
  Vanity640    articulation: body + desk drawer (prismatic); oval mirror, riser shelf                (화장대)
  Closet640    articulation: body + 2 drawers (prismatic); hanger rail and coat hooks                (옷장)
  -- kitchen --
  BarStool     rigid (C600G: blue saddle seat, white wire sled frame)
  Showcase     articulation: body + 2 rear sliding glass doors (prismatic)            (GWRKHM07CS2F)
  TrashBin3    plinth + 3 hollow bins (each a rigid body), coloured lids with a slot
  FireExtinguisher rigid (cylinder, head, hose, red stand)
  WaterPurifier  rigid (WPU-B600F stand type)
  Sink         rigid top-mount stainless sink, measured 600 x 445 x 200 (bowl 550 x 370 x 170), variantSet faucet (origin = counter top)
  Induction    rigid built-in 3-zone cooktop (IHR-B32A3; origin = counter top; needs a counter cut-out)
  Microwave    articulation: body (open cavity) + left-hinged door (revolute) + push button (prismatic, spring)
  Fridge       articulation: body (two cavities, shelves) + freezer door + fridge door (revolute, right hinges)
  -- central hall --
  Desk6 (6-person desk), Chair6, TVStand, Armchair (1-seat sofa, Sofa3 style), OfficeDesk, OfficeChair: rigid
  CoffeeRobot  Dr.Presso booth: static colliders + posed Doosan A0509 (referenced, physics disabled); CoffeeKiosk: rigid
  CareDesk     rigid desk (top in the wall tone, white legs); TShirt: surface deformable two-layer T-shirt (sewn seams)

Runs inside Isaac Sim (headless) because the deformable tet meshes are cooked by PhysX and stored in the asset.
"""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from omni.physx import get_physx_cooking_interface  # noqa: E402
from omni.physx.scripts import deformableUtils, physicsUtils  # noqa: E402
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSemantics, UsdShade, Vt  # noqa: E402

USD = Path("/home/cai/P2&DX_project/P2&DX/USD")
CFG = json.load(open(USD / "config" / "add_obj_v1_furniture.json"))
A, COL = CFG["assets"], CFG["colors"]
OUT = USD / "DX_Hub_Building_add_obj_v1" / "assets"
ctx = omni.usd.get_context()


def lin(c):
    c = np.asarray(c, float) / 255.0
    return tuple(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).tolist())


# ------------------------------------------------------------------ mesh primitives (V: Nx3, F: list of index lists)
def box_geo(c, s):
    (cx, cy, cz), (sx, sy, sz) = c, s
    x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz - sz / 2, cz + sz / 2
    quads = [[(x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)], [(x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0)],
             [(x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (x0, y0, z0)], [(x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)],
             [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], [(x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)]]
    V = np.array([p for q in quads for p in q], float)
    return V, [list(range(4 * k, 4 * k + 4)) for k in range(6)]


def superquadric(c, s, e=0.2, nu=40, nv=20):
    """Rounded box / ellipsoid: |x/a|^(2/e) + ... ; e small = boxy with rounded edges, e=1 ellipsoid."""
    a, b, h = s[0] / 2, s[1] / 2, s[2] / 2

    def sp(x, p):
        return np.sign(x) * np.abs(x) ** p
    V = [[c[0], c[1], c[2] - h]]
    for i in range(1, nv):
        v = -math.pi / 2 + math.pi * i / nv
        for j in range(nu):
            u = -math.pi + 2 * math.pi * j / nu
            V.append([c[0] + a * sp(math.cos(v), e) * sp(math.cos(u), e), c[1] + b * sp(math.cos(v), e) * sp(math.sin(u), e),
                      c[2] + h * sp(math.sin(v), e)])
    V.append([c[0], c[1], c[2] + h])
    F = [[0, 1 + (j + 1) % nu, 1 + j] for j in range(nu)]
    for i in range(nv - 2):
        for j in range(nu):
            a_, b_ = 1 + i * nu + j, 1 + i * nu + (j + 1) % nu
            F.append([a_, b_, b_ + nu, a_ + nu])
    top = len(V) - 1
    F += [[top, 1 + (nv - 2) * nu + j, 1 + (nv - 2) * nu + (j + 1) % nu] for j in range(nu)]
    return np.array(V, float), F


def cylinder(c, r0, r1, h, n=20, axis="z"):
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    bot = np.c_[r0 * np.cos(ang), r0 * np.sin(ang), np.zeros(n)]
    top = np.c_[r1 * np.cos(ang), r1 * np.sin(ang), np.full(n, h)]
    V = np.vstack([bot, top, [[0, 0, 0], [0, 0, h]]])
    F = [[j, (j + 1) % n, n + (j + 1) % n, n + j] for j in range(n)]
    F += [[2 * n, (j + 1) % n, j] for j in range(n)] + [[2 * n + 1, n + j, n + (j + 1) % n] for j in range(n)]
    if axis == "y":
        V = V[:, [0, 2, 1]] * [1, 1, 1]
        V[:, 1] -= h / 2
        F = [f[::-1] for f in F]
    elif axis == "x":
        V = V[:, [2, 0, 1]]
        V[:, 0] -= h / 2
    return V + np.asarray(c, float), F


def frustum_box(c, top, bottom, h):
    """Square pot tapering from `bottom` (z=0) to `top` (z=h)."""
    b, t = bottom / 2, top / 2
    V = np.array([[-b, -b, 0], [b, -b, 0], [b, b, 0], [-b, b, 0], [-t, -t, h], [t, -t, h], [t, t, h], [-t, t, h]], float)
    F = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return V + np.asarray(c, float), F


def stadium(c, w, h, t, n=24):
    """Oval (stadium) plate in the x-z plane, thickness t along y."""
    r = w / 2
    half = h / 2 - r
    pts = [(r * math.cos(a), half + r * math.sin(a)) for a in np.linspace(0, math.pi, n)]
    pts += [(r * math.cos(a), -half + r * math.sin(a)) for a in np.linspace(math.pi, 2 * math.pi, n)]
    m = len(pts)
    V = np.array([[x, -t / 2, z] for x, z in pts] + [[x, t / 2, z] for x, z in pts] + [[0, -t / 2, 0], [0, t / 2, 0]], float)
    F = [[j, (j + 1) % m, m + (j + 1) % m, m + j] for j in range(m)]
    F += [[2 * m, (j + 1) % m, j] for j in range(m)] + [[2 * m + 1, m + j, m + (j + 1) % m] for j in range(m)]
    return V + np.asarray(c, float), F


def rod(p0, p1, r, n=12):
    """Cylinder between two points (wire frames, rails)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    V, F = cylinder((0, 0, 0), r, r, L, n=n)
    z = d / L
    a = np.cross([0, 0, 1], z)
    if np.linalg.norm(a) < 1e-9:
        R = np.eye(3) if z[2] > 0 else np.diag([1, -1, -1])
    else:
        a /= np.linalg.norm(a)
        th = math.acos(np.clip(z[2], -1, 1))
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K
    return V @ R.T + p0, F


def rot_z(V, deg, about=(0, 0, 0)):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return (np.asarray(V) - about) @ R.T + about


# ------------------------------------------------------------------ asset writer
class Asset:
    def __init__(self, name, label):
        ctx.new_stage()
        for _ in range(3):
            app.update()
        self.st = ctx.get_stage()
        self.name, self.R = name, f"/{name}"
        UsdGeom.SetStageUpAxis(self.st, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(self.st, 1.0)
        UsdPhysics.SetStageKilogramsPerUnit(self.st, 1.0)
        root = UsdGeom.Xform.Define(self.st, self.R)
        self.st.SetDefaultPrim(root.GetPrim())
        Usd.ModelAPI(root).SetKind(Kind.Tokens.component)
        self.label(root.GetPrim(), label)
        self.mats = {}
        self.n = 0
        pm = UsdShade.Material.Define(self.st, f"{self.R}/Looks/PhysicsMaterial")
        api = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
        api.CreateStaticFrictionAttr(0.6)
        api.CreateDynamicFrictionAttr(0.5)
        api.CreateRestitutionAttr(0.0)
        self.phys_mat = pm

    @staticmethod
    def label(prim, cls):
        UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([cls])

    def material(self, key, rough=0.6, metal=0.0, opacity=1.0):
        k = (key, rough, metal, opacity)
        if k not in self.mats:
            m = UsdShade.Material.Define(self.st, f"{self.R}/Looks/{key}_{len(self.mats)}")
            sh = UsdShade.Shader.Define(self.st, f"{m.GetPath()}/Shader")
            sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*lin(COL[key])))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
            sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
            m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
            self.mats[k] = m
        return self.mats[k]

    def mesh(self, parent, geo, mat, collide=True, name=None, smooth=False, label=None):
        V, F = geo
        path = f"{parent}/{name or f'part_{self.n:03d}'}"
        self.n += 1
        me = UsdGeom.Mesh.Define(self.st, path)
        me.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in np.asarray(V, float)]))
        me.CreateFaceVertexCountsAttr([len(f) for f in F])
        me.CreateFaceVertexIndicesAttr([i for f in F for i in f])
        me.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*np.min(V, 0)), Gf.Vec3f(*np.max(V, 0))]))
        me.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        if smooth:                                                    # smooth vertex normals (rounded parts)
            nrm = np.zeros_like(np.asarray(V, float))
            Vn = np.asarray(V, float)
            for f in F:
                n_ = np.cross(Vn[f[1]] - Vn[f[0]], Vn[f[2]] - Vn[f[0]])
                nrm[f] += n_
            nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
            me.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in nrm]))
            me.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(mat)
        if collide:
            UsdPhysics.CollisionAPI.Apply(me.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(me.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
            UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(self.phys_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
        if label:
            self.label(me.GetPrim(), label)
        return me

    def box(self, parent, c, s, mat, **kw):
        return self.mesh(parent, box_geo(c, s), mat, **kw)

    def body(self, name, mass, label, translate=(0, 0, 0)):
        x = UsdGeom.Xform.Define(self.st, f"{self.R}/{name}")
        if any(translate):
            x.AddTranslateOp().Set(Gf.Vec3d(*translate))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        UsdPhysics.MassAPI.Apply(x.GetPrim()).CreateMassAttr(mass)
        self.label(x.GetPrim(), label)
        return str(x.GetPath())

    def joint(self, kind, name, b0, b1, pos0, pos1, axis=None, limits=None, damping=None, drive=None):
        scope = f"{self.R}/Joints"
        UsdGeom.Scope.Define(self.st, scope)
        J = {"fixed": UsdPhysics.FixedJoint, "revolute": UsdPhysics.RevoluteJoint, "prismatic": UsdPhysics.PrismaticJoint}[kind]
        j = J.Define(self.st, f"{scope}/{name}")
        j.CreateBody0Rel().SetTargets([b0])
        j.CreateBody1Rel().SetTargets([b1])
        j.CreateLocalPos0Attr(Gf.Vec3f(*pos0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*pos1))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        if axis:
            j.CreateAxisAttr(axis)
        if limits:
            j.CreateLowerLimitAttr(limits[0])
            j.CreateUpperLimitAttr(limits[1])
        if drive:
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), drive)
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(0.0)
            d.CreateDampingAttr(damping)
            d.CreateTargetVelocityAttr(0.0)
            d.CreateMaxForceAttr(1e4)
        return j

    def save(self):
        path = OUT / self.name / f"{self.name}.usda"
        path.parent.mkdir(parents=True, exist_ok=True)
        lay = self.st.GetRootLayer()
        lay.documentation = f"{self.name}: care-room furniture for DX_Hub_Building_add_obj_v1 (scripts/20_furniture_assets.py)"
        lay.Export(str(path))
        print("wrote", path)


# ------------------------------------------------------------------ Bed
def build_bed():
    c = A["Bed"]
    W, L = c["size"]
    a = Asset("Bed", "bed")
    white, mint = a.material("white", 0.5), a.material("mint_bed", 0.5)
    fr = a.body("Frame", c["mass_frame"], "bed_frame")
    hb, fb, t = c["headboard"], c["footboard"], c["side_rail_t"]
    zt, (tk_h, tk_in) = c["base_top_z"], c["toe_kick"]
    y_head_in = L / 2 - hb["depth"]                                 # headboard front face
    y_foot_in = -L / 2 + fb["t"]
    a.box(fr, (0, (y_head_in + y_foot_in) / 2, tk_h / 2), (W - 2 * tk_in, y_head_in - y_foot_in, tk_h), white, name="toe_kick")
    a.box(fr, (-t / 2, (y_head_in + y_foot_in) / 2, (tk_h + zt) / 2), (W - t, y_head_in - y_foot_in, zt - tk_h), white, name="base")
    a.box(fr, (W / 2 - t / 2, (y_head_in + y_foot_in) / 2, zt / 2 + 0.005), (t, y_head_in - y_foot_in, zt - 0.01), mint, name="side_rail_east")
    for k, (y0, y1) in enumerate(((y_foot_in + 0.02, -0.01), (0.01, y_head_in - 0.02))):  # two drawer fronts on the west side (visual)
        a.box(fr, (-W / 2 - 0.002, (y0 + y1) / 2, (tk_h + 0.02 + zt - 0.03) / 2), (0.006, y1 - y0, zt - 0.03 - tk_h - 0.02), white,
              collide=False, name=f"drawer_front_{k}")
    a.box(fr, (0, L / 2 - hb["depth"] / 2, hb["block_h"] / 2), (W, hb["depth"] - hb["panel_t"], hb["block_h"]), mint, name="headboard_block")
    a.box(fr, (0, L / 2 - hb["panel_t"] / 2, hb["panel_h"] / 2), (W, hb["panel_t"], hb["panel_h"]), mint, name="headboard_panel")
    a.box(fr, (0, -L / 2 + fb["t"] / 2, fb["h"] / 2), (W, fb["t"], fb["h"]), mint, name="footboard")
    m = c["mattress"]
    mz = zt + m["size"][2] / 2
    my = (y_head_in + y_foot_in) / 2
    mt = a.body("Mattress", m["mass"], "mattress", translate=(0, my, mz))
    a.mesh(mt, superquadric((0, 0, 0), m["size"], e=0.15), a.material("mattress", 0.85), smooth=True, collide=False, name="mattress")
    # collider 1 cm above the visible top (guide purpose): a cloth lying on it would otherwise sit exactly in the
    # rendered mattress surface and z-fight with it (surface deformables ignore the collision rest offset)
    cb = a.box(mt, (0, 0, m["collider_lift"] / 2), (m["size"][0], m["size"][1], m["size"][2] + m["collider_lift"]),
               a.material("mattress", 0.85), name="collider")
    cb.CreatePurposeAttr(UsdGeom.Tokens.guide)
    a.joint("fixed", "mattress_to_frame", fr, mt, (0, my, mz), (0, 0, 0))
    top = zt + m["size"][2] + m["collider_lift"]
    # pillow: volume deformable (auto hexahedral simulation mesh cooked from the render mesh)
    p = c["pillow"]
    py = y_head_in - p["gap_to_headboard"] - p["size"][1] / 2
    root = UsdGeom.Xform.Define(a.st, f"{a.R}/Pillow")
    root.AddTranslateOp().Set(Gf.Vec3d(0, py, top + p["size"][2] / 2 + 0.005))
    a.label(root.GetPrim(), "pillow")
    pm = a.mesh(str(root.GetPath()), superquadric((0, 0, 0), p["size"], e=0.45, nu=48, nv=24), a.material("pillow", 0.9),
                collide=False, smooth=True, name="mesh")
    pp = str(root.GetPath())
    assert deformableUtils.create_auto_volume_deformable_hierarchy(
        a.st, pp, simulation_tetmesh_path=f"{pp}/simulation_mesh", collision_tetmesh_path=f"{pp}/collision_mesh",
        cooking_src_mesh_path=str(pm.GetPath()), simulation_hex_mesh_enabled=True, cooking_src_simplification_enabled=False,
        set_visibility_with_guide_purpose=True)
    rp = a.st.GetPrimAtPath(pp)
    if rp.HasAttribute("physxDeformableBody:resolution"):
        rp.GetAttribute("physxDeformableBody:resolution").Set(p["hex_resolution"])
    deformableUtils.add_deformable_material(a.st, f"{a.R}/Looks/PillowPhysics", density=p["density"], youngs_modulus=p["youngs_modulus"],
                                            poissons_ratio=p["poissons_ratio"], static_friction=p["friction"][0], dynamic_friction=p["friction"][1])
    pmat = a.st.GetPrimAtPath(f"{a.R}/Looks/PillowPhysics")
    pmat.ApplyAPI("PhysxDeformableMaterialAPI")
    pmat.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(p["elasticity_damping"])
    physicsUtils.add_physics_material_to_prim(a.st, rp, f"{a.R}/Looks/PillowPhysics")
    get_physx_cooking_interface().cook_auto_deformable_body(pp)
    # blanket: surface deformable triangle sheet lying flat just above the mattress, foot end inside the footboard
    b = c["blanket"]
    bw, bl = b["size"]
    nx, ny = int(round(bw / b["res"])) + 1, int(round(bl / b["res"])) + 1
    y0 = y_foot_in + b["foot_inset"]
    V = np.array([[x, y, top + b["lift"]] for y in np.linspace(y0, y0 + bl, ny) for x in np.linspace(-bw / 2, bw / 2, nx)])
    F = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            q = j * nx + i
            F += [[q, q + 1, q + nx + 1], [q, q + nx + 1, q + nx]] if (i + j) % 2 == 0 else [[q, q + 1, q + nx], [q + 1, q + nx + 1, q + nx]]
    bm = a.mesh(a.R, (V, F), a.material("sheet", 0.9), collide=False, name="Blanket", label="blanket")
    bm.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(bm).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray([Gf.Vec2f(float(v[0] / bw), float((v[1] - y0) / bl)) for v in V]))
    bp = str(bm.GetPath())
    assert deformableUtils.set_physics_surface_deformable_body(a.st, bp)
    deformableUtils.add_surface_deformable_material(a.st, f"{a.R}/Looks/BlanketPhysics", density=b["density"], youngs_modulus=b["youngs_modulus"],
                                                    poissons_ratio=b["poissons_ratio"], static_friction=b["friction"][0], dynamic_friction=b["friction"][1],
                                                    surface_thickness=b["thickness"], surface_bend_stiffness=b["bend_stiffness"])
    physicsUtils.add_physics_material_to_prim(a.st, bm.GetPrim(), f"{a.R}/Looks/BlanketPhysics")
    bm.GetPrim().ApplyAPI("PhysxBaseDeformableBodyAPI")
    bm.GetPrim().ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    bm.GetPrim().GetAttribute("physxDeformableBody:selfCollision").Set(b["self_collision"])
    bm.GetPrim().ApplyAPI("PhysxCollisionAPI")
    bm.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(b["rest_offset"])
    bm.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(b["contact_offset"])
    for _ in range(10):
        app.update()
    a.save()


# ------------------------------------------------------------------ Sofa
def build_sofa(name="Sofa3"):
    c = A[name]
    W, D, H = c["size"]
    a = Asset(name, "sofa")
    blk, chrome, trim = a.material("leather_black", 0.35), a.material("chrome", 0.2, 1.0), a.material("white", 0.4)
    bd = a.body("Body", c["mass"], "sofa")
    lh, bh, aw, ah, bt = c["leg_h"], c["base_h"], c["arm"]["w"], c["arm"]["h"], c["back"]["t"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            a.mesh(bd, cylinder((sx * (W / 2 - 0.07), sy * (D / 2 - 0.07), 0), 0.012, 0.022, lh), chrome, name=f"leg_{sx}_{sy}".replace("-", "m"))
    a.box(bd, (0, 0, lh + bh / 2), (W - 2 * aw, D, bh), blk, name="base")
    for sx in (-1, 1):
        x = sx * (W / 2 - aw / 2)
        a.mesh(bd, superquadric((x, 0, (lh + ah) / 2 + 0.01), (aw, D, ah - lh + 0.02), e=0.12), blk, smooth=True, name=f"arm_{'L' if sx < 0 else 'R'}")
        xo = x + sx * (aw / 2 - 0.006)                                # thin white piping on the outer front edge of the arm
        a.box(bd, (xo, D / 2 - 0.004, (lh + ah) / 2 + 0.005), (0.008, 0.012, ah - lh - 0.01), trim, collide=False, name=f"arm_trim_{'L' if sx < 0 else 'R'}")
    a.mesh(bd, superquadric((0, -D / 2 + bt / 2, (lh + H) / 2), (W - 2 * aw + 0.02, bt, H - lh), e=0.12), blk, smooth=True, name="back")
    n, sd, stt = c["seat"]["n"], c["seat"]["depth"], c["seat"]["t"]
    sw = (W - 2 * aw) / n
    for k in range(n):
        x = -W / 2 + aw + sw * (k + 0.5)
        a.mesh(bd, superquadric((x, D / 2 - sd / 2 - 0.01, lh + bh + stt / 2), (sw - 0.006, sd, stt), e=0.25), blk, smooth=True, name=f"seat_{k}")
        a.mesh(bd, superquadric((x, -D / 2 + bt + 0.06, lh + bh + stt + 0.17), (sw - 0.006, 0.12, 0.34), e=0.3), blk, smooth=True,
               name=f"back_cushion_{k}")
    a.save()


# ------------------------------------------------------------------ Sofa table
def build_table():
    c = A["SofaTable"]
    W, D, H = c["size"]
    a = Asset("SofaTable", "table")
    top, side, chrome = a.material("table_top", 0.45), a.material("table_side", 0.5), a.material("chrome", 0.3, 1.0)
    bd = a.body("Body", c["mass"], "table")
    tt, pt, ins, ch = c["top_t"], c["panel_t"], c["panel_inset"], c["caster_h"]
    a.box(bd, (0, 0, H - tt / 2), (W, D, tt), top, name="top")
    for sy in (-1, 1):
        a.box(bd, (0, sy * (D / 2 - ins - pt / 2), (ch + H - tt) / 2), (W - 2 * ins, pt, H - tt - ch), side, name=f"side_{'S' if sy < 0 else 'N'}")
    a.box(bd, (0, 0, ch + pt / 2), (W - 2 * ins, D - 2 * ins, pt), side, name="shelf")
    for sx in (-1, 1):
        for sy in (-1, 1):
            a.mesh(bd, cylinder((sx * (W / 2 - ins - 0.04), sy * (D / 2 - ins - 0.04), 0), 0.022, 0.022, ch), chrome,
                   name=f"caster_{sx}_{sy}".replace("-", "m"))
    a.save()


# ------------------------------------------------------------------ Plant
def build_plant():
    c = A["Plant"]
    a = Asset("Plant", "plant")
    white, soil, leaf, trunk = a.material("planter", 0.4), a.material("soil", 0.95), a.material("leaf", 0.45), a.material("trunk", 0.8)
    bd = a.body("Body", c["mass"], "plant")
    ts, th = c["tray"]
    a.box(bd, (0, 0, th / 2), (ts, ts, th), white, name="tray")
    pot = c["pot"]
    a.mesh(bd, frustum_box((0, 0, th), pot["top"], pot["bottom"], pot["h"]), white, name="pot")
    zt = th + pot["h"]
    a.box(bd, (0, 0, zt + 0.003), (pot["top"] - 0.03, pot["top"] - 0.03, 0.006), soil, collide=False, name="soil")
    a.mesh(bd, cylinder((0, 0, zt), 0.016, 0.01, c["foliage_h"] * 0.7), trunk, collide=False, name="trunk")
    rng = np.random.default_rng(7)
    for k in range(11):                                              # rubber-plant leaves: broad ovals around the stem
        z = zt + 0.08 + c["foliage_h"] * (0.12 + 0.75 * k / 10)
        yaw, tilt = 137.5 * k, 25 + 20 * rng.random()
        L_, W_ = 0.17 - 0.006 * k, 0.09 - 0.003 * k
        V, F = superquadric((0, 0, 0), (L_, W_, 0.008), e=0.9, nu=16, nv=8)
        V[:, 0] += L_ / 2 + 0.01
        tz = math.radians(tilt)
        V = V @ np.array([[math.cos(tz), 0, -math.sin(tz)], [0, 1, 0], [math.sin(tz), 0, math.cos(tz)]]).T
        V = rot_z(V, yaw) + [0, 0, z]
        a.mesh(bd, (V, F), leaf, collide=False, smooth=True, name=f"leaf_{k:02d}")
    a.save()


# ------------------------------------------------------------------ cabinets (articulations)
def cabinet_shell(a, bd, W, D, H, t, plinth, mat, shelves=(), back_mat=None, open_front_above=None):
    """Side panels, top, bottom, back and fixed shelves of a cabinet whose front is at +y."""
    a.box(bd, (-W / 2 + t / 2, 0, H / 2), (t, D, H), mat, name="side_L")
    a.box(bd, (W / 2 - t / 2, 0, H / 2), (t, D, H), mat, name="side_R")
    a.box(bd, (0, 0, H - t / 2), (W - 2 * t, D, t), mat, name="top")
    a.box(bd, (0, 0, plinth + t / 2), (W - 2 * t, D, t), mat, name="bottom")
    a.box(bd, (0, D / 2 - 0.03, plinth / 2), (W - 2 * t, 0.018, plinth), mat, name="plinth")
    a.box(bd, (0, -D / 2 + 0.004, H / 2), (W - 2 * t, 0.008, H - 2 * t), back_mat or mat, name="back")
    for k, z in enumerate(shelves):
        a.box(bd, (0, 0.0, z - t / 2), (W - 2 * t, D - 0.01, t), mat, name=f"shelf_{k}")


def drawer(a, name, W, D, z0, z1, front_mat, box_mat, mass, t=0.016):
    """Drawer link: front panel flush with the cabinet front (+y) and an open tray behind it."""
    lk = a.body(name, mass, "drawer")
    h = z1 - z0
    a.box(lk, (0, D / 2 - t / 2, (z0 + z1) / 2), (W - 0.006, t, h - 0.004), front_mat, name="front")
    dd, dw, dh = D - 0.06, W - 0.05, h - 0.05
    yb = D / 2 - t - dd / 2
    a.box(lk, (0, yb, z0 + 0.02 + 0.006), (dw, dd, 0.012), box_mat, name="tray_bottom")
    for sx in (-1, 1):
        a.box(lk, (sx * (dw / 2 - 0.006), yb, z0 + 0.02 + dh / 2), (0.012, dd, dh), box_mat, name=f"tray_side_{'L' if sx < 0 else 'R'}")
    a.box(lk, (0, yb - dd / 2 + 0.006, z0 + 0.02 + dh / 2), (dw, 0.012, dh), box_mat, name="tray_back")
    a.box(lk, (0, D / 2 + 0.012, (z0 + z1) / 2 + h * 0.15), (min(0.16, W * 0.3), 0.012, 0.014), a.material("chrome", 0.25, 1.0),
          collide=False, name="handle")
    return lk


def build_wardrobe():
    c = A["Wardrobe800"]
    W, D, H = c["size"]
    t = c["panel_t"]
    a = Asset("Wardrobe800", "wardrobe")
    UsdPhysics.ArticulationRootAPI.Apply(a.st.GetPrimAtPath(a.R))
    white, chrome = a.material("white", 0.45), a.material("chrome", 0.25, 1.0)
    bd = a.body("Body", c["mass_body"], "wardrobe")
    dz0, dz1 = c["doors"]["z"]
    cabinet_shell(a, bd, W, D - 0.02, H, t, c["plinth_h"], white, shelves=(dz0 - 0.005, 1.78))
    a.mesh(bd, cylinder((0, -0.02, 1.68), 0.012, 0.012, W - 2 * t - 0.01, axis="x"), chrome, collide=False, name="hanger_rod")
    dw = W / 2 - 0.003
    for side, sx in (("L", -1), ("R", 1)):                           # hinges at the outer edges, opening toward +y
        hx, hy = sx * W / 2, D / 2 - 0.01
        lk = a.body(f"Door{side}", c["doors"]["mass"], "cabinet_door", translate=(hx, hy, 0))
        a.box(lk, (-sx * dw / 2, t / 2, (dz0 + dz1) / 2), (dw, t, dz1 - dz0), white, name="panel")       # overlay door in front of the carcass
        a.box(lk, (-sx * (dw - 0.045), t + 0.006, (dz0 + dz1) / 2 + 0.05), (0.014, 0.012, 0.32), chrome, collide=False, name="handle")
        lim = (0, c["doors"]["open_deg"]) if sx < 0 else (-c["doors"]["open_deg"], 0)
        a.joint("revolute", f"door_{side}", bd, lk, (hx, hy, 0), (0, 0, 0), axis="Z", limits=lim, damping=c["doors"]["damping"], drive="angular")
    for k, dr in enumerate(c["drawers"]):
        lk = drawer(a, f"Drawer{k}", W - 2 * t, D - 0.02, dr["z"][0], dr["z"][1], white, white, dr["mass"])
        a.joint("prismatic", f"drawer_{k}", bd, lk, (0, 0, 0), (0, 0, 0), axis="Y", limits=(0, dr["travel"]), damping=c["drawer_damping"],
                drive="linear")
    a.save()


def build_vanity():
    c = A["Vanity640"]
    W, D, H = c["size"]
    t = c["panel_t"]
    a = Asset("Vanity640", "vanity")
    UsdPhysics.ArticulationRootAPI.Apply(a.st.GetPrimAtPath(a.R))
    mint, white, mirror = a.material("mint_cabinet", 0.5), a.material("white", 0.45), a.material("mirror", 0.03, 1.0)
    bd = a.body("Body", c["mass_body"], "vanity")
    dz = c["desk_z"]
    drz = c["drawers"][0]["z"]
    cabinet_shell(a, bd, W, D, H, t, c["plinth_h"], white, shelves=(c["cubby_z"],), back_mat=mint)
    a.box(bd, (0, 0, dz + t / 2), (W - 2 * t, D, t), mint, name="desk_top")
    a.box(bd, (0, 0, drz[0] - t / 2), (W - 2 * t, D - 0.01, t), white, name="drawer_bottom_rail")
    rx, ry, rz = c["riser"]
    a.box(bd, (0, -D / 2 + 0.01 + ry / 2, dz + t + rz - 0.009), (rx, ry, 0.018), mint, name="riser_top")
    for sx in (-1, 1):
        a.box(bd, (sx * (rx / 2 - 0.009), -D / 2 + 0.01 + ry / 2, dz + t + (rz - 0.018) / 2), (0.018, ry, rz - 0.018), mint, name=f"riser_leg_{sx}".replace("-", "m"))
    m = c["mirror"]
    a.mesh(bd, stadium((-0.06, -D / 2 + 0.03, m["z"] + m["h"] / 2), m["w"] + 0.03, m["h"] + 0.03, 0.012), white, collide=False, name="mirror_frame")
    a.mesh(bd, stadium((-0.06, -D / 2 + 0.038, m["z"] + m["h"] / 2), m["w"], m["h"], 0.006), mirror, collide=False, name="mirror")
    a.box(bd, (W / 2 - 0.06, -D / 2 + 0.03, m["z"] + m["h"] / 2), (0.025, 0.02, m["h"] + 0.15), white, collide=False, name="mirror_post")
    dr = c["drawers"][0]
    lk = drawer(a, "Drawer0", W - 2 * t, D, drz[0], drz[1], white, white, dr["mass"])
    a.joint("prismatic", "drawer_0", bd, lk, (0, 0, 0), (0, 0, 0), axis="Y", limits=(0, dr["travel"]), damping=c["drawer_damping"], drive="linear")
    a.save()


def build_closet():
    c = A["Closet640"]
    W, D, H = c["size"]
    t = c["panel_t"]
    a = Asset("Closet640", "closet")
    UsdPhysics.ArticulationRootAPI.Apply(a.st.GetPrimAtPath(a.R))
    mint, white, chrome = a.material("mint_cabinet", 0.5), a.material("white", 0.45), a.material("chrome", 0.25, 1.0)
    bd = a.body("Body", c["mass_body"], "closet")
    cz = c["cabinet_top_z"]
    cabinet_shell(a, bd, W, D, H, t, c["plinth_h"], white, shelves=(c["cubby_z"],), back_mat=mint)
    a.box(bd, (0, 0, cz + t / 2), (W - 2 * t, D, t), mint, name="cabinet_top")
    a.box(bd, (0, 0, c["drawers"][1]["z"][0] - 0.005), (W - 2 * t, D - 0.01, 0.01), white, name="drawer_rail")
    a.mesh(bd, cylinder((0, 0.0, c["rail_z"]), 0.012, 0.012, W - 2 * t - 0.01, axis="x"), white, collide=False, name="hanger_rail")
    for k in range(3):                                               # coat hooks on the front edge of the cubby shelf
        a.box(bd, (-W / 2 + 0.15 + k * (W - 0.3) / 2, D / 2 - 0.03, c["cubby_z"] - 0.05), (0.02, 0.06, 0.02), white, collide=False, name=f"hook_{k}")
    for k, dr in enumerate(c["drawers"]):
        lk = drawer(a, f"Drawer{k}", W - 2 * t, D, dr["z"][0], dr["z"][1], white, white, dr["mass"])
        a.joint("prismatic", f"drawer_{k}", bd, lk, (0, 0, 0), (0, 0, 0), axis="Y", limits=(0, dr["travel"]), damping=c["drawer_damping"],
                drive="linear")
    a.save()


# ------------------------------------------------------------------ kitchen
def build_stool():
    c = A["BarStool"]
    W, D, H = c["size"]
    a = Asset("BarStool", "chair")
    blue, wire = a.material("stool_blue", 0.7), a.material("wire_white", 0.35)
    bd = a.body("Body", c["mass"], "chair")
    st, ts, r, sp = c["seat_top_z"], c["seat_t"], c["rod_r"], c["floor_splay"]
    a.mesh(bd, superquadric((0, 0, st - ts / 2), (W, D - 0.02, ts), e=0.35), blue, smooth=True, name="seat")
    a.mesh(bd, superquadric((0, -D / 2 + 0.04, H - 0.035), (W * 0.95, 0.07, 0.07), e=0.5), blue, smooth=True, name="back_lip")
    zs = st - ts                                                      # frame top (under the seat)
    xs, ys = W / 2 - 0.03, D / 2 - 0.04
    for sx in (-1, 1):                                                # side sled loops
        top_f, top_b = (sx * xs, ys, zs), (sx * xs, -ys, zs)
        bot_f, bot_b = (sx * (xs + sp), ys + sp, r), (sx * (xs + sp), -ys - sp, r)
        for k, (p0, p1) in enumerate(((top_f, top_b), (top_f, bot_f), (top_b, bot_b), (bot_f, bot_b))):
            a.mesh(bd, rod(p0, p1, r), wire, name=f"rod_{'L' if sx < 0 else 'R'}{k}")
    fz = c["footrest_z"]
    fy = ys + sp * (1 - fz / zs)
    fx = xs + sp * (1 - fz / zs)
    a.mesh(bd, rod((-fx, fy, fz), (fx, fy, fz), r), wire, name="footrest")
    a.mesh(bd, rod((-xs, 0, zs), (xs, 0, zs), r), wire, collide=False, name="seat_bar")
    a.save()


def build_showcase():
    c = A["Showcase"]
    W, D, H = c["size"]
    a = Asset("Showcase", "showcase")
    UsdPhysics.ArticulationRootAPI.Apply(a.st.GetPrimAtPath(a.R))
    steel, black, glass, chrome = a.material("stainless", 0.3, 0.9), a.material("black", 0.4), a.material("glass", 0.02, 0.0, 0.1), a.material("chrome", 0.25, 1.0)
    bd = a.body("Body", c["mass_body"], "showcase")
    bh, rt, pst = c["base_h"], c["roof_t"], c["post"]
    a.box(bd, (0, 0, bh / 2), (W, D, bh), steel, name="base")
    a.box(bd, (0, D / 2 + 0.002, bh * 0.45), (W * 0.7, 0.004, bh * 0.45), a.material("dark_grey", 0.6), collide=False, name="vent_grille")
    a.box(bd, (0, 0, bh + 0.005), (W - 0.02, D - 0.02, 0.01), steel, name="floor")
    for sx in (-1, 1):
        for sy in (-1, 1):
            a.box(bd, (sx * (W / 2 - pst / 2), sy * (D / 2 - pst / 2), (bh + H) / 2), (pst, pst, H - bh), black, name=f"post_{sx}_{sy}".replace("-", "m"))
    a.box(bd, (0, 0, H - 0.004), (W - 2 * pst, D - 2 * pst, 0.008), glass, name="roof_glass")
    for sy in (-1, 1):
        a.box(bd, (0, sy * (D / 2 - pst / 2), H - rt / 2), (W, pst, rt), black, name=f"roof_frame_y{'m' if sy < 0 else 'p'}")
    for sx in (-1, 1):
        a.box(bd, (sx * (W / 2 - pst / 2), 0, H - rt / 2), (pst, D, rt), black, name=f"roof_frame_x{'m' if sx < 0 else 'p'}")
    gh = H - rt - bh
    lt = UsdLux.RectLight.Define(a.st, f"{a.R}/InteriorLight")          # LED strip under the roof (showcases are lit inside)
    lt.CreateWidthAttr(W - 0.1)
    lt.CreateHeightAttr(D - 0.15)
    lt.CreateIntensityAttr(c["interior_light"])
    lt.CreateColorTemperatureAttr(5500.0)
    lt.CreateEnableColorTemperatureAttr(True)
    lt.AddTranslateOp().Set(Gf.Vec3d(0, 0, H - rt - 0.01))
    a.box(bd, (0, D / 2 - 0.006, bh + gh / 2), (W - 2 * pst, 0.006, gh), glass, name="front_glass")
    for sx in (-1, 1):
        a.box(bd, (sx * (W / 2 - 0.006), 0, bh + gh / 2), (0.006, D - 2 * pst, gh), glass, name=f"side_glass_{'L' if sx < 0 else 'R'}")
    a.box(bd, (0, -D / 2 + 0.004, bh + 0.01), (W - 2 * pst, 0.008, 0.02), black, collide=False, name="rear_track_bottom")
    a.box(bd, (0, -D / 2 + 0.004, H - rt - 0.01), (W - 2 * pst, 0.008, 0.02), black, collide=False, name="rear_track_top")
    for k, z in enumerate(c["shelves_z"]):                            # wire shelves: frame + lengthwise wires
        sw, sd = W - 0.09, D - 0.12
        for p0, p1 in (((-sw / 2, -sd / 2, z), (sw / 2, -sd / 2, z)), ((-sw / 2, sd / 2, z), (sw / 2, sd / 2, z)),
                       ((-sw / 2, -sd / 2, z), (-sw / 2, sd / 2, z)), ((sw / 2, -sd / 2, z), (sw / 2, sd / 2, z))):
            a.mesh(bd, rod(p0, p1, 0.004, n=8), chrome, collide=False, name=f"shelf{k}_frame_{a.n}")
        for x in np.linspace(-sw / 2 + 0.03, sw / 2 - 0.03, 12):
            a.mesh(bd, rod((x, -sd / 2, z + 0.004), (x, sd / 2, z + 0.004), 0.002, n=6), chrome, collide=False, name=f"shelf{k}_wire_{a.n}")
        a.box(bd, (0, 0, z), (sw, sd, 0.006), chrome, name=f"shelf{k}_collider").CreatePurposeAttr(UsdGeom.Tokens.guide)
    dw = (W - 2 * pst) / 2 + 0.01
    for k, (sx, ty) in enumerate(((-1, -D / 2 + 0.012), (1, -D / 2 + 0.026))):   # two tracks, doors overlap at the middle
        lk = a.body(f"Door{'L' if sx < 0 else 'R'}", c["doors"]["mass"], "sliding_door")
        cx = sx * dw / 2
        a.box(lk, (cx, ty, bh + gh / 2), (dw, 0.006, gh - 0.03), glass, name="glass")
        a.box(lk, (cx, ty, bh + 0.025), (dw, 0.012, 0.02), black, collide=False, name="rail_bottom")
        a.box(lk, (cx - sx * (dw / 2 - 0.02), ty - 0.008, bh + gh / 2), (0.02, 0.01, 0.12), black, collide=False, name="grip")
        lim = (0, c["doors"]["travel"]) if sx < 0 else (-c["doors"]["travel"], 0)
        a.joint("prismatic", f"door_{'L' if sx < 0 else 'R'}", bd, lk, (0, 0, 0), (0, 0, 0), axis="X", limits=lim, damping=c["doors"]["damping"], drive="linear")
    a.save()


def build_trash():
    c = A["TrashBin3"]
    a = Asset("TrashBin3", "trash_bin")
    white, body = a.material("white", 0.5), a.material("bin_body", 0.6)
    pw, pd, ph = c["plinth"]
    pl = a.body("Plinth", c["mass_plinth"], "trash_bin_base")
    a.box(pl, (0, 0, ph / 2), (pw, pd, ph), white, name="plinth")
    b = c["bin"]
    (bw0, bd0), (bw1, bd1), bh, t = b["bottom"], b["top"], b["h"], b["wall"]
    lh = c["lid"]["h"]
    for k, lidc in enumerate(c["lids"]):
        cx = (1 - k) * (pw / 3)                                         # lids are listed left -> right seen from the front (+y): left = +x
        lk = a.body(f"Bin{k}", b["mass"], "trash_bin")
        z0 = ph
        a.box(lk, (cx, 0, z0 + t / 2), (bw0, bd0, t), body, name="bottom")
        for sx in (-1, 1):                                            # tapered side walls (convex quads)
            V = np.array([[sx * bw0 / 2, -bd0 / 2, z0], [sx * bw0 / 2, bd0 / 2, z0], [sx * bw1 / 2, bd1 / 2, z0 + bh], [sx * bw1 / 2, -bd1 / 2, z0 + bh]])
            V = np.vstack([V, V - [sx * t, 0, 0]]) + [cx, 0, 0]
            a.mesh(lk, (V, [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]), body, name=f"wall_x{'m' if sx < 0 else 'p'}").CreateDoubleSidedAttr(True)
        for sy in (-1, 1):
            V = np.array([[-bw0 / 2, sy * bd0 / 2, z0], [bw0 / 2, sy * bd0 / 2, z0], [bw1 / 2, sy * bd1 / 2, z0 + bh], [-bw1 / 2, sy * bd1 / 2, z0 + bh]])
            V = np.vstack([V, V - [0, sy * t, 0]]) + [cx, 0, 0]
            a.mesh(lk, (V, [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]]), body, name=f"wall_y{'m' if sy < 0 else 'p'}").CreateDoubleSidedAttr(True)
        sl, sd = c["lid"]["slot"]                                     # lid: coloured frame around a drop slot
        lm = a.material(lidc, 0.5)
        zl = z0 + bh + lh / 2
        a.box(lk, (cx, -(sd / 2 + (bd1 - sd) / 4), zl), (bw1 + 0.01, (bd1 - sd) / 2, lh), lm, name="lid_back")
        a.box(lk, (cx, sd / 2 + (bd1 - sd) / 4, zl), (bw1 + 0.01, (bd1 - sd) / 2, lh), lm, name="lid_front")
        for sx in (-1, 1):
            a.box(lk, (cx + sx * (sl / 2 + (bw1 - sl) / 4), 0, zl), ((bw1 - sl) / 2 + 0.005, sd, lh), lm, name=f"lid_side_{sx}".replace("-", "m"))
        a.box(lk, (cx, sd / 2 + (bd1 - sd) / 4, z0 + bh + lh + 0.001), (0.08, 0.06, 0.002), a.material("white", 0.5), collide=False, name="label")
    a.save()


def build_extinguisher():
    c = A["FireExtinguisher"]
    a = Asset("FireExtinguisher", "fire_extinguisher")
    red, black, chrome = a.material("ext_red", 0.35), a.material("black", 0.5), a.material("chrome", 0.25, 1.0)
    sw, sd, sh = c["stand"]
    st_ = a.body("Stand", 1.0, "fire_extinguisher_stand")
    a.box(st_, (0, 0, 0.01), (sw, sd, 0.02), red, name="base")
    for sx in (-1, 1):
        a.box(st_, (sx * (sw / 2 - 0.006), 0, sh / 2), (0.012, sd, sh), red, name=f"side_x{sx}".replace("-", "m"))
    a.box(st_, (0, -sd / 2 + 0.006, sh / 2), (sw, 0.012, sh), red, name="back")
    ex = a.body("Extinguisher", c["mass"], "fire_extinguisher")
    r, h = c["body_r"], c["body_h"]
    a.mesh(ex, cylinder((0, 0, 0.02), r, r, h, n=24), red, name="body")
    a.mesh(ex, superquadric((0, 0, 0.02 + h), (2 * r, 2 * r, 0.06), e=1.0, nu=24, nv=12), red, smooth=True, name="shoulder")
    a.mesh(ex, cylinder((0, 0, 0.02 + h + 0.02), 0.016, 0.016, c["head_h"]), chrome, name="valve")
    a.box(ex, (0.03, 0, 0.02 + h + c["head_h"]), (0.11, 0.02, 0.012), black, collide=False, name="handle")
    a.mesh(ex, superquadric((0, 0.03, 0.02 + h + 0.045), (0.035, 0.012, 0.035), e=1.0, nu=16, nv=8), a.material("white", 0.4), smooth=True, collide=False, name="gauge")
    a.mesh(ex, rod((-0.02, 0, 0.02 + h + 0.05), (-r - 0.01, 0, 0.12), 0.008), black, collide=False, name="hose")
    a.save()


def build_purifier():
    c = A["WaterPurifier"]
    W, D, H = c["size"]
    a = Asset("WaterPurifier", "water_purifier")
    white, dark, chrome = a.material("white", 0.35), a.material("dark_grey", 0.4), a.material("chrome", 0.25, 1.0)
    bd = a.body("Body", c["mass"], "water_purifier")
    z0, z1 = c["dispense_z"]
    a.mesh(bd, superquadric((0, 0, z0 / 2), (W, D, z0), e=0.1), white, smooth=True, name="lower")
    a.mesh(bd, superquadric((0, -0.06, (z0 + z1) / 2), (W, D - 0.12, z1 - z0), e=0.1), white, smooth=True, name="recess_back")
    a.mesh(bd, superquadric((0, 0, (z1 + H) / 2), (W, D, H - z1), e=0.1), white, smooth=True, name="upper")
    a.box(bd, (0, D / 2 - 0.08, z0 + 0.012), (W * 0.6, 0.14, 0.024), dark, collide=False, name="drip_tray")
    a.box(bd, (0, D / 2 - 0.05, z1 - 0.025), (0.05, 0.03, 0.05), chrome, collide=False, name="tap")
    a.box(bd, (0, D / 2 + 0.002, H - 0.13), (0.2, 0.004, 0.12), a.material("black", 0.15), collide=False, name="display")
    a.save()


def rrect(w, h, rb, rf, n=8):
    """Rounded rectangle (x width w, y depth h), CCW seen from +z; back corners (y < 0) radius rb, front corners rf."""
    pts = []
    for (cx, cy, r, a0) in ((w / 2 - rb, -h / 2 + rb, rb, -90), (w / 2 - rf, h / 2 - rf, rf, 0), (-w / 2 + rf, h / 2 - rf, rf, 90),
                            (-w / 2 + rb, -h / 2 + rb, rb, 180)):
        for k in range(n + 1):
            t = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(t), cy + r * math.sin(t)))
    return np.array(pts)


def loft(c0, z0, c1, z1):
    """Quad strip between two closed contours with the same point count."""
    n = len(c0)
    V = np.vstack([np.c_[c0, np.full(n, z0)], np.c_[c1, np.full(n, z1)]])
    return V, [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]


def fan(c, z, up=True):
    n = len(c)
    V = np.vstack([np.c_[c, np.full(n, z)], [[*c.mean(0), z]]])
    return V, [([n, k, (k + 1) % n] if up else [n, (k + 1) % n, k]) for k in range(n)]


def ring(c_out, c_in, z):
    """Flat ring between two contours (shapely constrained triangulation)."""
    import shapely
    from shapely.geometry import Polygon
    tri = shapely.constrained_delaunay_triangles(Polygon(c_out).difference(Polygon(c_in)))
    V, F = [], []
    for t in tri.geoms:
        q = np.asarray(t.exterior.coords)[:3]
        if (q[1][0] - q[0][0]) * (q[2][1] - q[0][1]) - (q[1][1] - q[0][1]) * (q[2][0] - q[0][0]) < 0:
            q = q[::-1]
        F.append([len(V), len(V) + 1, len(V) + 2])
        V += [[x, y, z] for x, y in q]
    return np.array(V), F


def hull(P):
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float)
    h = ConvexHull(P)
    F = []
    for simp, eq in zip(h.simplices, h.equations):                     # outward winding
        a_, b_, c_ = P[simp]
        F.append([int(i) for i in (simp if np.dot(np.cross(b_ - a_, c_ - a_), eq[:3]) > 0 else simp[::-1])])
    return P, F


def build_sink():
    """Top-mount stainless sink, measured (user 2026-10-15): outer 600 x 445, total depth 200; bowl opening 550 x 370, 170 deep.
    Local origin = counter-top level at the sink centre, front (+y) = work side, faucet behind (-y). variantSet faucet = pullout | straight."""
    c = A["Sink"]
    a = Asset("Sink", "sink")
    steel, chrome, dark = a.material("stainless", 0.4, 0.85), a.material("chrome", 0.12, 1.0), a.material("dark_grey", 0.5)
    bd = a.body("Body", c["mass"], "sink")
    (ow, od), (iw, idp), (bw, bdp) = c["outer"], c["inner_top"], c["inner_bottom"]
    rb, rf, ft, t = c["corner_r"]["back"], c["corner_r"]["front"], c["flange_t"], c["wall"]
    hi, ho = c["depth_inner"], c["depth_outer"] - ft
    top_in = rrect(iw, idp, rb, rf)
    top_out = rrect(iw + 2 * t, idp + 2 * t, rb + t, rf + t)
    bot_in = rrect(bw, bdp, rb, rf * 0.8)
    bot_out = rrect(bw + 2 * t, bdp + 2 * t, rb + t, rf * 0.8 + t)
    fl_out = rrect(ow, od, 0.03, 0.05)
    parts = [("bowl_inner_wall", loft(top_in, 0.0, bot_in, -hi)), ("bowl_inner_bottom", fan(bot_in, -hi, up=True)),
             ("bowl_outer_wall", loft(top_out, 0.0, bot_out, -ho)), ("bowl_outer_bottom", fan(bot_out, -ho, up=False)),
             ("flange_top", ring(fl_out, top_in, ft)), ("flange_edge", loft(fl_out, 0.0, fl_out, ft)), ("flange_lip", loft(top_in, ft, top_in, 0.0))]
    for nm, geo in parts:
        a.mesh(bd, geo, steel, collide=False, smooth=nm.endswith("wall"), name=nm).CreateDoubleSidedAttr(True)
    a.mesh(bd, cylinder((0, 0, -hi), 0.058, 0.058, 0.003, n=32), dark, collide=False, name="drain_ring")
    a.mesh(bd, cylinder((0, 0, -hi + 0.001), 0.045, 0.045, 0.004, n=32), steel, collide=False, name="drain_strainer")
    # colliders (guide): flange strips, bowl wall slabs (convex hulls), thick bottom (no tunnelling)
    for nm_, cx, cy, sx, sy in (("flange_b", 0, -(od + idp) / 4, ow, (od - idp) / 2), ("flange_f", 0, (od + idp) / 4, ow, (od - idp) / 2),
                                ("flange_l", -(ow + iw) / 4, 0, (ow - iw) / 2, idp), ("flange_r", (ow + iw) / 4, 0, (ow - iw) / 2, idp)):
        a.box(bd, (cx, cy, ft / 2), (sx, sy, ft), steel, name=f"col_{nm_}").CreatePurposeAttr(UsdGeom.Tokens.guide)
    n = len(top_in)
    for k in range(0, n, 2):
        k1 = (k + 2) % n
        P = [[*top_in[k], 0], [*top_in[k1], 0], [*bot_in[k], -hi], [*bot_in[k1], -hi],
             [*top_out[k], 0], [*top_out[k1], 0], [*bot_out[k], -ho], [*bot_out[k1], -ho]]
        a.mesh(bd, hull(P), steel, name=f"col_wall_{k:02d}").CreatePurposeAttr(UsdGeom.Tokens.guide)
    a.box(bd, (0, 0, -hi - 0.015), (bw + 0.02, bdp + 0.02, 0.03), steel, name="col_bottom").CreatePurposeAttr(UsdGeom.Tokens.guide)
    # faucet variants (materials exist outside the variants)
    fy = c["faucet_y"]
    root = a.st.GetPrimAtPath(a.R)
    vs = root.GetVariantSets().AddVariantSet("faucet")
    for v in ("pullout", "straight"):
        vs.AddVariant(v)
        vs.SetVariantSelection(v)
        with vs.GetVariantEditContext():
            fp = f"{bd}/Faucet"
            UsdGeom.Xform.Define(a.st, fp)
            a.mesh(fp, cylinder((0, fy, 0), 0.028, 0.025, 0.04), chrome, name="base")
            if v == "pullout":                                         # sink1.jpg: tall column, swan spout forward and down
                a.mesh(fp, cylinder((0, fy, 0.04), 0.019, 0.019, 0.24), chrome, name="column")
                arc = [(0, fy + 0.11 * (1 - math.cos(t)), 0.28 + 0.11 * math.sin(t)) for t in np.linspace(0, math.pi * 0.95, 10)]
                for k in range(len(arc) - 1):
                    a.mesh(fp, rod(arc[k], arc[k + 1], 0.016, n=12), chrome, name=f"spout_{k}")
                a.mesh(fp, cylinder((0, arc[-1][1], arc[-1][2] - 0.07), 0.021, 0.019, 0.07), chrome, name="spray_head")
                lz = 0.2
            else:                                                      # sink2.jpg: straight column, short head pointing forward
                a.mesh(fp, cylinder((0, fy, 0.04), 0.024, 0.024, 0.2), chrome, name="column")
                a.mesh(fp, superquadric((0, fy + 0.03, 0.25), (0.05, 0.11, 0.05), e=0.6), chrome, smooth=True, name="head")
                lz = 0.12
            a.mesh(fp, rod((-0.02, fy, lz), (-0.1, fy + 0.02, lz + 0.03), 0.008), chrome, collide=False, name="lever")   # viewer's right
    vs.SetVariantSelection("straight")
    a.save()


def build_induction():
    c = A["Induction"]
    a = Asset("Induction", "cooktop")
    glass, black, ring = a.material("induction_glass", 0.12), a.material("black", 0.5), a.material("dark_grey", 0.4)
    bd = a.body("Body", c["mass"], "cooktop")
    gx, gy, gt = c["glass"]
    bx, by, bh = c["body"]
    a.box(bd, (0, 0, gt / 2), (gx, gy, gt), glass, name="glass")
    a.box(bd, (0, 0, -bh / 2), (bx, by, bh), black, name="body")
    for k, (zx, zy, zr) in enumerate(c["zones"]):                     # cooking-zone rings printed on the glass
        ang = np.linspace(0, 2 * math.pi, 41)
        for i in range(40):
            p0 = (zx + zr * math.cos(ang[i]), zy + zr * math.sin(ang[i]), gt + 0.0006)
            p1 = (zx + zr * math.cos(ang[i + 1]), zy + zr * math.sin(ang[i + 1]), gt + 0.0006)
            a.mesh(bd, rod(p0, p1, 0.0012, n=4), ring, collide=False, name=f"zone{k}_{i:02d}")
    a.box(bd, (0, gy / 2 - 0.04, gt + 0.0006), (0.3, 0.025, 0.0012), ring, collide=False, name="touch_slider")
    a.save()


def build_microwave():
    """Seen from the front (+y) the viewer's left is +x: door on the left (+x), control panel and button on the right (-x)."""
    c = A["Microwave"]
    W, D, H = c["size"]
    a = Asset("Microwave", "microwave")
    UsdPhysics.ArticulationRootAPI.Apply(a.st.GetPrimAtPath(a.R))
    white, inner, dark, glass = a.material("white", 0.35), a.material("interior_white", 0.5), a.material("dark_grey", 0.3), a.material("black", 0.1)
    bd = a.body("Body", c["mass_body"], "microwave")
    t = 0.015
    dw = c["door_w"]
    cw = W - dw                                                       # control column width
    xc = -W / 2 + cw / 2                                              # control column centre (viewer's right)
    xd = W / 2 - dw / 2                                               # door / cavity centre (viewer's left)
    a.box(bd, (0, 0, t / 2), (W, D, t), white, name="bottom")
    a.box(bd, (0, 0, H - t / 2), (W, D, t), white, name="top")
    a.box(bd, (W / 2 - t / 2, 0, H / 2), (t, D, H), white, name="side_L")
    a.box(bd, (xc, 0, H / 2), (cw, D, H - 0.002), white, name="control_column")
    a.box(bd, (0, -D / 2 + t / 2, H / 2), (W, t, H), white, name="back")
    a.box(bd, (xd, 0, t + 0.001), (dw - t - 0.02, D - 2 * t, 0.002), inner, collide=False, name="cavity_floor")
    a.mesh(bd, cylinder((xd, -0.01, t + 0.004), 0.12, 0.12, 0.006, n=32), a.material("glass", 0.05, 0.0, 0.6), collide=False, name="turntable")
    a.box(bd, (xc, D / 2 + 0.001, H * 0.72), (cw * 0.7, 0.002, 0.03), a.material("black", 0.15), collide=False, name="display")
    for k in range(10):                                               # keypad
        a.box(bd, (xc + (k % 2 - 0.5) * 0.04, D / 2 + 0.001, H * 0.6 - (k // 2) * 0.028), (0.03, 0.002, 0.016), dark, collide=False, name=f"key_{k}")
    # door: hinged at the left front edge (+x), opens outward (toward +y)
    lk = a.body("Door", c["door_mass"], "appliance_door", translate=(W / 2, D / 2, 0))
    a.box(lk, (-dw / 2, c["door_t"] / 2, H / 2), (dw, c["door_t"], H - 0.004), white, name="panel")
    a.box(lk, (-dw / 2 + 0.02, c["door_t"] + 0.001, H / 2), (dw * 0.68, 0.002, H * 0.6), glass, collide=False, name="window")
    a.joint("revolute", "door", bd, lk, (W / 2, D / 2, 0), (0, 0, 0), axis="Z", limits=(-c["open_deg"], 0), damping=c["door_damping"], drive="angular")
    # push button (lower right of the control panel): spring-returned prismatic joint, pressing moves it into the body (-y)
    b = c["button"]
    bt = a.body("Button", 0.02, "button")
    a.box(bt, (xc, D / 2 + b["size"][1] / 2, 0.045), b["size"], dark, name="cap")
    j = a.joint("prismatic", "button", bd, bt, (0, 0, 0), (0, 0, 0), axis="Y", limits=(-b["travel"], 0), damping=b["damping"], drive="linear")
    UsdPhysics.DriveAPI(j.GetPrim(), "linear").GetStiffnessAttr().Set(b["stiffness"])
    root = a.st.GetPrimAtPath(a.R)
    root.SetCustomDataByKey("latch", {"button_joint": "Joints/button", "door_joint": "Joints/door", "press_to_open_m": b["press_to_open_m"],
                                      "door_open_sign": -1,
                                      "note": "door is released (opened by the controller) when the button joint position < -press_to_open_m"})
    a.save()


def build_fridge():
    c = A["Fridge"]
    W, D, H = c["size"]
    a = Asset("Fridge", "refrigerator")
    UsdPhysics.ArticulationRootAPI.Apply(a.st.GetPrimAtPath(a.R))
    inox, inner, dark, glass = a.material("fridge_inox", 0.35, 0.6), a.material("interior_white", 0.4), a.material("dark_grey", 0.5), a.material("glass", 0.05, 0.0, 0.35)
    bd = a.body("Body", c["mass_body"], "refrigerator")
    t, dt, zs = 0.04, c["door_t"], c["freezer_split_z"]
    Db = D - dt                                                       # carcass depth (doors in front)
    yc = -dt / 2
    a.box(bd, (0, yc, 0.03), (W, Db, 0.06), inox, name="plinth")
    a.box(bd, (0, yc, H - t / 2), (W, Db, t), inox, name="top")
    for sx in (-1, 1):
        a.box(bd, (sx * (W / 2 - t / 2), yc, H / 2), (t, Db, H), inox, name=f"side_{'L' if sx < 0 else 'R'}")
    a.box(bd, (0, yc - Db / 2 + t / 2, H / 2), (W - 2 * t, t, H), inner, name="back")
    a.box(bd, (0, yc, 0.06 + t / 2), (W - 2 * t, Db, t), inner, name="bottom")
    a.box(bd, (0, yc, zs - 0.03), (W - 2 * t, Db, 0.06), inner, name="divider")
    for k, z in enumerate((0.45, 0.75, 1.0)):
        a.box(bd, (0, yc - 0.02, z), (W - 2 * t - 0.01, Db - 0.06, 0.006), glass, name=f"shelf_{k}")
    a.box(bd, (0, yc - 0.02, zs + (H - zs) / 2), (W - 2 * t - 0.01, Db - 0.06, 0.006), glass, name="freezer_shelf")
    doors = (("FreezerDoor", "freezer_door", zs, H, c["door_mass"][0]), ("FridgeDoor", "fridge_door", 0.06, zs - 0.005, c["door_mass"][1]))
    for name, jn, z0, z1, m in doors:                                 # hinges on the viewer's right (-x), opening toward +y
        lk = a.body(name, m, "appliance_door", translate=(-W / 2, D / 2 - dt, 0))
        a.box(lk, (W / 2, dt / 2, (z0 + z1) / 2), (W, dt, z1 - z0 - 0.004), inox, name="panel")
        a.box(lk, (W / 2, -0.005, (z0 + z1) / 2), (W - 0.08, 0.01, z1 - z0 - 0.06), inner, collide=False, name="liner")
        a.box(lk, (W - 0.012, dt + 0.001, z1 - 0.12 if name == "FridgeDoor" else z0 + 0.12), (0.012, 0.002, 0.18), dark, collide=False, name="handle_recess")
        a.joint("revolute", jn, bd, lk, (-W / 2, D / 2 - dt, 0), (0, 0, 0), axis="Z", limits=(0, c["open_deg"]), damping=c["door_damping"], drive="angular")
    a.box(bd, (0.15, D / 2 - dt + 0.001, zs + 0.2), (0.08, 0.002, 0.1), a.material("white", 0.5), collide=False, name="energy_label")
    a.save()


# ------------------------------------------------------------------ central hall
def build_desk6():
    c = A["Desk6"]
    W, D, H = c["size"]
    a = Asset("Desk6", "table")
    top, leg = a.material("desk_top", 0.45), a.material("desk_leg", 0.35, 0.6)
    bd = a.body("Body", c["mass"], "table")
    tt, lg, ah = c["top_t"], c["leg"], c["apron_h"]
    a.box(bd, (0, 0, H - tt / 2), (W, D, tt), top, name="top")
    for sx in (-1, 1):
        for sy in (-1, 1):
            a.box(bd, (sx * (W / 2 - lg / 2 - 0.02), sy * (D / 2 - lg / 2 - 0.02), (H - tt) / 2), (lg, lg, H - tt), leg, name=f"leg_{sx}_{sy}".replace("-", "m"))
    for sy in (-1, 1):
        a.box(bd, (0, sy * (D / 2 - 0.03), H - tt - ah / 2), (W - 0.1, 0.02, ah), leg, name=f"apron_{'S' if sy < 0 else 'N'}")
    for sx in (-1, 1):
        a.box(bd, (sx * (W / 2 - 0.045), 0, H - tt - ah / 2), (0.02, D - 0.1, ah), leg, name=f"apron_{'L' if sx < 0 else 'R'}")
    a.save()


def build_chair6():
    c = A["Chair6"]
    W, D, H = c["size"]
    a = Asset("Chair6", "chair")
    shell, pad, chrome = a.material("shell_white", 0.35), a.material("seat_dark", 0.7), a.material("chrome", 0.25, 1.0)
    bd = a.body("Body", c["mass"], "chair")
    sh = c["seat_h"]
    a.mesh(bd, superquadric((0, 0.02, sh - 0.02), (W - 0.04, D - 0.06, 0.04), e=0.3), shell, smooth=True, name="seat_shell")
    a.mesh(bd, superquadric((0, 0.03, sh + 0.008), (W - 0.1, D - 0.12, 0.025), e=0.3), pad, smooth=True, name="seat_pad")
    V, F = superquadric((0, 0, 0), (W - 0.02, 0.05, H - sh - 0.04), e=0.3)    # back shell, tilted 10 deg
    t = math.radians(10)
    V = V @ np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]).T + [0, -D / 2 + 0.06, sh + 0.02 + (H - sh - 0.04) / 2]
    a.mesh(bd, (V, F), shell, smooth=True, name="back_shell")
    for sx in (-1, 1):
        for sy in (-1, 1):
            a.mesh(bd, rod((sx * (W / 2 - 0.06), sy * (D / 2 - 0.07), sh - 0.04), (sx * (W / 2 - 0.03), sy * (D / 2 - 0.03), 0.01), 0.011), chrome,
                   name=f"leg_{sx}_{sy}".replace("-", "m"))
    a.save()


def build_tvstand():
    c = A["TVStand"]
    a = Asset("TVStand", "tv")
    black, steel = a.material("tv_black", 0.2), a.material("black", 0.5)
    bd = a.body("Body", c["mass"], "tv")
    tw, tt, th = c["tv"]
    a.box(bd, (0, 0.05, c["tv_center_z"]), (tw, tt, th), black, name="tv")
    a.box(bd, (0, 0.05 + tt / 2 + 0.001, c["tv_center_z"]), (tw - 0.02, 0.002, th - 0.02), a.material("black", 0.05), collide=False, name="screen")
    px_, ph = c["post_x"], c["post_h"]
    for sx in (-1, 1):
        a.box(bd, (sx * px_, 0, ph / 2), (0.06, 0.04, ph), steel, name=f"post_{sx}".replace("-", "m"))
        a.box(bd, (sx * px_, 0, 0.06), (0.06, c["foot_len"], 0.04), steel, name=f"foot_{sx}".replace("-", "m"))
        for sy in (-1, 1):
            a.mesh(bd, cylinder((sx * px_, sy * (c["foot_len"] / 2 - 0.03), 0), 0.035, 0.035, 0.04), steel, name=f"caster_{sx}_{sy}".replace("-", "m"))
    a.box(bd, (0, 0, 0.25), (2 * px_ + 0.06, 0.03, 0.05), steel, name="crossbar")
    a.box(bd, (0, -0.1, c["shelf_z"]), (2 * px_ - 0.04, 0.3, 0.02), steel, name="shelf")
    a.save()


def build_office_desk():
    c = A["OfficeDesk"]
    W, D, H = c["size"]
    a = Asset("OfficeDesk", "desk")
    white = a.material("office_white", 0.4)
    frame = a.material("office_white", 0.3, 0.3)
    bd = a.body("Body", c["mass"], "desk")
    tt, lw = c["top_t"], c["leg_w"]
    a.box(bd, (0, 0, H - tt / 2), (W, D, tt), white, name="top")
    for sx in (-1, 1):                                                # A-frame legs at both ends
        x = sx * (W / 2 - 0.08)
        for sy in (-1, 1):
            a.mesh(bd, rod((x, sy * 0.06, H - tt), (x, sy * (D / 2 - 0.05), 0.02), lw / 2), frame, name=f"leg_{sx}_{sy}".replace("-", "m"))
        a.mesh(bd, rod((x, -(D / 2 - 0.05), 0.03), (x, D / 2 - 0.05, 0.03), lw / 2), frame, name=f"foot_{sx}".replace("-", "m"))
    a.box(bd, (0, -D / 2 + 0.06, H - tt - 0.04), (W - 0.2, 0.02, 0.08), frame, name="back_rail")
    a.save()


def build_office_chair():
    c = A["OfficeChair"]
    a = Asset("OfficeChair", "chair")
    mesh, black, chrome = a.material("mesh_black", 0.8), a.material("black", 0.5), a.material("chrome", 0.3, 0.9)
    bd = a.body("Body", c["mass"], "chair")
    (sw, sd), sh, (bw, bh), r = c["seat"], c["seat_h"], c["back"], c["base_r"]
    for k in range(5):                                                # 5-star base with casters
        ang = 2 * math.pi * k / 5 + math.pi / 2
        p1 = (r * math.cos(ang), r * math.sin(ang), 0.07)
        a.mesh(bd, rod((0, 0, 0.1), p1, 0.018), black, name=f"star_{k}")
        a.mesh(bd, superquadric((p1[0], p1[1], 0.03), (0.06, 0.05, 0.06), e=1.0, nu=12, nv=8), black, smooth=True, name=f"caster_{k}")
    a.mesh(bd, cylinder((0, 0, 0.1), 0.025, 0.025, sh - 0.16), chrome, name="gas_lift")
    a.mesh(bd, superquadric((0, 0.02, sh - 0.03), (sw, sd, 0.07), e=0.3), black, smooth=True, name="seat")
    V, F = superquadric((0, 0, 0), (bw, 0.05, bh), e=0.35)           # mesh back, slightly reclined
    t = math.radians(12)
    V = V @ np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]).T + [0, -sd / 2 - 0.02, sh + 0.08 + bh / 2]
    a.mesh(bd, (V, F), mesh, smooth=True, name="back")
    a.mesh(bd, superquadric((0, -sd / 2 - 0.1, sh + bh + 0.17), (0.28, 0.06, 0.12), e=0.4), black, smooth=True, name="headrest")
    for sx in (-1, 1):
        a.box(bd, (sx * (sw / 2 + 0.02), -0.02, sh + 0.21), (0.05, 0.26, 0.03), black, name=f"armrest_{sx}".replace("-", "m"))
        a.box(bd, (sx * (sw / 2 + 0.02), -0.05, sh + 0.1), (0.03, 0.03, 0.2), black, collide=False, name=f"armpost_{sx}".replace("-", "m"))
    a.save()


# ------------------------------------------------------------------ coffee robot (Dr.Presso) + kiosk
def textured_material(a, name, png, rough=0.6):
    m = UsdShade.Material.Define(a.st, f"{a.R}/Looks/{name}")
    sh = UsdShade.Shader.Define(a.st, f"{m.GetPath()}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    rd = UsdShade.Shader.Define(a.st, f"{m.GetPath()}/st")
    rd.CreateIdAttr("UsdPrimvarReader_float2")
    rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tx = UsdShade.Shader.Define(a.st, f"{m.GetPath()}/Texture")
    tx.CreateIdAttr("UsdUVTexture")
    tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./{png}")
    tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(rd.ConnectableAPI(), "result")
    tx.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def textured_quad(a, parent, name, p0, p1, p2, p3, mat):
    """Quad p0 (u0,v0) p1 (u1,v0) p2 (u1,v1) p3 (u0,v1)."""
    me = a.mesh(parent, (np.array([p0, p1, p2, p3]), [[0, 1, 2, 3]]), mat, collide=False, name=name)
    UsdGeom.PrimvarsAPI(me).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray([Gf.Vec2f(0, 0), Gf.Vec2f(1, 0), Gf.Vec2f(1, 1), Gf.Vec2f(0, 1)]))
    return me


def sign_band_png(path, w_m, h_m):
    """Cream sign band of the booth: warning line + QR HERE badge (photo coffee_robot.jpg)."""
    from PIL import Image, ImageDraw, ImageFont
    ppm = 800
    W, H = int(w_m * ppm), int(h_m * ppm)
    im = Image.new("RGB", (W, H), tuple(COL["cream"]))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", int(0.04 * ppm), index=1)
    txt = "문이 자동으로 열린 후 픽 업대가 앞으로 나옵니다."
    tw = d.textlength(txt, font=f)
    x0, y0 = W * 0.58 - tw / 2, H * 0.32
    d.text((x0, y0), txt, font=f, fill=(40, 40, 40))
    for xc in (x0 - 0.04 * ppm, x0 + tw + 0.04 * ppm):                   # warning triangles
        s_ = 0.035 * ppm
        d.polygon([(xc, y0 + 0.005 * ppm), (xc - s_ / 2, y0 + s_), (xc + s_ / 2, y0 + s_)], fill=(232, 150, 30))
    bx, by = int(0.1 * ppm), int(0.1 * ppm)                             # QR HERE badge + reader
    d.rectangle((bx + 0.03 * ppm, by - 0.06 * ppm, bx + 0.08 * ppm, by - 0.01 * ppm), fill=(30, 30, 30))
    d.rounded_rectangle((bx, by, bx + 0.11 * ppm, by + 0.05 * ppm), radius=8, fill=(200, 40, 40))
    d.text((bx + 0.012 * ppm, by + 0.004 * ppm), "QR HERE", font=ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
           int(0.028 * ppm), index=1), fill=(255, 220, 120))
    im.save(path)


def build_coffee_robot():
    """Dr.Presso booth (static colliders, no rigid body) with a posed Doosan A0509 (physics disabled), espresso machine, pick-up trays.
    Local origin = floor centre, front (+y) = customer side; viewer's left = +x (arm side), viewer's right = -x (coffee machine)."""
    import a0509_fk
    c = A["CoffeeRobot"]
    W, D, H = c["size"]
    a = Asset("CoffeeRobot", "coffee_robot")
    white, oak, oak_l = a.material("booth_white", 0.35), a.material("oak", 0.55), a.material("oak_light", 0.45)
    glass, steel, black, chrome = a.material("glass", 0.02, 0.0, 0.1), a.material("stainless", 0.3, 0.9), a.material("black", 0.3), a.material("chrome", 0.15, 1.0)
    bd = f"{a.R}/Body"
    UsdGeom.Xform.Define(a.st, bd)                                    # static: colliders without a rigid body
    a.label(a.st.GetPrimAtPath(bd), "coffee_robot")
    ch, st_, r, wz = c["caster_h"], c["shell_t"], c["corner_r"], c["worktop_z"]
    # shell: rounded-rectangle tube along y (front profile in x-z)
    def prof(w, h, z0, rr, n=10):
        pts = []
        for (cx, cz, a0) in ((w / 2 - rr, z0 + rr, -90), (w / 2 - rr, z0 + h - rr, 0), (-w / 2 + rr, z0 + h - rr, 90), (-w / 2 + rr, z0 + rr, 180)):
            for k in range(n + 1):
                t = math.radians(a0 + 90 * k / n)
                pts.append((cx + rr * math.cos(t), cz + rr * math.sin(t)))
        return np.array(pts)
    po, pi = prof(W, H - ch, ch, r), prof(W - 2 * st_, H - ch - 2 * st_, ch + st_, r - st_)
    n = len(po)
    y0, y1 = -D / 2, D / 2
    V = np.vstack([[[x, y0, z] for x, z in po], [[x, y1, z] for x, z in po], [[x, y0, z] for x, z in pi], [[x, y1, z] for x, z in pi]])
    F = [[k, (k + 1) % n, n + (k + 1) % n, n + k] for k in range(n)]                      # outer
    F += [[2 * n + k, 3 * n + k, 3 * n + (k + 1) % n, 2 * n + (k + 1) % n] for k in range(n)]   # inner
    F += [[n + k, n + (k + 1) % n, 3 * n + (k + 1) % n, 3 * n + k] for k in range(n)]      # front rim
    F += [[k, 2 * n + k, 2 * n + (k + 1) % n, (k + 1) % n] for k in range(n)]              # back rim
    a.mesh(bd, (V, F), white, collide=False, smooth=True, name="shell").CreateDoubleSidedAttr(True)
    ib = pi.copy()
    Vb = np.vstack([[[x, y0 + 0.02, z] for x, z in ib], [[ib[:, 0].mean(), y0 + 0.02, ib[:, 1].mean()]]])
    a.mesh(bd, (Vb, [[n, (k + 1) % n, k] for k in range(n)]), oak_l, collide=False, name="back_wall").CreateDoubleSidedAttr(True)
    xi = W / 2 - st_
    zi0, zi1 = ch + st_, H - st_
    a.box(bd, (0, (y0 + 0.02 + y1 - 0.08) / 2, (zi0 + wz - 0.03) / 2), (2 * xi, D - 0.1, wz - 0.03 - zi0), oak, collide=False, name="cabinet")   # front panels sit in front of it
    a.box(bd, (0, -0.02, wz - 0.015), (2 * xi, D - 0.06, 0.03), oak_l, collide=False, name="worktop")
    bz0, bz1 = c["band_z"]
    a.box(bd, (0, y1 - 0.06, (zi0 + bz0) / 2), (2 * xi, 0.02, bz0 - zi0), oak, collide=False, name="front_wood")
    png = OUT / "CoffeeRobot" / "sign_band.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    sign_band_png(png, 2 * xi, bz1 - bz0)
    band = textured_material(a, "SignBand", "sign_band.png")
    a.box(bd, (0, y1 - 0.06, (bz0 + bz1) / 2), (2 * xi, 0.02, bz1 - bz0), a.material("cream", 0.5), collide=False, name="band")
    textured_quad(a, bd, "band_face", (xi, y1 - 0.049, bz0), (-xi, y1 - 0.049, bz0), (-xi, y1 - 0.049, bz1), (xi, y1 - 0.049, bz1), band)
    a.box(bd, (0, y1 - 0.04, (wz + zi1) / 2), (2 * xi, 0.008, zi1 - wz), glass, collide=False, name="front_glass")
    for sx in (-1, 1):
        a.box(bd, (sx * (xi - 0.004), -0.1, 1.3), (0.006, 0.5, 0.4), a.material("black", 0.1), collide=False, name=f"side_window_{sx}".replace("-", "m"))
    for sx in (-1, 1):                                                 # casters
        for sy in (-1, 1):
            a.mesh(bd, cylinder((sx * (W / 2 - 0.15), sy * (D / 2 - 0.15), 0.0), 0.03, 0.03, ch, n=16), black, collide=False,
                   name=f"caster_{sx}_{sy}".replace("-", "m"))
    # interior: monitor, espresso machine, gooseneck dispensers, cup tubes, pick-up station, LED light
    tv = c["tv"]
    a.box(bd, (tv["x"], y0 + 0.04, tv["z"]), (tv["size"][0] + 0.02, 0.03, tv["size"][1] + 0.02), black, collide=False, name="monitor")
    a.box(bd, (tv["x"], y0 + 0.056, tv["z"]), (tv["size"][0], 0.002, tv["size"][1]), a.material("screen", 0.2), collide=False, name="monitor_screen")
    cm = c["coffee_machine"]
    (mw, md, mh), mx, my = cm["size"], cm["x"], cm["y"]
    a.box(bd, (mx, my, wz + mh / 2), (mw, md, mh), steel, collide=False, name="espresso_body")
    a.box(bd, (mx, my + md / 2 + 0.002, wz + mh * 0.62), (mw - 0.02, 0.004, mh * 0.62), black, collide=False, name="espresso_front")
    a.box(bd, (mx, my + md / 2 + 0.005, wz + mh * 0.88), (0.16, 0.004, 0.06), a.material("screen", 0.2), collide=False, name="espresso_display")
    a.box(bd, (mx, my + md / 2 - 0.05, wz + 0.03), (mw - 0.06, 0.12, 0.02), steel, collide=False, name="drip_tray")
    a.mesh(bd, cylinder((mx, my + md / 2 - 0.06, wz + 0.2), 0.03, 0.03, 0.05), black, collide=False, name="group_head")
    for k, (gx, gy) in enumerate(c["goosenecks"]):                    # inverted-J gooseneck dispensers
        pts = [(gx, gy, wz), (gx, gy, wz + 0.26)] + [(gx, gy + 0.06 * (1 - math.cos(t)), wz + 0.26 + 0.06 * math.sin(t)) for t in np.linspace(0, math.pi, 8)[1:]]
        pts += [(gx, gy + 0.12, wz + 0.18)]
        for i in range(len(pts) - 1):
            a.mesh(bd, rod(pts[i], pts[i + 1], 0.011, n=10), chrome, collide=False, name=f"gooseneck{k}_{i}")
        a.mesh(bd, cylinder((gx, gy + 0.12, wz), 0.05, 0.05, 0.006, n=20), chrome, collide=False, name=f"gooseneck{k}_plate")
    for k, (tx_, ty_) in enumerate(c["cup_tubes"]):
        a.mesh(bd, cylinder((tx_, ty_, wz), 0.04, 0.04, 0.12 + 0.04 * (k % 2), n=20), steel, collide=False, name=f"cup_tube_{k}")
    pk = c["pickup"]
    (px0, px1), py, pn = pk["x"], pk["y"], pk["n"]
    pw = (px0 - px1) / pn
    a.box(bd, ((px0 + px1) / 2, py - 0.12, wz + 0.25), (px0 - px1, 0.02, 0.03), white, collide=False, name="pickup_bar")
    for k in range(pn + 1):
        a.box(bd, (px0 - k * pw, py - 0.12, wz + 0.125), (0.012, 0.06, 0.25), white, collide=False, name=f"pickup_post_{k}")
    for k in range(pn):
        xk = px0 - (k + 0.5) * pw
        a.box(bd, (xk, py, wz + 0.015), (pw * 0.75, 0.16, 0.03), white, collide=False, name=f"pickup_tray_{k}")
        a.mesh(bd, cylinder((xk, py, wz + 0.03), 0.045, 0.045, 0.006, n=20), a.material("dark_grey", 0.4), collide=False, name=f"pickup_ring_{k}")
    lt = UsdLux.RectLight.Define(a.st, f"{a.R}/InteriorLight")
    lt.CreateWidthAttr(2 * xi - 0.3)
    lt.CreateHeightAttr(D - 0.5)
    lt.CreateIntensityAttr(c["interior_light"])
    lt.CreateColorTemperatureAttr(4000.0)
    lt.CreateEnableColorTemperatureAttr(True)
    lt.AddTranslateOp().Set(Gf.Vec3d(0, 0, zi1 - 0.03))
    for k, yy in enumerate((0.25, -0.25)):
        a.box(bd, (0, yy, zi1 - 0.006), (2 * xi - 0.3, 0.03, 0.01), a.material("warm_led", 0.3), collide=False, name=f"led_strip_{k}")
    # static colliders (guide boxes)
    for nm_, cc_, ss_ in (("col_base", (0, 0, (ch + wz) / 2), (W, D, wz - ch)), ("col_back", (0, y0 + st_ / 2, (wz + H) / 2), (W, st_, H - wz)),
                          ("col_left", (W / 2 - st_ / 2, 0, (wz + H) / 2), (st_, D, H - wz)), ("col_right", (-W / 2 + st_ / 2, 0, (wz + H) / 2), (st_, D, H - wz)),
                          ("col_top", (0, 0, H - st_ / 2), (W, D, st_)), ("col_glass", (0, y1 - 0.04, (wz + H) / 2), (W, 0.02, H - wz))):
        a.box(bd, cc_, ss_, white, name=nm_).CreatePurposeAttr(UsdGeom.Tokens.guide)
    # Doosan A0509: referenced from the metres-corrected copy, posed by FK, physics disabled
    arm = c["arm"]
    usd = OUT / "CoffeeRobot" / "a0509_meters.usd"
    ap = f"{a.R}/Arm"
    ax_ = UsdGeom.Xform.Define(a.st, ap)
    ax_.GetPrim().GetReferences().AddReference(str(usd), "/Root/a0509")     # absolute while authoring (in-memory stage)
    ax_.ClearXformOpOrder()                                          # the referenced root has its own (identity) ops
    ax_.AddTranslateOp(opSuffix="place").Set(Gf.Vec3d(arm["base"][0], arm["base"][1], wz))
    ax_.AddRotateZOp(opSuffix="place").Set(arm["yaw"])
    a.label(ax_.GetPrim(), "robot_arm")
    L = a0509_fk.posed_links(usd, arm["q_deg"])
    for link, M in L.items():
        lp = a.st.GetPrimAtPath(f"{ap}/{link}")
        xf = UsdGeom.Xformable(lp)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(M)
    for p_ in Usd.PrimRange(a.st.GetPrimAtPath(ap)):
        if p_.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(p_).CreateRigidBodyEnabledAttr(False)
    for p_ in list(Usd.PrimRange(a.st.GetPrimAtPath(ap))):
        if p_.IsA(UsdPhysics.Joint):
            p_.SetActive(False)
    gp = f"{ap}/Gripper"                                               # 2-finger gripper on the tool flange (photo)
    g = UsdGeom.Xform.Define(a.st, gp)
    g.AddTransformOp().Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(0, 0, 0.0)) * L["link_6"])
    gw = a.material("booth_white", 0.4)
    a.mesh(gp, cylinder((0, 0, 0), 0.035, 0.035, 0.03, n=20), gw, collide=False, name="flange")
    a.box(gp, (0, 0, 0.07), (0.07, 0.05, 0.08), gw, collide=False, name="body")
    for sx in (-1, 1):
        a.mesh(gp, rod((sx * 0.02, 0, 0.11), (sx * 0.04, 0, 0.17), 0.007, n=8), a.material("dark_grey", 0.4), collide=False, name=f"finger_{sx}".replace("-", "m"))
        a.mesh(gp, rod((sx * 0.04, 0, 0.17), (sx * 0.025, 0, 0.2), 0.007, n=8), a.material("dark_grey", 0.4), collide=False, name=f"finger_tip_{sx}".replace("-", "m"))
    a.st.GetPrimAtPath(a.R).SetCustomDataByKey("arm_pose_deg", Vt.DoubleArray([float(v) for v in arm["q_deg"]]))
    refs = ax_.GetPrim().GetReferences()                              # saved asset: reference relative to its own folder
    refs.ClearReferences()
    refs.AddReference("./a0509_meters.usd", "/Root/a0509")
    a.save()


def build_kiosk():
    c = A["CoffeeKiosk"]
    a = Asset("CoffeeKiosk", "kiosk")
    white, gold, black = a.material("booth_white", 0.35), a.material("gold", 0.3, 0.6), a.material("black", 0.1)
    bd = a.body("Body", c["mass"], "kiosk")
    (bw, bdp, bh), (cw, cd, chh), (hw, hd, hh), (sw, sh) = c["base"], c["column"], c["head"], c["screen"]
    a.box(bd, (0, 0, bh / 2 + 0.02), (bw, bdp, bh), white, name="base")
    a.box(bd, (0, -0.05, 0.06 + chh / 2), (cw, cd, chh), white, name="column")
    a.box(bd, (0, -0.05, 0.06 + chh + hh / 2), (hw, hd, hh), white, name="head")
    for sx in (-1, 1):
        a.box(bd, (sx * (hw / 2 + 0.006), -0.05, 0.06 + chh + hh / 2), (0.012, hd, hh), gold, collide=False, name=f"trim_{sx}".replace("-", "m"))
    a.box(bd, (0, -0.05 + hd / 2 + 0.002, 0.06 + chh + hh / 2 + 0.06), (sw, 0.003, sh), black, collide=False, name="screen")
    a.box(bd, (0, -0.05 + hd / 2 + 0.003, 0.06 + chh + hh / 2 + 0.06), (sw - 0.04, 0.002, sh - 0.06), a.material("screen", 0.2), collide=False, name="screen_ui")
    a.box(bd, (0, -0.05 + cd / 2 + 0.002, 0.06 + chh - 0.12), (0.12, 0.004, 0.08), black, collide=False, name="card_reader")
    a.save()


# ------------------------------------------------------------------ care room additions (2026-10-15): desk + deformable T-shirt
def build_care_desk():
    c = A["CareDesk"]
    W, D, H = c["size"]
    a = Asset("CareDesk", "desk")
    top, white = a.material("desk_wall_tone", 0.5), a.material("white", 0.35)
    bd = a.body("Body", c["mass"], "desk")
    tt, lg = c["top_t"], c["leg"]
    a.box(bd, (0, 0, H - tt / 2), (W, D, tt), top, name="top")
    for sx in (-1, 1):
        for sy in (-1, 1):
            a.box(bd, (sx * (W / 2 - lg / 2 - 0.02), sy * (D / 2 - lg / 2 - 0.02), (H - tt) / 2), (lg, lg, H - tt), white,
                  name=f"leg_{sx}_{sy}".replace("-", "m"))
    for sy in (-1, 1):
        a.box(bd, (0, sy * (D / 2 - 0.035), H - tt - 0.03), (W - 0.08, 0.02, 0.05), white, name=f"rail_{'S' if sy < 0 else 'N'}")
    for sx in (-1, 1):
        a.box(bd, (sx * (W / 2 - 0.037), 0, H - tt - 0.03), (0.02, D - 0.08, 0.05), white, name=f"rail_{'L' if sx < 0 else 'R'}")
    a.save()


def tshirt_mesh(c):
    """Two-layer flat T-shirt (front z = gap, back z = 0) sharing the vertices of the sewn seams; open at hem, cuffs and neck.
    Local frame: sleeves along x, neck toward +y, origin at the middle of the garment. Openness is decided by position
    (hem line, neck span, cuff segments); triangles with all three vertices on a seam are dropped so every edge is manifold."""
    from scipy.spatial import Delaunay
    from shapely.geometry import LineString, Point, Polygon
    g = c["size"]
    W, L, sl, so, nw = g["body_w"] / 2, g["length"], g["sleeve"], g["sleeve_open"], g["neck_w"] / 2
    a_ = math.radians(25)                                              # sleeves angled down
    arm = (W, L - 0.24)
    cuff_b = (arm[0] + sl * math.cos(a_), arm[1] - sl * math.sin(a_))
    cuff_t = (cuff_b[0] + so * math.sin(a_), cuff_b[1] + so * math.cos(a_))
    right = [(0, 0), (W, 0), arm, cuff_b, cuff_t, (W - 0.02, L - 0.02), (nw, L), (0, L - 0.035)]
    pts = right + [(-x, y) for x, y in right[1:-1]][::-1]             # CCW
    poly = Polygon(pts)
    cuffs = [LineString([cuff_b, cuff_t]), LineString([(-cuff_b[0], cuff_b[1]), (-cuff_t[0], cuff_t[1])])]

    def is_open(q):                                                    # interior points of the openings (their end points are sewn)
        x, y = q
        if abs(y) < 1e-9 and abs(x) < W - 1e-6:
            return True                                                # hem
        if y > L - 0.04 and abs(x) < nw - 1e-6:
            return True                                                # neck
        return any(cl.distance(Point(q)) < 1e-6 and min(Point(q).distance(Point(e)) for e in cl.coords) > 1e-6 for cl in cuffs)
    res = c["res"]
    B = []
    for k in range(len(pts)):                                          # densified boundary
        seg = LineString([pts[k], pts[(k + 1) % len(pts)]])
        m = max(1, int(math.ceil(seg.length / res)))
        B += [(seg.interpolate(i / m, normalized=True).x, seg.interpolate(i / m, normalized=True).y) for i in range(m)]
    xs = np.arange(-cuff_t[0], cuff_t[0], res)
    I = [(x + (res / 2 if int(round(y / res)) % 2 else 0), y) for y in np.arange(0, L, res) for x in xs]
    I = [q for q in I if poly.contains(Point(q)) and poly.exterior.distance(Point(q)) > 0.6 * res]
    P2 = np.array(B + I)
    nb = len(B)
    seam = np.zeros(len(P2), bool)
    seam[:nb] = [not is_open(q) for q in B]
    keep = []
    for t in Delaunay(P2).simplices:
        e = max(np.linalg.norm(P2[t[i]] - P2[t[(i + 1) % 3]]) for i in range(3))
        if poly.contains(Point(P2[t].mean(0))) and e < 2.6 * res and not seam[t].all():
            keep.append(t)
    used = np.unique(np.array(keep).ravel())                           # drop vertices left unused
    remap = -np.ones(len(P2), int)
    remap[used] = np.arange(len(used))
    P2, seam, keep = P2[used], seam[used], [remap[t] for t in keep]
    gap = c["layer_gap"]
    V = [(x, y, 0.0) for x, y in P2]                                   # back layer
    front_idx = np.arange(len(P2))
    for i in range(len(P2)):
        if not seam[i]:
            front_idx[i] = len(V)
            V.append((P2[i][0], P2[i][1], gap))
    F = []
    for t in keep:
        a0, b0, c0 = (int(v) for v in t)
        d1, d2 = P2[b0] - P2[a0], P2[c0] - P2[a0]
        if d1[0] * d2[1] - d1[1] * d2[0] < 0:
            b0, c0 = c0, b0
        F.append([a0, c0, b0])                                         # back: facing down
        F.append([int(front_idx[a0]), int(front_idx[b0]), int(front_idx[c0])])   # front: facing up
    V = np.array(V) - [0, L / 2, 0]
    return V, F, int(seam.sum())


def build_tshirt():
    c = A["TShirt"]
    a = Asset("TShirt", "tshirt")
    V, F, n_seam = tshirt_mesh(c)
    me = a.mesh(a.R, (V, F), a.material("tshirt_black", 0.9), collide=False, name="Shirt", label="tshirt")
    me.CreateDoubleSidedAttr(True)
    UsdGeom.PrimvarsAPI(me).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray([Gf.Vec2f(float(v[0]), float(v[1])) for v in V]))
    sp = str(me.GetPath())
    assert deformableUtils.set_physics_surface_deformable_body(a.st, sp)
    deformableUtils.add_surface_deformable_material(a.st, f"{a.R}/Looks/TShirtPhysics", density=c["density"], youngs_modulus=c["youngs_modulus"],
                                                    poissons_ratio=c["poissons_ratio"], static_friction=c["friction"][0],
                                                    dynamic_friction=c["friction"][1], surface_thickness=c["thickness"],
                                                    surface_bend_stiffness=c["bend_stiffness"])
    mp = a.st.GetPrimAtPath(f"{a.R}/Looks/TShirtPhysics")
    mp.ApplyAPI("PhysxDeformableMaterialAPI")
    mp.GetAttribute("physxDeformableMaterial:elasticityDamping").Set(c["elasticity_damping"])
    mp.ApplyAPI("PhysxSurfaceDeformableMaterialAPI")
    mp.CreateAttribute("physxDeformableMaterial:bendDamping", Sdf.ValueTypeNames.Float).Set(c["bend_damping"])
    physicsUtils.add_physics_material_to_prim(a.st, me.GetPrim(), f"{a.R}/Looks/TShirtPhysics")
    pr = me.GetPrim()
    pr.ApplyAPI("PhysxBaseDeformableBodyAPI")
    pr.ApplyAPI("PhysxSurfaceDeformableBodyAPI")
    pr.GetAttribute("physxDeformableBody:selfCollision").Set(c["self_collision"])
    pr.ApplyAPI("PhysxCollisionAPI")
    pr.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(c["rest_offset"])
    pr.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(c["contact_offset"])
    a.st.GetPrimAtPath(a.R).SetCustomDataByKey("mesh", {"vertices": len(V), "triangles": len(F), "seam_vertices": n_seam})
    for _ in range(5):
        app.update()
    print(f"TShirt: {len(V)} vertices, {len(F)} triangles, {n_seam} shared seam vertices")
    a.save()


BUILDERS = {"Bed": build_bed, "Sofa3": build_sofa, "SofaTable": build_table, "Plant": build_plant, "Wardrobe800": build_wardrobe,
            "Vanity640": build_vanity, "Closet640": build_closet, "BarStool": build_stool, "Showcase": build_showcase, "TrashBin3": build_trash,
            "FireExtinguisher": build_extinguisher, "WaterPurifier": build_purifier, "Sink": build_sink, "Induction": build_induction,
            "Microwave": build_microwave, "Fridge": build_fridge, "Desk6": build_desk6, "Chair6": build_chair6, "TVStand": build_tvstand,
            "Armchair": lambda: build_sofa("Armchair"), "OfficeDesk": build_office_desk, "OfficeChair": build_office_chair,
            "CoffeeRobot": build_coffee_robot, "CoffeeKiosk": build_kiosk, "CareDesk": build_care_desk, "TShirt": build_tshirt}
for name in (os.environ.get("ASSETS") or ",".join(BUILDERS)).split(","):          # ASSETS=Sink,Fridge builds a subset
    BUILDERS[name]()
app.close()
