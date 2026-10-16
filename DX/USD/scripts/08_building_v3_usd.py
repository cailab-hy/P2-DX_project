"""Stage 8 (v3): build the empty-building USD from the measured plan, openings and user decisions.

Inputs : USD/config/dx_hub_no_obj_v1_plan.json, dx_hub_no_obj_v1_openings.json, dx_hub_no_obj_v1_overrides.json, dx_hub_no_obj_v1_zones.json
         (+ LAS for the ceiling height map, WORK/lights.json for the recessed panels found in v2)
Output : USD/DX_Hub_Building_no_obj_v1/DX_Hub_Building_no_obj_v1.usda (asset, defaultPrim /DX_Hub_Building)
         USD/DX_Hub_Building_no_obj_v1/payloads/building.usdc (geometry), DX_Hub_Building_no_obj_v1_scene.usda

Every element carries a semantic class (UsdSemantics.LabelsAPI, taxonomy "class") and wall/floor faces carry
the zone they face (GeomSubset per zone) so finishes can be assigned per zone in stage 9 (materials).
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np
import shapely
import shapely.affinity
from pxr import Gf, Kind, Sdf, Tf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSemantics, UsdShade, Vt
from scipy import ndimage
from shapely.geometry import LineString, Polygon, box as sbox
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

import v3_geom
from common import LAS, OUT, WORK, apply, read_las

ROOT = OUT.parent
CFG = ROOT / "config"
DST = ROOT / "DX_Hub_Building_no_obj_v1"
PAY = DST / "payloads"
PAY.mkdir(parents=True, exist_ok=True)
PLAN = json.load(open(CFG / "dx_hub_no_obj_v1_plan.json"))
OPN = json.load(open(CFG / "dx_hub_no_obj_v1_openings.json"))
OVR = json.load(open(CFG / "dx_hub_no_obj_v1_overrides.json"))
ZONES = json.load(open(CFG / "dx_hub_no_obj_v1_zones.json"))
T_L2W = np.array(PLAN["frame"]["T_las_to_world"])

# ------------------------------------------------------------------ ceiling height map (as in v2, 5 cm)
C, ZB, Z0 = 0.05, 0.05, -0.10
LO = np.array([-14.0, -23.0])
NX, NY, NZ = int(28 / C), int(46 / C), int((3.1 - Z0) / ZB)
xyz, _ = read_las(LAS)
P = apply(T_L2W, xyz).astype(np.float32)
del xyz
i_ = np.floor((P[:, 0] - LO[0]) / C).astype(np.int64)
j_ = np.floor((P[:, 1] - LO[1]) / C).astype(np.int64)
k_ = np.floor((P[:, 2] - Z0) / ZB).astype(np.int64)
ok = (i_ >= 0) & (i_ < NX) & (j_ >= 0) & (j_ < NY) & (k_ >= 0) & (k_ < NZ)
P, i_, j_, k_ = P[ok], i_[ok], j_[ok], k_[ok]
H = np.bincount((i_ * NY + j_) * NZ + k_, minlength=NX * NY * NZ).reshape(NX, NY, NZ)
zc = Z0 + ZB * (np.arange(NZ) + 0.5)
occ = H >= 3
hi = np.where(zc > 1.9)[0]
oc = occ[:, :, hi]
ceil_bin = np.where(oc.any(2), zc[hi[0] + oc.shape[2] - 1 - np.argmax(oc[:, :, ::-1], axis=2)], np.nan)
c_at = ceil_bin[i_, j_]
sel = np.isfinite(c_at) & (np.abs(P[:, 2] - c_at) < 0.06)
idx = i_[sel] * NY + j_[sel]
s_, n_ = np.bincount(idx, P[sel, 2], NX * NY), np.bincount(idx, minlength=NX * NY)
cz = np.where(n_ >= 3, s_ / np.maximum(n_, 1), np.nan).reshape(NX, NY)
BAND = P[(P[:, 2] > 1.6) & (P[:, 2] < 2.3)][::3, :2].astype(np.float64)   # for gap checks
del P, i_, j_, k_, H, c_at, sel, idx

rasters = np.load(WORK / "v3_rasters.npz")
RC = 0.025
fp_hr = rasters["fp_hr"]
fp = ndimage.zoom(fp_hr.astype(np.uint8), RC / C, order=0).astype(bool)[:NX, :NY]

dist_idx = ndimage.distance_transform_edt(~np.isfinite(cz), return_distances=False, return_indices=True)
czf = cz[tuple(dist_idx)]
med = ndimage.median_filter(czf, size=5)
jump = np.zeros(fp.shape, bool)
for ax in (0, 1):
    d = np.abs(np.diff(med, axis=ax)) > 0.03
    if ax == 0:
        jump[1:] |= d
        jump[:-1] |= d
    else:
        jump[:, 1:] |= d
        jump[:, :-1] |= d
rl, nr = ndimage.label(fp & ~jump)
areas = ndimage.sum(np.ones_like(rl), rl, range(1, nr + 1)) * C * C
big = np.zeros(nr + 1, bool)
big[1:] = areas >= 1.5
_, nearest = ndimage.distance_transform_edt(~big[rl], return_indices=True)
rlab = rl[tuple(nearest)] * fp
ids = np.unique(rlab[rlab > 0])
rh = {r: float(np.nanmedian(np.where(rlab == r, cz, np.nan))) for r in ids}
parent = {r: r for r in ids}


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for a_, b_ in [(rlab[1:], rlab[:-1]), (rlab[:, 1:], rlab[:, :-1])]:
    for a, b in np.unique(np.stack([a_.ravel(), b_.ravel()], 1), axis=0):
        if a and b and a != b and abs(rh[a] - rh[b]) < 0.025:
            parent[find(a)] = find(b)
root_ = np.arange(rlab.max() + 1)
for r in ids:
    root_[r] = find(r)
rlab = root_[rlab]
ids = np.unique(rlab[rlab > 0])
rh = {r: float(np.nanmedian(np.where(rlab == r, cz, np.nan))) for r in ids}
# regularisation (user 2026-10-09, photos ~/Downloads/re): the real soffit / coffer edges are clean flowing curves.
# Heights within level_gap_m form one ceiling level (1-3 cm differences are scan noise and made false steps + coves);
# the level partition is smoothed as a whole (Gaussian on one-hot levels -> argmax) so neighbouring regions share one
# boundary; strips thinner than 2 x open_radius_m (glass heads, wall tops) and islands < min_region_m2 are absorbed
# by the surrounding level.
CR = ZONES["ceiling_regularization"]
rh_old = dict(rh)
hs = sorted(ids, key=lambda r: rh[r])
area_r = {r: float((rlab == r).sum()) for r in ids}
levels, cur = [], [hs[0]]
for r in hs[1:]:
    (cur.append(r) if rh[r] - rh[cur[-1]] <= CR["level_gap_m"] else (levels.append(cur), cur := [r]))
levels.append(cur)
lev_h = [sum(rh[r] * area_r[r] for r in L) / sum(area_r[r] for r in L) for L in levels]
lmap = np.zeros(rlab.shape, np.int32)
for k, L in enumerate(levels, 1):
    lmap[np.isin(rlab, L)] = k
gx_, gy_ = np.meshgrid(LO[0] + (np.arange(rlab.shape[0]) + 0.5) * C, LO[1] + (np.arange(rlab.shape[1]) + 0.5) * C, indexing="ij")
from matplotlib.path import Path as MPath                         # noqa: E402


def paint_overrides(lm):
    """config ceiling_shape_overrides: polygons whose ceiling level is fixed (measured edges, noise inside)"""
    for ov in CR.get("shape_overrides", []):
        inside = MPath(ov["polygon"]).contains_points(np.c_[gx_.ravel(), gy_.ravel()]).reshape(lm.shape) & fp
        lm[inside] = 1 + int(np.argmin([abs(h_ - ov["level_m"]) for h_ in lev_h]))
    return lm


lmap = paint_overrides(lmap)
sig = CR["smooth_sigma_m"] / C
soft = np.stack([ndimage.gaussian_filter((lmap == k).astype(np.float32), sig) for k in range(1, len(levels) + 1)])
lmap = np.where(fp, soft.argmax(0) + 1, 0)
rad = int(round(CR["open_radius_m"] / C))
disk = np.hypot(*np.mgrid[-rad:rad + 1, -rad:rad + 1]) <= rad
for _ in range(2):                                                # thin strips, then small islands
    keep = np.zeros_like(fp)
    for k in range(1, len(levels) + 1):
        m = ndimage.binary_opening(lmap == k, disk)
        lab_, n_ = ndimage.label(m)
        if n_:
            a_ = ndimage.sum(np.ones_like(lab_), lab_, range(1, n_ + 1)) * C * C
            m = np.isin(lab_, np.where(a_ >= CR["min_region_m2"])[0] + 1)
        keep |= m
    _, near_ = ndimage.distance_transform_edt(~keep, return_indices=True)
    lmap = np.where(fp, lmap[tuple(near_)], 0)
lmap = paint_overrides(lmap)
rlab = np.zeros_like(rlab)
rh, nxt = {}, 1
for k in range(1, len(levels) + 1):
    lab_, n_ = ndimage.label(lmap == k)
    for q in range(1, n_ + 1):
        rlab[lab_ == q] = nxt
        rh[nxt] = float(np.nanmedian(np.where(lab_ == q, cz, np.nan)))   # measured height of this region
        nxt += 1
ids = np.unique(rlab[rlab > 0])
print("ceiling levels:", [f"{h:.3f} <- {sorted(round(rh_old[r], 3) for r in L)}" for h, L in zip(lev_h, levels)])
ceil_map = np.zeros(fp.shape)
for r in ids:
    ceil_map[rlab == r] = rh[r]
H_TOP = max(rh.values()) + 0.02
print(f"ceiling: {len(ids)} regions {sorted(round(v, 3) for v in rh.values())}, H_TOP {H_TOP:.3f}")


def ceil_at(x, y):
    i, j = int((x - LO[0]) / C), int((y - LO[1]) / C)
    if 0 <= i < NX and 0 <= j < NY and ceil_map[i, j] > 0:
        return float(ceil_map[i, j])
    return H_TOP


# ------------------------------------------------------------------ zones (finish areas)
ROOM_LAB = {"raster": None, "names": {}}                         # filled once walls/glass/doors are known


def zone_at(x, y):
    for z in ZONES["zones"]:
        if "box" not in z:
            continue
        x0, x1, y0, y1 = z["box"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return z["name"]
    i, j = int((x - LO[0]) / RC), int((y - LO[1]) / RC)
    if ROOM_LAB["raster"] is not None and 0 <= i < ROOM_LAB["raster"].shape[0] and 0 <= j < ROOM_LAB["raster"].shape[1]:
        nm = ROOM_LAB["names"].get(int(ROOM_LAB["raster"][i, j]))
        if nm:
            return nm
    if 0 <= i < fp_hr.shape[0] and 0 <= j < fp_hr.shape[1] and fp_hr[i, j]:
        return ZONES["default_zone"]
    return "outside"


# ------------------------------------------------------------------ mesh accumulation
class MB:
    """Mesh builder with per-face tags (become GeomSubsets)."""

    def __init__(self):
        self.V, self.F, self.tag = [], [], []
        self.n = 0

    def quad(self, a, b, c, d, tag):
        self.V += [a, b, c, d]
        self.F += [[self.n, self.n + 1, self.n + 2], [self.n, self.n + 2, self.n + 3]]
        self.tag += [tag, tag]
        self.n += 4

    def tri(self, a, b, c, tag):
        self.V += [a, b, c]
        self.F.append([self.n, self.n + 1, self.n + 2])
        self.tag.append(tag)
        self.n += 3

    def box(self, x0, x1, y0, y1, z0, z1, tags):
        """Axis-aligned box; tags: dict side -> tag for '-x', '+x', '-y', '+y', '-z', '+z' (default 'edge')."""
        t = lambda s: tags.get(s, tags.get("edge", "edge"))
        self.quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), t("-x"))       # outward normals
        self.quad((x1, y1, z0), (x1, y1, z1), (x1, y0, z1), (x1, y0, z0), t("+x"))
        self.quad((x1, y0, z0), (x1, y0, z1), (x0, y0, z1), (x0, y0, z0), t("-y"))
        self.quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), t("+y"))
        self.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), t("+z"))       # normal +z
        self.quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0), t("-z"))       # normal -z

    def obox(self, p0, p1, t, z0, z1, tags):
        """Oriented box around the segment p0-p1 (thickness t)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = (p1 - p0) / np.linalg.norm(p1 - p0)
        n = np.array([-d[1], d[0]]) * t / 2
        a, b, c, e = p0 - n, p1 - n, p1 + n, p0 + n
        g = lambda q, z: (float(q[0]), float(q[1]), z)
        tg = lambda s: tags.get(s, tags.get("edge", "edge"))
        self.quad(g(a, z0), g(b, z0), g(b, z1), g(a, z1), tg("-n"))
        self.quad(g(c, z0), g(e, z0), g(e, z1), g(c, z1), tg("+n"))
        self.quad(g(b, z0), g(c, z0), g(c, z1), g(b, z1), tg("edge"))
        self.quad(g(e, z0), g(a, z0), g(a, z1), g(e, z1), tg("edge"))
        self.quad(g(a, z1), g(b, z1), g(c, z1), g(e, z1), tg("+z"))
        self.quad(g(e, z0), g(c, z0), g(b, z0), g(a, z0), tg("-z"))

    def cap(self, poly, z, up, tag):
        for t_ in shapely.constrained_delaunay_triangles(poly).geoms:
            p = np.asarray(t_.exterior.coords)[:3]
            ccw = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) > 0
            if ccw != up:
                p = p[::-1]
            self.tri(*[(float(q[0]), float(q[1]), z) for q in p], tag)

    def prism(self, poly, z0, z1, tag, top=True, bottom=False, side_tag=None):
        poly = orient(poly, 1.0)
        for ring in [poly.exterior, *poly.interiors]:
            pts = np.asarray(ring.coords)[:-1]
            for a, b in zip(pts, np.roll(pts, -1, 0)):
                self.quad((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1), (a[0], a[1], z1), side_tag or tag)
        if top:
            self.cap(poly, z1, True, tag)
        if bottom:
            self.cap(poly, z0, False, tag)


# ------------------------------------------------------------------ USD helpers
stage = Usd.Stage.CreateNew(str(PAY / "building.usdc"))
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
R = "/DX_Hub_Building"
UsdGeom.Xform.Define(stage, R)
stage.SetDefaultPrim(stage.GetPrimAtPath(R))
for g in ["Floor", "Walls", "Columns", "Openings", "Openings/Doors", "Openings/Windows", "Glass", "Baseboards",
          "Ceiling", "Ceiling/Zones", "Ceiling/Coves", "Ceiling/LightPanels", "Ceiling/HVAC", "Fixtures",
          "Collision", "Looks"]:
    UsdGeom.Scope.Define(stage, f"{R}/{g}") if g == "Looks" else UsdGeom.Xform.Define(stage, f"{R}/{g}")

MATS = {}


CAL = json.load(open(CFG / "dx_hub_no_obj_v1_finishes_calibrated.json"))["materials"]
TEXDIR = PAY / "textures"
TEXDIR.mkdir(exist_ok=True)


def make_ceiling_tile_texture(path, albedo_lin, px=512):
    """Speckled acoustic tile, 300 x 600 mm staggered, texture covers 1.2 x 1.2 m."""
    from PIL import Image
    rng = np.random.default_rng(11)
    base = np.array([(1.055 * a ** (1 / 2.4) - 0.055) if a > 0.0031308 else 12.92 * a for a in albedo_lin])
    img = np.ones((px, px, 3)) * base
    img *= 1 + rng.normal(0, 0.012, (px, px, 1))
    n = int(px * px * 0.006)                                     # dark pinholes / fissures
    ys, xs = rng.integers(0, px, n), rng.integers(0, px, n)
    for yy, xx in zip(ys, xs):
        l = rng.integers(1, 4)
        img[yy, xx:xx + l] *= 0.55
    m = px / 1.2                                                  # pixels per metre
    for k in range(5):                                            # joints along x every 0.3 m
        img[:, int(k * 0.3 * m) % px] *= 0.86
    for k in range(4):                                            # staggered joints along y every 0.6 m
        y0 = int(k * 0.3 * m)
        off = 0 if k % 2 == 0 else int(0.3 * m)
        for c in range(2):
            x = (off + int(c * 0.6 * m)) % px
            img[y0:y0 + int(0.3 * m), x] *= 0.86
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def make_wall_tile_texture(path, albedo_lin, px=(512, 1024)):
    """Large porcelain wall tiles 600 x 1200 mm (portrait), texture covers 1.2 m x 2.4 m (2 x 2 tiles)."""
    from PIL import Image
    rng = np.random.default_rng(5)
    w, h = px
    base = np.array([(1.055 * a ** (1 / 2.4) - 0.055) if a > 0.0031308 else 12.92 * a for a in albedo_lin])
    img = np.ones((h, w, 3)) * base
    yy, xx = np.mgrid[0:h, 0:w]
    for tx in range(2):
        for ty in range(2):
            sl = (slice(ty * h // 2, (ty + 1) * h // 2), slice(tx * w // 2, (tx + 1) * w // 2))
            img[sl] *= 1 + rng.normal(0, 0.02)                    # tile-to-tile shade
    cloud = sum(np.sin(xx / w * f * 6.28 + rng.uniform(0, 6)) * np.sin(yy / h * f * 3.1 + rng.uniform(0, 6)) / f
                for f in (1, 2, 4, 8))
    img *= (1 + 0.025 * cloud)[..., None]                         # soft stone mottling
    img *= 1 + rng.normal(0, 0.008, (h, w, 1))
    for k in range(3):                                            # 2 mm joints
        img[:, max(min(k * w // 2, w - 1) - 1, 0):k * w // 2 + 1] *= 0.75
        img[max(min(k * h // 2, h - 1) - 1, 0):k * h // 2 + 1, :] *= 0.75
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(path)


def material(mid):
    """Finish material; colours from the video calibration (config/dx_hub_no_obj_v1_finishes_calibrated.json)."""
    if mid in MATS:
        return MATS[mid]
    spec = ZONES["finishes"].get(mid, {"rgb": [0.8, 0.8, 0.8], "rough": 0.8})
    path = f"{R}/Looks/{mid}"
    m = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    if spec.get("calibrated") in CAL:
        lin = CAL[spec["calibrated"]]["linear"]
    else:
        lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in spec["rgb"]]
    if spec.get("texture"):
        tex_file = TEXDIR / spec["texture"]
        (make_wall_tile_texture if spec.get("texture_kind") == "wall_tile" else make_ceiling_tile_texture)(tex_file, lin)
        st_r = UsdShade.Shader.Define(stage, f"{path}/STReader")
        st_r.CreateIdAttr("UsdPrimvarReader_float2")
        st_r.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(stage, f"{path}/Texture")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"./textures/{spec['texture']}")
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_r.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in lin]))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(spec.get("rough", 0.8))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(spec.get("metal", 0.0))
    if "opacity" in spec:
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(spec["opacity"])
    if "emissive" in spec:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*spec["emissive"]))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    if "emissive" in spec:
        # RTX ignores UsdPreviewSurface emission: add an OmniPBR (MDL) network for Omniverse renderers
        mdl = UsdShade.Shader.Define(stage, f"{path}/OmniPBR")
        mdl.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        mdl.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        mdl.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(v) for v in lin]))
        mdl.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(spec.get("rough", 0.5))
        mdl.CreateInput("enable_emission", Sdf.ValueTypeNames.Bool).Set(True)
        e = np.array(spec["emissive"], float)
        mdl.CreateInput("emissive_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*(e / e.max()).tolist()))
        mdl.CreateInput("emissive_intensity", Sdf.ValueTypeNames.Float).Set(float(spec.get("emissive_nits", 30000.0)))
        out_ = mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
        m.CreateSurfaceOutput("mdl").ConnectToSource(out_)
    m.GetPrim().SetCustomDataByKey("finish_id", mid)
    if spec.get("calibrated"):
        m.GetPrim().SetCustomDataByKey("calibrated_from", spec["calibrated"])
    MATS[mid] = m
    return m


def label(prim, cls, room=None):
    UsdSemantics.LabelsAPI.Apply(prim, "class").CreateLabelsAttr().Set([cls])
    if room:
        UsdSemantics.LabelsAPI.Apply(prim, "room").CreateLabelsAttr().Set([room])


def emit(path, mb, cls, mat_of_tag, room=None, st_scale=None, st_box=None):
    """Write a MB as one Mesh; one GeomSubset + material per tag. st_scale: planar XY texture coords (m per tile)."""
    if not mb.F:
        return None
    V = np.asarray(mb.V, np.float32)
    F = np.asarray(mb.F, np.int32)
    me = UsdGeom.Mesh.Define(stage, path)
    if st_box:                                                    # box mapping: walls (along, z), caps (x, y)
        st = np.zeros((len(V), 2), np.float32)
        for tri in F:
            p_ = V[tri]
            n_ = np.cross(p_[1] - p_[0], p_[2] - p_[0])
            n_ = n_ / max(np.linalg.norm(n_), 1e-9)
            if abs(n_[2]) > 0.7:
                st[tri] = p_[:, :2] / np.array(st_box[:1] * 2)
            else:
                along = p_[:, 1] if abs(n_[0]) > abs(n_[1]) else p_[:, 0]
                st[tri] = np.c_[along / st_box[0], p_[:, 2] / st_box[1]]
        UsdGeom.PrimvarsAPI(me).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy(st))
    if st_scale:
        UsdGeom.PrimvarsAPI(me).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
            Vt.Vec2fArray.FromNumpy((V[:, :2] / st_scale).astype(np.float32)))
    me.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
    me.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    me.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
    me.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())]))
    me.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    me.CreateDoubleSidedAttr(False)
    tags = np.asarray(mb.tag)
    uniq = list(dict.fromkeys(mb.tag))
    if len(uniq) == 1:
        UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(material(mat_of_tag(uniq[0])))
    else:
        api = UsdShade.MaterialBindingAPI.Apply(me.GetPrim())
        for t in uniq:
            sub = api.CreateMaterialBindSubset(Tf.MakeValidIdentifier(t), Vt.IntArray(np.where(tags == t)[0].tolist()))
            UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(material(mat_of_tag(t)))
        api.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)
    label(me.GetPrim(), cls, room)
    return me


COLL = []                                                       # (name, centre, size, rotZ_deg, semantic)


def coll_box(name, x0, x1, y0, y1, z0, z1, cls):
    COLL.append((name, ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), (x1 - x0, y1 - y0, z1 - z0), 0.0, cls))


def coll_obox(name, p0, p1, t, z0, z1, cls):
    p0, p1 = np.asarray(p0), np.asarray(p1)
    c = (p0 + p1) / 2
    L = float(np.linalg.norm(p1 - p0))
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    COLL.append((name, (c[0], c[1], (z0 + z1) / 2), (L, t, z1 - z0), ang, cls))


# ------------------------------------------------------------------ walls: rects minus doors / windows, plus fills
walls = [dict(w) for w in PLAN["walls"]]
doors = OPN["doors"]
windows = OPN["windows"]
fills = OPN.get("doors_filled_with_wall", [])


def wall_line_of(axis, u, a0, a1, reach=0.3):
    """Walls on the same line as an opening (for thickness / centre)."""
    c = [w for w in walls if w["axis"] == axis and abs(w["c"] - u) <= w["t"] / 2 + reach]
    near = sorted(c, key=lambda w: min(abs(w["v0"] - a1), abs(w["v1"] - a0), 0 if (w["v0"] < a1 and w["v1"] > a0) else 9))
    return near[0] if near else None


def subtract(axis, c, half, a0, a1):
    """Cut [a0, a1] out of every wall on this line that overlaps it."""
    global walls
    out = []
    for w in walls:
        if w["axis"] != axis or abs(w["c"] - c) > w["t"] / 2 + half or w["v1"] <= a0 or w["v0"] >= a1:
            out.append(w)
            continue
        for p0, p1 in [(w["v0"], a0), (a1, w["v1"])]:
            if p1 - p0 >= 0.03:
                out.append(dict(w, v0=p0, v1=p1))
    walls = out


extra_pieces = []                                               # (axis, c, t, v0, v1, z0, z1, kind)
for f in fills:                                                 # private doors / non-doors -> full wall
    ref = wall_line_of(f["axis"], f["face_u"], f["a0"], f["a1"])
    t = ref["t"] if ref else 0.12
    c = ref["c"] if ref else f["face_u"]
    subtract(f["axis"], c, 0.05, f["a0"], f["a1"])
    extra_pieces.append((f["axis"], c, t, f["a0"] - 0.02, f["a1"] + 0.02, 0.0, H_TOP, "wall_fill"))
door_specs = []
refs = [wall_line_of(d["axis"], d["face_u"], d["a0"], d["a1"]) for d in doors]
grp = list(range(len(doors)))                                # doors side by side share one wall line
for a in range(len(doors)):
    for b in range(a + 1, len(doors)):
        da, db = doors[a], doors[b]
        if da["axis"] == db["axis"] and abs(da["face_u"] - db["face_u"]) < 0.25 and \
                max(da["a0"], db["a0"]) - min(da["a1"], db["a1"]) < 1.0:
            grp[b] = grp[a]
line_ref = {}
for a, r in enumerate(refs):
    if r is not None and (grp[a] not in line_ref or r["v1"] - r["v0"] > line_ref[grp[a]]["v1"] - line_ref[grp[a]]["v0"]):
        line_ref[grp[a]] = r
for di, d in enumerate(doors):
    ref = line_ref.get(grp[di])
    t = ref["t"] if ref else 0.12
    c = ref["c"] if ref else d["face_u"] - d["face_sign"] * 0.06
    subtract(d["axis"], c, 0.05, d["a0"], d["a1"])
    # close small gaps between the opening and the neighbouring wall ends (jamb fillers)
    for side, a in (("lo", d["a0"]), ("hi", d["a1"])):
        nb = [w for w in walls if w["axis"] == d["axis"] and abs(w["c"] - c) < 0.2]
        if side == "lo":
            ends = [w["v1"] for w in nb if w["v1"] <= a + 0.01 and a - w["v1"] < 0.5]
            if ends and a - max(ends) > 0.01:
                extra_pieces.append((d["axis"], c, t, max(ends), a, 0.0, H_TOP, "jamb_fill"))
        else:
            ends = [w["v0"] for w in nb if w["v0"] >= a - 0.01 and w["v0"] - a < 0.5]
            if ends and min(ends) - a > 0.01:
                extra_pieces.append((d["axis"], c, t, a, min(ends), 0.0, H_TOP, "jamb_fill"))
    extra_pieces.append((d["axis"], c, t, d["a0"], d["a1"], d["z1"], H_TOP, "header"))
    door_specs.append(dict(d, wall_c=c, wall_t=t))

# ------------------------------------------------------------------ columns: keep only LiDAR-confirmed ones
from scipy.spatial import cKDTree as _KD  # noqa: E402
_btree = _KD(BAND)
CC = OVR.get("column_check", {"min_face_coverage": 0.3})
cols_ok = []
for cl in PLAN["columns"]:
    if cl.get("name"):                                           # user-defined solids
        cols_ok.append(cl)
        continue
    x0c, x1c, y0c, y1c = cl["x0"], cl["x1"], cl["y0"], cl["y1"]
    sides = {"-x": ((x0c, y0c), (x0c, y1c)), "+x": ((x1c, y0c), (x1c, y1c)),
             "-y": ((x0c, y0c), (x1c, y0c)), "+y": ((x0c, y1c), (x1c, y1c))}

    def side_cov(sd):
        ls = LineString(sides[sd])
        pts_ = [ls.interpolate(d_) for d_ in np.arange(0.025, ls.length, 0.05)]
        return float(np.mean([len(_btree.query_ball_point((q.x, q.y), 0.04)) > 0 for q in pts_])) if pts_ else 0.0
    # a column standing against a window facade must show its room-facing face; others any two faces
    fac = [w for w in windows if (w["axis"] == "x" and abs(w["glass_plane_u"] - (x0c if w["room_side"] > 0 else x1c)) < 0.5
                                  and w["a0"] - 0.5 < (y0c + y1c) / 2 < w["a1"] + 0.5)
           or (w["axis"] == "y" and abs(w["glass_plane_u"] - (y0c if w["room_side"] > 0 else y1c)) < 0.5
               and w["a0"] - 0.5 < (x0c + x1c) / 2 < w["a1"] + 0.5)]
    if fac:
        w = fac[0]
        room_side = ("+x" if w["room_side"] > 0 else "-x") if w["axis"] == "x" else ("+y" if w["room_side"] > 0 else "-y")
        cov = side_cov(room_side)
    else:
        cov = sorted([side_cov(sd) for sd in sides])[-2]
    if cov >= CC["min_face_coverage"]:
        cols_ok.append(cl)
    else:
        print(f"column dropped (LiDAR face coverage {cov:.2f}): {[round(cl[k], 2) for k in ('x0', 'x1', 'y0', 'y1')]}")
PLAN["columns"] = cols_ok

# ------------------------------------------------------------------ facade regularization (one straight window strip)
FR_ = OVR.get("facade_regularization")
if FR_:
    groups_ = []
    for wn in sorted(windows, key=lambda q: (q["axis"], q["room_side"], q["glass_plane_u"])):
        g_ = next((g for g in groups_ if g[0]["axis"] == wn["axis"] and g[0]["room_side"] == wn["room_side"]
                   and abs(g[0]["glass_plane_u"] - wn["glass_plane_u"]) < 0.3), None)
        (g_.append(wn) if g_ is not None else groups_.append([wn]))
    new_windows = []
    for g_ in groups_:
        ax_, sgn_ = g_[0]["axis"], g_[0]["room_side"]
        L_ = np.array([w["a1"] - w["a0"] for w in g_])

        def wmed(key):
            v = np.array([w[key] for w in g_], float)
            o = np.argsort(v)
            cw = np.cumsum(L_[o])
            return float(v[o][np.searchsorted(cw, cw[-1] / 2)])
        gu_, sd_, sz_, hz_ = wmed("glass_plane_u"), wmed("sill_face_depth"), wmed("sill_z"), wmed("head_z")
        spans = sorted([(w["a0"], w["a1"]) for w in g_])
        merged_ = [list(spans[0])]
        for a, b in spans[1:]:                                   # bridge gaps up to bridge_gap_m
            if a - merged_[-1][1] <= FR_["bridge_gap_m"]:
                merged_[-1][1] = max(merged_[-1][1], b)
            else:
                merged_.append([a, b])
        k_ = 0 if ax_ == "x" else 1
        strip_lo, strip_hi = sorted([gu_ - sgn_ * 0.05, gu_ + sgn_ * (sd_ - 0.02)])
        for a0_, a1_ in merged_:
            nb = int(np.ceil((a1_ - a0_) / 0.05))
            centers = a0_ + (np.arange(nb) + 0.5) * 0.05
            # LiDAR: blind / glass plane visible at 1.6-2.3 m
            pts = BAND[(np.abs(BAND[:, k_] - gu_) < FR_["blind_plane_tol_m"]) & (BAND[:, 1 - k_] > a0_) & (BAND[:, 1 - k_] < a1_)]
            cnt = np.bincount(np.clip(((pts[:, 1 - k_] - a0_) / 0.05).astype(int), 0, nb - 1), minlength=nb)
            orig = np.zeros(nb, bool)                            # measured bays stay windows
            for w in g_:
                orig |= (centers > w["a0"]) & (centers < w["a1"])
            win = orig | (cnt > 0)                               # bridged gaps only where the blind plane is seen
            # obstacles: perpendicular walls / columns / diagonal walls crossing the sill strip
            obst = np.zeros(nb, bool)
            obs_geoms = [sbox(*((w["c"] - w["t"] / 2, w["v0"], w["c"] + w["t"] / 2, w["v1"]) if w["axis"] == "x"
                                else (w["v0"], w["c"] - w["t"] / 2, w["v1"], w["c"] + w["t"] / 2)))
                         for w in walls if w["axis"] != ax_]
            obs_geoms += [sbox(cl["x0"], cl["y0"], cl["x1"], cl["y1"]) for cl in PLAN["columns"]]
            for og in obs_geoms:
                x0o, y0o, x1o, y1o = og.bounds
                lo_o, hi_o = (x0o, x1o) if ax_ == "x" else (y0o, y1o)
                if hi_o < strip_lo or lo_o > strip_hi:
                    continue
                ao0, ao1 = (y0o, y1o) if ax_ == "x" else (x0o, x1o)
                obst |= (centers > ao0 - 0.005) & (centers < ao1 + 0.005)
            win &= ~obst
            # close small holes (occluded by lamps / plants) that do not cross an obstacle
            hole_ = int(FR_["close_holes_m"] / 0.05)
            lab_, n_ = ndimage.label(~win)
            for q_ in range(1, n_ + 1):
                idx_ = np.where(lab_ == q_)[0]
                if len(idx_) <= hole_ and idx_[0] > 0 and idx_[-1] < nb - 1 and not obst[idx_].any():
                    win[idx_] = True
            lab_, n_ = ndimage.label(win)
            for q_ in range(1, n_ + 1):
                idx_ = np.where(lab_ == q_)[0]
                b0, b1 = a0_ + idx_[0] * 0.05, a0_ + (idx_[-1] + 1) * 0.05
                if b1 - b0 < FR_["min_window_m"]:
                    continue
                # snap the ends to the obstacle faces / original span ends
                new_windows.append({**g_[0], "glass_plane_u": gu_, "sill_face_depth": sd_, "sill_z": sz_, "head_z": hz_,
                                    "a0": round(max(b0, a0_), 3), "a1": round(min(b1, a1_), 3),
                                    "width": round(min(b1, a1_) - max(b0, a0_), 3), "mullion_pitch": g_[0].get("mullion_pitch"),
                                    "source": "facade-regularized"})
        print(f"facade {ax_}={gu_:.3f} (side {sgn_:+d}): {len(g_)} bays -> "
              f"{sum(1 for w in new_windows if w['axis'] == ax_ and abs(w['glass_plane_u'] - gu_) < 1e-6)} bays; "
              f"sill depth {sd_:.3f}, sill {sz_:.3f}, head {hz_:.3f}")
    # windows enclosed by a pilaster box (a parallel wall in front covering the whole bay) are hidden: drop them
    vis_windows = []
    for w in new_windows:
        hidden = any(o["axis"] == w["axis"] and 0.3 < (o["c"] - w["glass_plane_u"]) * w["room_side"] < 1.5
                     and o["v0"] <= w["a0"] + 0.02 and o["v1"] >= w["a1"] - 0.02 for o in walls)
        if hidden:
            print(f"window hidden inside a pilaster box dropped: {w['axis']}={w['glass_plane_u']:.2f} {w['a0']}..{w['a1']}")
        else:
            vis_windows.append(w)
    windows = vis_windows

window_specs = []
for wn in windows:
    sgn = wn["room_side"]
    gu = wn["glass_plane_u"]
    t_out = 0.25
    c_out = gu - sgn * (0.05 + t_out / 2)                      # outer wall beyond the glass line
    subtract(wn["axis"], c_out, 0.2, wn["a0"], wn["a1"])
    sill_d = wn["sill_face_depth"]
    sill_z = wn["sill_z"] or 0.8
    head = min(wn["head_z"] or 2.6, ceil_at(*((gu, (wn["a0"] + wn["a1"]) / 2) if wn["axis"] == "x" else ((wn["a0"] + wn["a1"]) / 2, gu))) - 0.02)
    # outer wall below the sill and above the head; room-side sill block (wall below the window, measured depth)
    extra_pieces.append((wn["axis"], c_out, t_out, wn["a0"], wn["a1"], 0.0, sill_z, "exterior_below_sill"))
    extra_pieces.append((wn["axis"], c_out, t_out, wn["a0"], wn["a1"], head, H_TOP, "exterior_header"))
    sw0 = gu - sgn * 0.05                                        # sill block: glass line -> measured sill face
    sw1 = gu + sgn * sill_d
    extra_pieces.append((wn["axis"], (sw0 + sw1) / 2, abs(sw1 - sw0), wn["a0"], wn["a1"], 0.0, sill_z - 0.03, "sill_wall"))
    extra_pieces.append((wn["axis"], (sw0 + sw1) / 2 + sgn * 0.01, abs(sw1 - sw0) + 0.02, wn["a0"], wn["a1"], sill_z - 0.03, sill_z, "sill_ledge"))
    if head < H_TOP - 0.01:
        extra_pieces.append((wn["axis"], gu + sgn * 0.06, 0.12, wn["a0"], wn["a1"], head, H_TOP, "window_head_soffit"))
    window_specs.append(dict(wn, head_eff=head, sill_eff=sill_z))


def rect_of(axis, c, t, v0, v1):
    return (c - t / 2, c + t / 2, v0, v1) if axis == "x" else (v0, v1, c - t / 2, c + t / 2)


def side_zone(x0, x1, y0, y1, z, side):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    off = 0.2
    q = {"-x": (x0 - off, cy), "+x": (x1 + off, cy), "-y": (cx, y0 - off), "+y": (cx, y1 + off)}[side]
    return zone_at(*q)


def wall_tags(x0, x1, y0, y1, kind):
    tags = {}
    for s in ("-x", "+x", "-y", "+y"):
        zn = side_zone(x0, x1, y0, y1, 1.0, s)
        tags[s] = f"wall_{zn}" if zn != "outside" else "wall_exterior"
    tags["+z"] = tags["-z"] = "wall_hidden"
    tags["edge"] = "wall_" + ZONES["default_zone"]
    if kind == "sill_ledge":
        tags = {k: "sill_ledge" for k in ("-x", "+x", "-y", "+y", "+z", "-z", "edge")}
    return tags


def finish_of_tag(tag):
    if tag in ZONES["finishes"]:
        return tag
    if tag.startswith("wall_"):
        z = tag[5:]
        return ZONES["zone_finish"].get(z, {}).get("wall", "wall_paint")
    if tag.startswith("floor_"):
        z = tag[6:]
        return ZONES["zone_finish"].get(z, {}).get("floor", "floor_hall")
    return "wall_paint"


# ------------------------------------------------------------------ absorb wall stubs (< 15 cm) into a neighbour
def _absorb_stubs():
    global walls
    keep = []
    for w in walls:
        if w["v1"] - w["v0"] >= 0.15:
            keep.append(w)
            continue
        lo_w, hi_w = w["c"] - w["t"] / 2, w["c"] + w["t"] / 2
        done = False
        for q, ep in enumerate(extra_pieces):                    # extend an adjacent fill/jamb piece
            ax_, c_, t_, v0_, v1_, z0_, z1_, kind_ = ep
            if ax_ != w["axis"] or z0_ > 0.01 or min(c_ + t_ / 2, hi_w) - max(c_ - t_ / 2, lo_w) <= 0:
                continue
            if -0.03 <= v0_ - w["v1"] <= 0.03 or -0.03 <= w["v0"] - v1_ <= 0.03 or (v0_ <= w["v0"] and w["v1"] <= v1_):
                extra_pieces[q] = (ax_, c_, t_, min(v0_, w["v0"]), max(v1_, w["v1"]), z0_, z1_, kind_)
                done = True
                break
        if not done:
            for o in keep + [x for x in walls if x is not w]:
                if o is w or o["axis"] != w["axis"] or o["v1"] - o["v0"] < 0.15:
                    continue
                if min(o["c"] + o["t"] / 2, hi_w) - max(o["c"] - o["t"] / 2, lo_w) <= 0:
                    continue
                if -0.03 <= o["v0"] - w["v1"] <= 0.03 or -0.03 <= w["v0"] - o["v1"] <= 0.03:
                    o["v0"], o["v1"] = min(o["v0"], w["v0"]), max(o["v1"], w["v1"])
                    done = True
                    break
        if not done:
            keep.append(w)
    walls = keep


_absorb_stubs()

# ------------------------------------------------------------------ gap closing (no broken walls)
from shapely.geometry import Point  # noqa: E402
from shapely.strtree import STRtree  # noqa: E402

GC = OVR.get("gap_closing", {"always_below_m": 0.6, "check_las_up_to_m": 1.5})

# (a) L corners: two perpendicular walls whose ends both stop short of their common corner -> extend both
n_corner = 0
for A in walls:
    for B in walls:
        if A is B or A["axis"] == B["axis"]:
            continue
        cA, cB = A["c"], B["c"]                                   # corner = (cB along A) / (cA along B)
        if A["v0"] - 0.01 <= cB <= A["v1"] + 0.01 or B["v0"] - 0.01 <= cA <= B["v1"] + 0.01:
            continue                                              # already meets (T or L)
        eA = min(("v0", "v1"), key=lambda e: abs(A[e] - cB))
        eB = min(("v0", "v1"), key=lambda e: abs(B[e] - cA))
        if abs(A[eA] - cB) > 0.6 or abs(B[eB] - cA) > 0.6:
            continue
        sA = 1 if cB > A[eA] else -1
        sB = 1 if cA > B[eB] else -1
        A[eA] = cB + sA * B["t"] / 2
        B[eB] = cA + sB * A["t"] / 2
        n_corner += 1
# (b) parallel walls with a small sideways offset that overlap (or almost meet) along the line -> bridge
bridges = []
for i, A in enumerate(walls):
    for B in walls[i + 1:]:
        if A["axis"] != B["axis"]:
            continue
        aLo, aHi = A["c"] - A["t"] / 2, A["c"] + A["t"] / 2
        bLo, bHi = B["c"] - B["t"] / 2, B["c"] + B["t"] / 2
        across = max(bLo - aHi, aLo - bHi)
        ov0, ov1 = max(A["v0"], B["v0"]), min(A["v1"], B["v1"])
        if not (-0.3 <= ov1 - ov0 <= 0.5) or across > 0.35:
            continue
        if across <= 0.005 and ov1 - ov0 <= 0.005 and ov1 - ov0 > -0.005:
            continue                                              # already touching
        if across <= 0.005 and ov1 - ov0 > 0.005:
            continue                                              # overlapping side by side: nothing to bridge
        lo_, hi_ = min(aLo, bLo), max(aHi, bHi)
        a0_, a1_ = (ov0, ov1) if ov1 > ov0 else (ov1, ov0)
        bridges.append((A["axis"], (lo_ + hi_) / 2, hi_ - lo_, a0_ - 0.005, a1_ + 0.005, 0.0, H_TOP, "gap_fill_jog"))
extra_pieces += bridges
print(f"corners closed: {n_corner}, jogs bridged: {len(bridges)}")
elems = []                                                       # (polygon, axis, lo, hi, a0, a1, kind)


WALL_OF = {}


def add_elem(axis, lo, hi, a0, a1, kind, src=None):
    x0, x1, y0, y1 = (lo, hi, a0, a1) if axis == "x" else (a0, a1, lo, hi)
    if src is not None:
        WALL_OF[len(elems)] = src
    elems.append((sbox(x0, y0, x1, y1), axis, lo, hi, a0, a1, kind))


for w in walls:
    add_elem(w["axis"], w["c"] - w["t"] / 2, w["c"] + w["t"] / 2, w["v0"], w["v1"], "wall", src=w)
for (axis, c, t, v0, v1, z0, z1, kind) in extra_pieces:
    if z0 < 1.0 < z1:
        add_elem(axis, c - t / 2, c + t / 2, v0, v1, kind)
for wn in window_specs:
    sgn, gu = wn["room_side"], wn["glass_plane_u"]
    lo_, hi_ = sorted([gu - sgn * 0.30, gu + sgn * wn["sill_face_depth"]])
    add_elem(wn["axis"], lo_, hi_, wn["a0"], wn["a1"], "window")
for g in OPN["glass_lines"]:
    add_elem(g["axis"], g["c"] - 0.03, g["c"] + 0.03, g["a0"], g["a1"], "glass")
for d in door_specs:
    add_elem(d["axis"], d["wall_c"] - d["wall_t"] / 2, d["wall_c"] + d["wall_t"] / 2, d["a0"], d["a1"], "door")
for cl in PLAN["columns"]:
    elems.append((sbox(cl["x0"], cl["y0"], cl["x1"], cl["y1"]), None, 0, 0, 0, 0, "column"))
for dg in PLAN["diagonal_walls"]:
    elems.append((LineString([dg["p0"], dg["p1"]]).buffer(dg["t"] / 2, cap_style=2), None, 0, 0, 0, 0, "diagonal"))
etree = STRtree([e[0] for e in elems])
btree = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(BAND)
fills_gc = set()
FILLED = []
for k, (pg, axis, lo, hi, a0, a1, kind) in enumerate(elems):
    if axis is None or kind == "door":
        continue
    mid = (lo + hi) / 2
    for sg, end in ((-1, a0), (1, a1)):
        hit = None
        for g in np.arange(0.01, GC["check_las_up_to_m"] + 0.001, 0.01):
            a = end + sg * g
            q = Point(mid, a) if axis == "x" else Point(a, mid)
            idx = [i for i in etree.query(q) if i != k and elems[i][0].distance(q) < 1e-9]
            if idx:
                hit = (g - 0.01, elems[idx[0]])
                break
        if hit is None or hit[0] < 0.005:
            continue
        g, other = hit                                           # a door edge ends the fill (never fills the doorway)
        if g >= GC["always_below_m"]:
            # LiDAR check: is the gap a wall the detector missed (points along it) or a real passage?
            n_bins = max(int(g / 0.05), 1)
            occ_bins = 0
            for b_ in range(n_bins):
                a = end + sg * (b_ + 0.5) * 0.05
                q = (mid, a) if axis == "x" else (a, mid)
                if len(btree.query_ball_point(q, max((hi - lo) / 2, 0.05) + 0.03)) >= 3:
                    occ_bins += 1
            if occ_bins / n_bins < 0.5:
                continue
        a_lo, a_hi = sorted([end, end + sg * g])
        if any(f_[0] == axis and abs(f_[1] - mid) < 0.15 and min(f_[3], a_hi) - max(f_[2], a_lo) > 0.5 * (a_hi - a_lo)
               for f_ in FILLED):
            continue                                             # the same gap seen from its other end
        FILLED.append((axis, mid, a_lo, a_hi))
        if kind == "window":                                     # pier between window bays
            fills_gc.add((axis, round(lo, 2), round(a_lo, 2)))
            extra_pieces.append((axis, (lo + hi) / 2, hi - lo, a_lo - 0.005, a_hi + 0.005, 0.0, H_TOP, "gap_fill_pier"))
            continue
        fx0, fx1, fy0, fy1 = (lo + 0.01, hi - 0.01, a_lo + 0.01, a_hi - 0.01) if axis == "x" else \
            (a_lo + 0.01, a_hi - 0.01, lo + 0.01, hi - 0.01)
        if fx1 > fx0 and fy1 > fy0 and any(e_[6] in ("glass", "window", "door") and e_[0].intersects(sbox(fx0, fy0, fx1, fy1))
                                           for e_ in elems):
            continue                                             # never cover glass / windows / doors with a wall
        if k in WALL_OF and g < 0.15:                             # short gap: extend the wall itself
            w_ = WALL_OF[k]
            if sg < 0:
                w_["v0"] = min(w_["v0"], a_lo - 0.005)
            else:
                w_["v1"] = max(w_["v1"], a_hi + 0.005)
            fills_gc.add(("ext", k, sg))
            continue
        t_ = (hi - lo) if kind != "glass" else 0.10
        key = (axis, round(mid, 2), round(a_lo, 2))
        if key not in fills_gc:
            fills_gc.add(key)
            extra_pieces.append((axis, mid, t_, a_lo - 0.005, a_hi + 0.005, 0.0, H_TOP, "gap_fill"))
print(f"gap closing: {len(fills_gc)} fillers")

# ------------------------------------------------------------------ enclosed rooms (seed zones)
occ_r = np.zeros(fp_hr.shape, bool)


def _paint(x0, x1, y0, y1):
    i0, i1 = int(np.floor((x0 - LO[0]) / RC)), int(np.ceil((x1 - LO[0]) / RC))
    j0, j1 = int(np.floor((y0 - LO[1]) / RC)), int(np.ceil((y1 - LO[1]) / RC))
    occ_r[max(i0, 0):max(i1, i0 + 1), max(j0, 0):max(j1, j0 + 1)] = True


for (ax_, c_, t_, v0_, v1_, z0_, z1_, kind_) in [(w["axis"], w["c"], w["t"], w["v0"], w["v1"], 0.0, H_TOP, "wall") for w in walls] + extra_pieces:
    if z0_ < 1.0 < z1_:
        _paint(*rect_of(ax_, c_, max(t_, 0.03), v0_, v1_))
for wn in window_specs:
    _paint(*rect_of(wn["axis"], wn["glass_plane_u"], 0.10, wn["a0"], wn["a1"]))
for g in OPN["glass_lines"]:
    _paint(*rect_of(g["axis"], g["c"], 0.05, g["a0"], g["a1"]))
for d in door_specs:
    _paint(*rect_of(d["axis"], d["wall_c"], d["wall_t"], d["a0"], d["a1"]))
for cl in PLAN["columns"]:
    _paint(cl["x0"], cl["x1"], cl["y0"], cl["y1"])
for dg in PLAN["diagonal_walls"]:
    for t_ in np.linspace(0, 1, 400):
        q = np.array(dg["p0"]) + t_ * (np.array(dg["p1"]) - np.array(dg["p0"]))
        _paint(q[0] - 0.06, q[0] + 0.06, q[1] - 0.06, q[1] + 0.06)
occ_r = ndimage.binary_dilation(occ_r, iterations=2)          # close raster slits < 10 cm (not real openings)
lab_r, n_r = ndimage.label(~occ_r & fp_hr)
np.savez(WORK / "v3_room_occ.npz", occ=occ_r, lab=lab_r)
names_r = {}
for z in ZONES["zones"]:
    if "seed" not in z:
        continue
    i, j = int((z["seed"][0] - LO[0]) / RC), int((z["seed"][1] - LO[1]) / RC)
    k_ = int(lab_r[i, j])
    area = float((lab_r == k_).sum() * RC * RC) if k_ else 0.0
    if k_ and area < 250:
        names_r[k_] = z["name"]
        print(f"room zone {z['name']}: {area:.1f} m2")
    else:
        print(f"room zone {z['name']}: not enclosed (area {area:.0f} m2) -> default finishes")
# grow room labels under their walls (so wall faces / floor edges pick the room)
lab_keep = np.where(np.isin(lab_r, list(names_r.keys())), lab_r, 0)
_, (ni_, nj_) = ndimage.distance_transform_edt(lab_keep == 0, return_indices=True)
grown = lab_keep[ni_, nj_]
dist_ = ndimage.distance_transform_edt(lab_keep == 0) * RC
ROOM_LAB["raster"] = np.where(dist_ <= 0.15, grown, 0)
ROOM_LAB["names"] = names_r

all_pieces = [(w["axis"], w["c"], w["t"], w["v0"], w["v1"], 0.0, H_TOP, "wall") for w in walls] + extra_pieces
for k, (axis, c, t, v0, v1, z0, z1, kind) in enumerate(all_pieces):
    if v1 - v0 < 0.005 or z1 - z0 < 0.005:
        continue
    x0, x1, y0, y1 = rect_of(axis, c, t, v0, v1)
    mb = MB()
    tags_ = wall_tags(x0, x1, y0, y1, kind)
    if kind == "header":
        for hf in OVR.get("door_header_finish", []):
            if hf["axis"] == axis and abs(c - hf["u"]) < 0.3 and v0 - 0.05 <= hf["a_mid"] <= v1 + 0.05:
                tags_[("-" if hf["side"] < 0 else "+") + axis] = hf["finish"]
    mb.box(x0, x1, y0, y1, z0, z1, tags_)
    cls = {"sill_ledge": "window_sill", "sill_wall": "wall", "header": "wall", "jamb_fill": "wall"}.get(kind, "wall")
    emit(f"{R}/Walls/{kind}_{k:03d}", mb, cls, finish_of_tag, st_box=ZONES["finishes"]["wall_tile_gray"]["uv_scale_m"])
    coll_box(f"{kind}_{k:03d}", x0, x1, y0, y1, z0, z1, cls)
for k, d in enumerate(PLAN["diagonal_walls"]):
    mb = MB()
    p0_, p1_ = np.array(d["p0"]), np.array(d["p1"])
    dd = (p1_ - p0_) / np.linalg.norm(p1_ - p0_)
    nn = np.array([-dd[1], dd[0]])
    mid_ = (p0_ + p1_) / 2
    side_tag = {}
    for nm_, sg in (("+n", 1), ("-n", -1)):
        zn_ = zone_at(*(mid_ + sg * nn * 0.4))
        tv = OVR.get("diagonal_wall_finish", {}).get("hall_side") if zn_ == ZONES["default_zone"] else None
        side_tag[nm_] = tv or f"wall_{zn_}"
    mb.obox(d["p0"], d["p1"], d["t"], 0.0, H_TOP, {**side_tag, "edge": "wall_" + ZONES["default_zone"],
                                                    "+z": "wall_hidden", "-z": "wall_hidden"})
    emit(f"{R}/Walls/diagonal_{k}", mb, "wall", finish_of_tag)
    coll_obox(f"diagonal_{k}", d["p0"], d["p1"], d["t"], 0.0, H_TOP, "wall")
for k, cl in enumerate(PLAN["columns"]):
    mb = MB()
    mb.box(cl["x0"], cl["x1"], cl["y0"], cl["y1"], 0.0, H_TOP, wall_tags(cl["x0"], cl["x1"], cl["y0"], cl["y1"], "column"))
    emit(f"{R}/Columns/C{k:02d}", mb, "column", finish_of_tag)
    coll_box(f"column_{k}", cl["x0"], cl["x1"], cl["y0"], cl["y1"], 0.0, H_TOP, "column")
print(f"walls: {len(all_pieces)} pieces (+{len(PLAN['diagonal_walls'])} diagonal, {len(PLAN['columns'])} columns)")

# ------------------------------------------------------------------ doors: casing, leaf (own prim), handle
CASE_W, CASE_P, LEAF_T = 0.06, 0.015, 0.04
for k, d in enumerate(door_specs):
    ax, c, t, a0, a1, h = d["axis"], d["wall_c"], d["wall_t"], d["a0"], d["a1"], d["z1"]
    base = f"{R}/Openings/Doors/Door_{k:02d}"
    UsdGeom.Xform.Define(stage, base)
    mb = MB()
    for face in (-1, 1):                                       # casings on both faces: two jambs + head
        u_face = c + face * t / 2
        u0, u1 = sorted([u_face, u_face + face * CASE_P])
        for (p0, p1, z0, z1) in [(a0 - CASE_W, a0, 0.0, h + CASE_W), (a1, a1 + CASE_W, 0.0, h + CASE_W),
                                 (a0 - CASE_W, a1 + CASE_W, h, h + CASE_W)]:
            x0, x1, y0, y1 = (u0, u1, p0, p1) if ax == "x" else (p0, p1, u0, u1)
            mb.box(x0, x1, y0, y1, z0, z1, {"edge": "door_frame"})
    # jamb lining inside the opening
    for (p0, p1) in [(a0, a0 + 0.015), (a1 - 0.015, a1)]:
        x0, x1, y0, y1 = rect_of(ax, c, t, p0, p1)
        mb.box(x0, x1, y0, y1, 0.0, h, {"edge": "door_frame"})
    x0, x1, y0, y1 = rect_of(ax, c, t, a0, a1)
    mb.box(x0, x1, y0, y1, h - 0.015, h, {"edge": "door_frame"})
    emit(f"{base}/Frame", mb, "door_frame", lambda tg: tg)
    # hinge (rule): jamb nearer to the closest wall corner along the line; opens away from the corridor side
    line_w = [w for w in PLAN["walls"] if w["axis"] == ax and abs(w["c"] - c) < 0.3]
    ends = [w["v0"] for w in line_w] + [w["v1"] for w in line_w]
    d_lo = min([abs(e - a0) for e in ends if e <= a0 + 0.01] or [9])
    d_hi = min([abs(e - a1) for e in ends if e >= a1 - 0.01] or [9])
    hinge_at = a0 if d_lo <= d_hi else a1
    sides = {s: zone_at(*((c + s * 0.6, (a0 + a1) / 2) if ax == "x" else ((a0 + a1) / 2, c + s * 0.6))) for s in (-1, 1)}
    swing_to = min(sides, key=lambda s: 1 if sides[s].startswith("corridor") else 0)
    gap = 0.004
    leafs = [(a0 + gap, a1 - gap)] if d["leaf_type"] == "single_swing" else [(a0 + gap, (a0 + a1) / 2 - gap / 2), ((a0 + a1) / 2 + gap / 2, a1 - gap)]
    for q, (l0, l1) in enumerate(leafs):
        lp = f"{base}/Leaf{'' if len(leafs) == 1 else q}"
        mb = MB()
        x0, x1, y0, y1 = rect_of(ax, c, LEAF_T, l0, l1)
        mb.box(x0, x1, y0, y1, 0.01, h - gap, {"edge": "door_leaf"})
        hinge_q = hinge_at if len(leafs) == 1 else (a0 if q == 0 else a1)
        hp = l1 - 0.07 if abs(hinge_q - l0) < abs(hinge_q - l1) else l0 + 0.07       # handle on the free edge
        for s in (-1, 1):
            u_h = c + s * (LEAF_T / 2 + 0.03)
            hx0, hx1, hy0, hy1 = (u_h - 0.012, u_h + 0.012, hp - 0.06, hp + 0.06) if ax == "x" else (hp - 0.06, hp + 0.06, u_h - 0.012, u_h + 0.012)
            mb.box(hx0, hx1, hy0, hy1, 0.98, 1.02, {"edge": "door_handle"})
        leaf = emit(lp, mb, "door", lambda tg: tg)
        p = leaf.GetPrim()
        p.SetCustomDataByKey("hinge_along", float(hinge_q))
        p.SetCustomDataByKey("hinge_axis_point", Gf.Vec3d(*((c, hinge_q, 0) if ax == "x" else (hinge_q, c, 0))))
        p.SetCustomDataByKey("swing_side", int(swing_to))
        p.SetCustomDataByKey("hinge_source", d.get("hinge", "estimated"))
        p.SetCustomDataByKey("always_closed", bool(d.get("always_closed", False)))
        coll_box(f"door_{k}_{q}", x0, x1, y0, y1, 0.01, h, "door")
    stage.GetPrimAtPath(base).SetCustomDataByKey("width", float(a1 - a0))
    stage.GetPrimAtPath(base).SetCustomDataByKey("height", float(h))
print(f"doors: {len(door_specs)}")

# ------------------------------------------------------------------ windows: frame, glass, transom, blinds
FR = 0.05
for k, wn in enumerate(window_specs):
    ax, gu, sgn, a0, a1 = wn["axis"], wn["glass_plane_u"], wn["room_side"], wn["a0"], wn["a1"]
    sill, head = wn["sill_eff"], wn["head_eff"]
    base = f"{R}/Openings/Windows/Window_{k:02d}"
    UsdGeom.Xform.Define(stage, base)
    pitch = wn.get("mullion_pitch") or 1.54
    n = max(1, int(round((a1 - a0) / pitch)))
    edges = np.linspace(a0, a1, n + 1)
    transom = min(sill + 0.38, head - 0.3)                    # lower vent sash (measured 0.85-1.20 on the west wall)
    # the measured plane is the blind (glass returns almost nothing): glass + frames sit 5 cm outside it
    gq = gu - sgn * 0.05
    mb = MB()
    u0, u1 = sorted([gq - sgn * 0.035, gq + sgn * 0.035])
    for e in edges:
        x0, x1, y0, y1 = (u0, u1, e - FR / 2, e + FR / 2) if ax == "x" else (e - FR / 2, e + FR / 2, u0, u1)
        mb.box(x0, x1, y0, y1, sill, head, {"edge": "window_frame"})
    for zb in (sill, transom, head - FR):
        x0, x1, y0, y1 = (u0, u1, a0, a1) if ax == "x" else (a0, a1, u0, u1)
        mb.box(x0, x1, y0, y1, zb, zb + FR, {"edge": "window_frame"})
    emit(f"{base}/Frame", mb, "window", lambda tg: tg)
    mb = MB()
    x0, x1, y0, y1 = rect_of(ax, gq, 0.008, a0 - 0.02, a1 + 0.02)   # 2 cm into the reveals: no slits
    mb.box(x0, x1, y0, y1, sill + FR, head - FR, {"edge": "glass_window"})
    emit(f"{base}/Glass", mb, "window_glass", lambda tg: tg)
    coll_box(f"window_glass_{k}", x0, x1, y0, y1, sill, head, "window_glass")
    # blinds (variant controlled): thin translucent sheet 8 cm into the room, from the head down
    for state, z_bot in (("Down", transom + FR), ("Half", (transom + head) / 2), ("Up", head - 0.12)):
        mb = MB()
        x0, x1, y0, y1 = rect_of(ax, gu, 0.006, a0 + 0.02, a1 - 0.02)
        mb.box(x0, x1, y0, y1, z_bot, head - 0.01, {"edge": "blind"})
        bl = emit(f"{base}/Blind_{state}", mb, "blind", lambda tg: tg)
print(f"windows: {len(window_specs)}")

# ------------------------------------------------------------------ glass partitions (A-F)
DEF_BAND = [0.49, 1.72]                                       # median of the measured bands (D, E, F) -> B, C
for g in OPN["glass_lines"]:
    ax, c, a0, a1 = g["axis"], g["c"], g["a0"], g["a1"]
    name = g["name"]
    head = g["head_z"] or 2.34
    band = g["frosted_band_z"] if name in ("A_RoomPartition", "D_GlassDoorCorridor", "E_NorthGlassDoor", "F_RobotRoomNorth") else DEF_BAND
    lay = OVR["glass_door_layout"].get(name, {})
    opaque_doors = lay.get("door_leaves") == "opaque"
    clear_doors = lay.get("door_glass") == "clear"
    black_header = lay.get("header") == "black"
    base = f"{R}/Glass/{name}"
    UsdGeom.Xform.Define(stage, base)
    fr = MB()
    top = ceil_at(*((c, (a0 + a1) / 2) if ax == "x" else ((a0 + a1) / 2, c)))
    hdr = MB()                                                  # header between the frame and the ceiling
    if top - head > 0.02:
        x0, x1, y0, y1 = rect_of(ax, c, 0.10, a0, a1)
        hdr.box(x0, x1, y0, y1, head, H_TOP, {"edge": "glass_frame"} if black_header else wall_tags(x0, x1, y0, y1, "header"))
        coll_box(f"glass_header_{name}", x0, x1, y0, y1, head, H_TOP, "wall")
    emit(f"{base}/Header", hdr, "wall", finish_of_tag)
    for pi_, p in enumerate(g["panels"]):
        p0, p1 = p["a0"], p["a1"]
        for e in (p0, p1):                                      # posts (also colliders: no slits between panes)
            x0, x1, y0, y1 = rect_of(ax, c, 0.06, e - 0.025, e + 0.025)
            fr.box(x0, x1, y0, y1, 0.0, head, {"edge": "glass_frame"})
            coll_box(f"glass_post_{name}_{pi_}_{int(e * 1000)}", x0, x1, y0, y1, 0.0, head, "glass_partition")
        if p["type"] == "open_doorway":                         # black frame head over the open doorway
            x0, x1, y0, y1 = rect_of(ax, c, 0.10, p0, p1)
            fr.box(x0, x1, y0, y1, head - 0.06, head, {"edge": "glass_frame"})
            coll_box(f"doorway_head_{name}_{pi_}", x0, x1, y0, y1, head - 0.06, head, "door_frame")
            continue
        x0, x1, y0, y1 = rect_of(ax, c, 0.06, p0, p1)
        fr.box(x0, x1, y0, y1, head - 0.05, head, {"edge": "glass_frame"})
        fr.box(x0, x1, y0, y1, 0.0, 0.04, {"edge": "glass_frame"})
        is_door = p["type"] in ("sliding_door", "swing_door")
        pane = MB()
        gx0, gx1, gy0, gy1 = rect_of(ax, c, 0.012, p0 + 0.025, p1 - 0.025)
        if p["type"] == "wall_panel":                           # opaque painted panel inside the black frame
            px0, px1, py0, py1 = rect_of(ax, c, 0.05, p0 + 0.025, p1 - 0.025)
            pane.box(px0, px1, py0, py1, 0.04, head - 0.05, {"edge": "wall_paint"})
            emit(f"{base}/WallPanel_{pi_:02d}", pane, "wall", lambda tg: tg)
            coll_box(f"glass_{name}_{pi_}", px0, px1, py0, py1, 0.0, head, "wall")
            continue
        if is_door and clear_doors:
            pane.box(gx0, gx1, gy0, gy1, 0.04, head - 0.05, {"edge": "glass_clear"})
        elif is_door and opaque_doors:
            pane.box(gx0, gx1, gy0, gy1, 0.04, head - 0.05, {"edge": "glass_frosted"})
        else:
            for z0, z1, tg in ((0.04, band[0], "glass_clear"), (band[0], band[1], "glass_frosted"), (band[1], head - 0.05, "glass_clear")):
                if z1 - z0 > 0.005:
                    pane.box(gx0, gx1, gy0, gy1, z0, z1, {"edge": tg})
        pp = emit(f"{base}/{'Door' if is_door else 'Pane'}_{pi_:02d}", pane, "door" if is_door else "glass_partition", lambda tg: tg)
        if is_door:
            pp.GetPrim().SetCustomDataByKey("door_type", p["type"])
            pp.GetPrim().SetCustomDataByKey("always_closed", name in ("B_SouthRooms", "C_CorridorEast"))
        coll_box(f"glass_{name}_{pi_}", gx0, gx1, gy0, gy1, 0.0, head, "glass_partition")
    emit(f"{base}/Frames", fr, "glass_partition", lambda tg: tg)
print(f"glass lines: {len(OPN['glass_lines'])}")

# ------------------------------------------------------------------ counters and showcase
src = open(Path(__file__).with_name("07_building_v3_plan.py")).read()
exec(src[src.index("def rounded_u"):src.index("def face_x")])           # rounded_u()


def u_polygon(cn):
    s = 1 if cn["opening"] == "+y" else -1
    o, i = cn["outer"], cn["inner"]
    po = v3_geom.sample(rounded_u(o["x_left"], o["x_right"], o["y_end"], (o["corner_r_left"], o["corner_r_right"]), s,
                                  o["arm_len_left"], o["arm_len_right"], o["x_arm_left"], o["x_arm_right"]), 3.0)
    pi = v3_geom.sample(rounded_u(i["x_left"], i["x_right"], i["y_end"], (i["corner_r_left"], i["corner_r_right"]), s,
                                  i["arm_len_left"], i["arm_len_right"], i["x_arm_left"], i["x_arm_right"]), 3.0)
    poly = Polygon(np.vstack([po, pi[::-1]])).buffer(0)
    if poly.geom_type != "Polygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    return poly, po


TOP_T = 0.03
for cn in PLAN["counters"]:
    poly, outer = u_polygon(cn)
    levels = sorted(cn["top_levels_z"])
    top_main = max(z for z in levels if z > 1.05) if any(z > 1.05 for z in levels) else max(levels)
    base = f"{R}/Fixtures/{cn['name']}"
    UsdGeom.Xform.Define(stage, base)
    desk = [z for z in levels if 0.85 <= z <= 0.95]
    if desk and cn["name"] == "U_table_south":                 # reception: low work top inside a 1.13 m upstand
        up_w = 0.20                                            # upstand band (estimated from the 1.133 m rim lines)
        band_poly = poly.intersection(LineString(outer).buffer(up_w, cap_style=2)).buffer(0)
        low_poly = poly.difference(band_poly).buffer(0)
        parts = [(band_poly, top_main), (low_poly, desk[0])]
    else:
        parts = [(poly, top_main)]
    for q, (pg, ztop) in enumerate(parts):
        for gpoly in getattr(pg, "geoms", [pg]):
            if gpoly.area < 0.01:
                continue
            mb = MB()
            mb.prism(gpoly, 0.0, ztop - TOP_T, "counter_front", top=False)
            mb.prism(gpoly.buffer(0.015, join_style=2), ztop - TOP_T, ztop, "counter_top", top=True, bottom=True)
            me = emit(f"{base}/Part{q}_{len(stage.GetPrimAtPath(base).GetChildren())}", mb, "counter", lambda tg: tg)
            # collider: exact convex pieces = triangular prisms of the constrained triangulation of the footprint
            for ti, t_ in enumerate(shapely.constrained_delaunay_triangles(gpoly).geoms):
                tri = np.asarray(t_.exterior.coords)[:3]
                V = np.vstack([np.c_[tri, np.zeros(3)], np.c_[tri, np.full(3, ztop)]]).astype(np.float32)
                F = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]], np.int32)
                cm = UsdGeom.Mesh.Define(stage, f"{R}/Collision/{cn['name']}_p{q}_{len(COLL)}_{ti}")
                cm.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V))
                cm.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
                cm.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.ravel()))
                cm.CreatePurposeAttr(UsdGeom.Tokens.guide)
                UsdPhysics.CollisionAPI.Apply(cm.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(cm.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
                label(cm.GetPrim(), "counter")
for fb in PLAN.get("fixed_blocks", []):
    base = f"{R}/Fixtures/{fb['name']}"
    UsdGeom.Xform.Define(stage, base)
    parts_ = fb.get("parts") or [{"x": fb["x"], "y": fb["y"], "top_z": fb["top_z"]}]
    for q, pt in enumerate(parts_):                             # measured sections (bar _|-- : low base, step, high bar)
        (x0, x1), (y0, y1), tz = pt["x"], pt["y"], pt["top_z"]
        mb = MB()
        mb.box(x0, x1, y0, y1, 0.0, tz - TOP_T, {"edge": "counter_front"})
        mb.box(x0 - 0.015, x1 + 0.015, y0 - (0.015 if q == 0 else 0), y1 + (0.015 if q == len(parts_) - 1 else 0),
               tz - TOP_T, tz, {"edge": "counter_top"})
        emit(f"{base}/Section_{q}", mb, "counter", lambda tg: tg)
        coll_box(f"{fb['name']}_{q}", x0, x1, y0, y1, 0.0, tz, "counter")
    bs = fb.get("backsplash")
    if bs:                                                      # mint glass splash panel on the sink counter
        bx0, bx1, by0, by1 = bs["x"] - bs["t"] / 2, bs["x"] + bs["t"] / 2, bs["y"][0], bs["y"][1]
        mb = MB()
        mb.box(bx0, bx1, by0, by1, bs["z"][0], bs["z"][1], {"edge": bs["finish"]})
        emit(f"{base}/Backsplash", mb, "counter", lambda tg: tg)
        coll_box(f"{fb['name']}_backsplash", bx0, bx1, by0, by1, bs["z"][0], bs["z"][1], "counter")
    sc = fb.get("showcase")
    if sc:                                                      # closed glass showcase standing on the low base
        (x0, x1), (y0, y1), (z0, z1) = sc["x"], sc["y"], sc["z"]
        mb = MB()
        mb.box(x0, x1, y0, y1, z0, z0 + 0.12, {"edge": "counter_front"})
        mb.box(x0, x1, y0, y1, z0 + 0.12, z1, {"edge": "showcase_glass"})
        emit(f"{base}/Showcase", mb, "counter", lambda tg: tg)
        coll_box(f"{fb['name']}_showcase", x0, x1, y0, y1, z0, z1, "counter")

# ------------------------------------------------------------------ floor (per zone) + baseboards
free = rasters["free"]
fp_poly = []
cnts, _ = cv2.findContours(fp_hr.T.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in cnts:
    pts = cv2.approxPolyDP(cnt, 1.5, True)[:, 0, :].astype(float)
    if len(pts) >= 3:
        fp_poly.append(Polygon(np.c_[LO[0] + (pts[:, 0] + 0.5) * RC, LO[1] + (pts[:, 1] + 0.5) * RC]).buffer(0.10, join_style=2))
floor_all = unary_union(fp_poly)
zone_polys = {}
rest = floor_all
for z in ZONES["zones"]:
    if "box" not in z:
        continue
    zp = rest.intersection(sbox(*[z["box"][0], z["box"][2], z["box"][1], z["box"][3]]))
    zone_polys[z["name"]] = zp
    rest = rest.difference(zp)
for k_, nm_ in ROOM_LAB["names"].items():                         # enclosed rooms, grown under their walls
    m_ = (ROOM_LAB["raster"] == k_).T.astype(np.uint8).copy()
    cnts_, _ = cv2.findContours(m_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys_ = []
    for cn_ in cnts_:
        pts_ = cv2.approxPolyDP(cn_, 1.0, True)[:, 0, :].astype(float)
        if len(pts_) >= 3:
            polys_.append(Polygon(np.c_[LO[0] + (pts_[:, 0] + 0.5) * RC, LO[1] + (pts_[:, 1] + 0.5) * RC]).buffer(0))
    zp = rest.intersection(unary_union(polys_))
    zone_polys[nm_] = zp
    rest = rest.difference(zp)
zone_polys[ZONES["default_zone"]] = rest
for zn, zp in zone_polys.items():
    mb = MB()
    for g in getattr(zp, "geoms", [zp]):
        if g.geom_type == "Polygon" and g.area > 0.05:
            mb.cap(g.simplify(0.005), 0.0, True, f"floor_{zn}")
    emit(f"{R}/Floor/{zn}", mb, "floor", finish_of_tag, room=zn)
coll_box("floor", -14.5, 14.5, -23.5, 23.5, -0.1, 0.0, "floor")

BB_H, BB_T = 0.08, 0.012
door_spans = [(d["axis"], d["wall_c"], d["a0"] - CASE_W, d["a1"] + CASE_W) for d in door_specs]
mb = MB()
for (axis, c, t, v0, v1, z0, z1, kind) in all_pieces:
    if z0 > 0.01 or kind in ("exterior_below_sill",):
        continue
    for s in (-1, 1):
        u = c + s * t / 2
        probe = [(u + s * 0.1, a) if axis == "x" else (a, u + s * 0.1) for a in np.linspace(v0, v1, 5)[1:-1]]
        if not any(zone_at(*q) != "outside" for q in probe):
            continue
        segs = [(v0, v1)]
        for (dax, dc, d0, d1) in door_spans:
            if dax == axis and abs(dc - c) < 0.2:
                segs = [r for a, b in segs for r in ((a, min(b, d0)), (max(a, d1), b)) if r[1] - r[0] > 0.02]
        for a, b in segs:
            uu0, uu1 = sorted([u, u + s * BB_T])
            x0, x1, y0, y1 = (uu0, uu1, a, b) if axis == "x" else (a, b, uu0, uu1)
            mb.box(x0, x1, y0, y1, 0.0, BB_H, {"edge": "baseboard"})
emit(f"{R}/Baseboards/Baseboards", mb, "baseboard", lambda tg: tg)

# ------------------------------------------------------------------ ceiling: zones (arc-fitted outlines), coves
def region_polys(mask):
    out = []
    lab_, n_ = ndimage.label(mask)
    for q in range(1, n_ + 1):
        m = lab_ == q
        if m.sum() * C * C < 0.3:
            continue
        outline = v3_geom.mask_contour(m, C, LO, smooth=CR["contour_smooth_px"], step=0.02)
        prims = v3_geom.decompose(outline, tol=CR["fit_tol_m"])
        pts = v3_geom.sample(prims, max_deg=3.0)
        pg = Polygon(pts).buffer(0)
        if pg.geom_type != "Polygon":
            pg = max(pg.geoms, key=lambda g: g.area)
        holes = ndimage.binary_fill_holes(m) & ~m                 # coffers / openings inside this region
        hl, hn = ndimage.label(holes)
        for h_ in range(1, hn + 1):
            hm = hl == h_
            if hm.sum() * C * C < 0.3:
                continue
            hpts = v3_geom.sample(v3_geom.decompose(v3_geom.mask_contour(hm, C, LO, smooth=CR["contour_smooth_px"], step=0.02),
                                                    tol=CR["fit_tol_m"]), max_deg=3.0)
            hp = Polygon(hpts).buffer(0)
            pg = pg.difference(hp.buffer(-0.02, join_style=2))       # 2 cm overlap with the coffer region
        if pg.geom_type != "Polygon":
            pg = max(pg.geoms, key=lambda g: g.area)
        out.append((pg, prims))
    return out


cove = MB()
COVE_SEGS = []
OBST_NEAR = ndimage.binary_dilation(rasters["obst"], iterations=int(round(0.10 / RC)))   # within 10 cm of a wall
ceil_prims = {}
zone_cap_names = []
# core corridor: lowered charcoal panel with LED coves along both long edges (LAS + real-02 t2-9)
CC_ = ZONES.get("core_corridor_coffer")
if CC_:
    (cx0, cx1), (cy0, cy1) = CC_["x"], CC_["y"]
    mb = MB()
    mb.box(cx0, cx1, cy0, cy1, CC_["z_bottom"], CC_["z_top"], {"edge": CC_["finish"]})
    emit(f"{R}/Ceiling/Zones/CoreCorridorPanel", mb, "ceiling", lambda tg: tg, room="corridor_core_S")
    coll_box("core_corridor_panel", cx0, cx1, cy0, cy1, CC_["z_bottom"], CC_["z_top"], "ceiling")
    for xe, sg in ((cx0, -1), (cx1, 1)):
        cove.quad((xe + sg * 0.005, cy0, CC_["z_top"]), (xe + sg * 0.005, cy1, CC_["z_top"]),
                  (xe + sg * 0.005, cy1, CC_["z_top"] + 0.03), (xe + sg * 0.005, cy0, CC_["z_top"] + 0.03), "cove_led")
        COVE_SEGS.append((np.array([xe + sg * 0.04, cy0]), np.array([xe + sg * 0.04, cy1]), CC_["z_top"] + 0.01))
for r in ids:
    h = rh[r]
    for qi, (pg, prims) in enumerate(region_polys(rlab == r)):
        # one prim for the soffit sides of this height region + one prim per finish zone (room label = that zone)
        hcm = int(round(h * 100))
        sides = MB()
        pgb = pg.buffer(0.02, join_style=2)
        sides.prism(pgb, h, H_TOP + 0.01, "soffit_side", top=False, bottom=False)
        caps = {}
        left = pgb
        for zn_, zp_ in zone_polys.items():                      # bottom caps split by finish zone
            piece = pgb.intersection(zp_.buffer(0.2, join_style=2)).intersection(left)
            if piece.is_empty or piece.area < 0.01:
                continue
            left = left.difference(piece)
            fin = ZONES["zone_finish"].get(zn_, {}).get("ceiling", "ceiling_tile")
            if fin == "rule_hall":
                fin = "ceiling_gray" if h <= ZONES["hall_ceiling_rule"]["gray_below_m"] else "ceiling_tile"
            for g_ in getattr(piece, "geoms", [piece]):
                if g_.geom_type == "Polygon" and g_.area > 0.005:
                    caps.setdefault(zn_, MB()).cap(g_, h, False, "ceil:" + fin)
        for g_ in getattr(left, "geoms", [left]):
            if g_.geom_type == "Polygon" and g_.area > 0.005:
                caps.setdefault(zone_at(*g_.representative_point().coords[0]), MB()).cap(
                    g_, h, False, "ceil:" + ("ceiling_gray" if h <= ZONES["hall_ceiling_rule"]["gray_below_m"] else "ceiling_tile"))
        cname = f"Soffit_h{hcm}"
        cname += f"_{sum(k_.startswith(cname) for k_ in ceil_prims)}" if any(k_.startswith(cname) for k_ in ceil_prims) else ""
        emit(f"{R}/Ceiling/Zones/{cname}", sides, "ceiling", lambda tg: tg)
        ceil_prims[cname] = {"height": h, "primitives": prims, "zones": sorted(caps)}
        for zn_, cmb in caps.items():
            nm = f"{zn_}_h{hcm}"
            nm += f"_{sum(k_.startswith(nm) for k_ in zone_cap_names)}" if any(k_.startswith(nm) for k_ in zone_cap_names) else ""
            zone_cap_names.append(nm)
            emit(f"{R}/Ceiling/Zones/{nm}", cmb, "ceiling", lambda tg: tg[5:] if tg.startswith("ceil:") else tg,
                 room=zn_, st_scale=ZONES["finishes"]["ceiling_tile"]["uv_scale_m"][0])
        # cove LED: along edges where the neighbouring ceiling is higher (soffit lip facing a coffer); none on top of
        # a wall / column / glass line (a ceiling step that sits on a wall has no cove, user 2026-10-09)
        og = orient(pg, 1.0)
        segs_ = []
        for rg in [og.exterior, *og.interiors]:                   # right-hand side = outside of the region
            rc_ = np.asarray(rg.coords)
            segs_ += list(zip(rc_[:-1], rc_[1:]))
        for a, b in segs_:
            m_ = (a + b) / 2
            d_ = b - a
            L = np.linalg.norm(d_)
            if L < 1e-3:
                continue
            nrm = np.array([d_[1], -d_[0]]) / L                 # outward for a CCW ring
            q_ = m_ + nrm * 0.15
            hn = ceil_at(*q_)
            iw_, jw_ = int((m_[0] - LO[0]) / RC), int((m_[1] - LO[1]) / RC)
            on_wall = 0 <= iw_ < OBST_NEAR.shape[0] and 0 <= jw_ < OBST_NEAR.shape[1] and bool(OBST_NEAR[iw_, jw_])
            if hn - h > 0.04 and hn < H_TOP - 0.005 and not on_wall:
                p0, p1 = a + nrm * 0.02, b + nrm * 0.02
                cove.quad((p0[0], p0[1], h + 0.005), (p1[0], p1[1], h + 0.005), (p1[0], p1[1], h + 0.035), (p0[0], p0[1], h + 0.035), "cove_led")
                COVE_SEGS.append((p0 + nrm * 0.03, p1 + nrm * 0.03, h + 0.02))
emit(f"{R}/Ceiling/Coves/Coves", cove, "light_panel", lambda tg: tg)
mb = MB()
mb.cap(floor_all, H_TOP + 0.015, False, "ceiling_zone")         # backstop above all zones
emit(f"{R}/Ceiling/Zones/Backstop", mb, "ceiling", lambda tg: "ceiling_plain")
json.dump(ceil_prims, open(CFG / "dx_hub_no_obj_v1_ceiling_zones.json", "w"), indent=1)

# ------------------------------------------------------------------ light panels, HVAC cassettes (measured: 07c_ceiling_lights.py)
CL = json.load(open(CFG / "dx_hub_no_obj_v1_ceiling_lights.json"))
FRAME_W = 0.025                                                  # aluminium frame around the diffuser
lights_out = []
lp_mb, lf_mb, hv_mb, hg_mb = MB(), MB(), MB(), MB()
for pnl in CL["panels"]:
    x, y, sx, sy = pnl["x"], pnl["y"], pnl["sx"], pnl["sy"]
    z = ceil_at(x, y)
    if CC_ and CC_["x"][0] <= x <= CC_["x"][1] and CC_["y"][0] <= y <= CC_["y"][1]:
        z = CC_["z_bottom"]
    x0, x1, y0, y1 = x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2
    for (a0_, a1_, b0_, b1_) in [(x0, x1, y0, y0 + FRAME_W), (x0, x1, y1 - FRAME_W, y1),
                                 (x0, x0 + FRAME_W, y0 + FRAME_W, y1 - FRAME_W), (x1 - FRAME_W, x1, y0 + FRAME_W, y1 - FRAME_W)]:
        lf_mb.box(a0_, a1_, b0_, b1_, z - 0.012, z, {"edge": "light_frame"})
    lp_mb.box(x0 + FRAME_W, x1 - FRAME_W, y0 + FRAME_W, y1 - FRAME_W, z - 0.008, z - 0.002, {"edge": "light_panel"})
    lights_out.append({"x": x, "y": y, "z": z - 0.009, "sx": sx - 2 * FRAME_W, "sy": sy - 2 * FRAME_W, "size_mm": pnl["size_mm"]})
for hv in CL["hvac"]:
    x, y = hv["x"], hv["y"]
    z = ceil_at(x, y)
    if CC_ and CC_["x"][0] <= x <= CC_["x"][1] and CC_["y"][0] <= y <= CC_["y"][1]:
        z = CC_["z_bottom"]
    hv_mb.box(x - 0.42, x + 0.42, y - 0.42, y + 0.42, z - 0.025, z, {"edge": "hvac_grille"})          # face plate
    hg_mb.box(x - 0.28, x + 0.28, y - 0.28, y + 0.28, z - 0.03, z - 0.025, {"edge": "hvac_grille_dark"})  # intake grille
    for (a0_, a1_, b0_, b1_) in [(x - 0.30, x + 0.30, y - 0.40, y - 0.33), (x - 0.30, x + 0.30, y + 0.33, y + 0.40),
                                 (x - 0.40, x - 0.33, y - 0.30, y + 0.30), (x + 0.33, x + 0.40, y - 0.30, y + 0.30)]:
        hg_mb.box(a0_, a1_, b0_, b1_, z - 0.03, z - 0.025, {"edge": "hvac_grille_dark"})            # 4 supply slots
emit(f"{R}/Ceiling/LightPanels/Diffusers", lp_mb, "light_panel", lambda tg: tg)
emit(f"{R}/Ceiling/LightPanels/Frames", lf_mb, "light_panel", lambda tg: tg)
emit(f"{R}/Ceiling/HVAC/Cassettes", hv_mb, "hvac", lambda tg: tg)
emit(f"{R}/Ceiling/HVAC/Grilles", hg_mb, "hvac", lambda tg: tg)
json.dump(lights_out, open(CFG / "dx_hub_no_obj_v1_light_panels.json", "w"), indent=1)
print(f"light panels {len(lights_out)}, hvac {len(CL['hvac'])}")

# ------------------------------------------------------------------ lights (RectLights at the panels)
UsdGeom.Xform.Define(stage, f"{R}/Lights")
UsdGeom.Xform.Define(stage, f"{R}/Lights/Panels")
for k, L in enumerate(lights_out):
    lt = UsdLux.RectLight.Define(stage, f"{R}/Lights/Panels/Panel_{k:03d}")
    lt.CreateWidthAttr(L["sx"])
    lt.CreateHeightAttr(L["sy"])
    lt.CreateIntensityAttr(ZONES["lighting"]["panel_intensity"])
    lt.CreateColorTemperatureAttr(ZONES["lighting"]["panel_cct"])
    lt.CreateEnableColorTemperatureAttr(True)
    lt.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)   # camera sees the diffuser mesh
    UsdGeom.Xformable(lt).AddTranslateOp().Set(Gf.Vec3d(L["x"], L["y"], L["z"]))

# ------------------------------------------------------------------ downlights (measured: 07c_ceiling_lights.py)
def _free_ceiling(x, y):
    q = Point(x, y)
    return floor_all.buffer(-0.15).contains(q) and not occ_r[min(max(int((x - LO[0]) / RC), 0), occ_r.shape[0] - 1),
                                                            min(max(int((y - LO[1]) / RC), 0), occ_r.shape[1] - 1)]


dl_pts = [(d["x"], d["y"]) for d in CL["downlights"] if _free_ceiling(d["x"], d["y"])]
UsdGeom.Xform.Define(stage, f"{R}/Lights/Downlights")
dmb = MB()
for k, (x, y) in enumerate(dl_pts):
    zc_ = ceil_at(x, y)
    if CC_ and CC_["x"][0] <= x <= CC_["x"][1] and CC_["y"][0] <= y <= CC_["y"][1]:
        zc_ = CC_["z_bottom"]
    rr = ZONES["lighting"]["downlight_radius"]
    ang_ = np.linspace(0, 2 * np.pi, 13)
    for a0_, a1_ in zip(ang_[:-1], ang_[1:]):                     # small emissive disc
        dmb.tri((x, y, zc_ - 0.003), (x + rr * np.cos(a1_), y + rr * np.sin(a1_), zc_ - 0.003),
                (x + rr * np.cos(a0_), y + rr * np.sin(a0_), zc_ - 0.003), "light_panel")
    lt = UsdLux.DiskLight.Define(stage, f"{R}/Lights/Downlights/Down_{k:03d}")
    lt.CreateRadiusAttr(rr)
    lt.CreateIntensityAttr(ZONES["lighting"]["downlight_intensity"])
    lt.CreateColorTemperatureAttr(ZONES["lighting"]["downlight_cct"])
    lt.CreateEnableColorTemperatureAttr(True)
    lt.GetPrim().CreateAttribute("visibleInPrimaryRay", Sdf.ValueTypeNames.Bool).Set(False)
    UsdGeom.Xformable(lt).AddTranslateOp().Set(Gf.Vec3d(x, y, zc_ - 0.006))
emit(f"{R}/Ceiling/LightPanels/Downlights", dmb, "light_panel", lambda tg: tg)
print(f"downlights: {len(dl_pts)}")

UsdGeom.Xform.Define(stage, f"{R}/Lights/Coves")
n_cove = 0
# chain consecutive cove segments (arcs are sampled every 3 deg, i.e. a few cm) and light them with RectLights of
# chords that stay within cove_chord_sag_m of the cove line (long on straights, short on arcs), so curved coves glow
# as evenly as straight ones
pieces, run = [], []
for sg_ in COVE_SEGS + [None]:
    if run and (sg_ is None or np.linalg.norm(sg_[0] - run[-1][1]) > 0.06 or abs(sg_[2] - run[-1][2]) > 1e-3):
        pts_ = [run[0][0]] + [r_[1] for r_ in run]
        a_ = 0
        for b_ in range(1, len(pts_)):
            if b_ == len(pts_) - 1:
                pieces.append((pts_[a_], pts_[b_], run[0][2]))
                break
            c0_, c1_ = pts_[a_], pts_[b_ + 1]                    # would the chord to the next point still fit?
            d_ = c1_ - c0_
            n_ = np.array([-d_[1], d_[0]]) / max(np.linalg.norm(d_), 1e-9)
            if max(abs((q_ - c0_) @ n_) for q_ in pts_[a_ + 1:b_ + 1]) > CR["cove_chord_sag_m"]:
                pieces.append((pts_[a_], pts_[b_], run[0][2]))
                a_ = b_
        run = []
    if sg_ is not None:
        run.append(sg_)
for p0, p1, z in pieces:
    L = float(np.linalg.norm(p1 - p0))
    if L < 0.08:
        continue
    mid = (p0 + p1) / 2
    ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
    lt = UsdLux.RectLight.Define(stage, f"{R}/Lights/Coves/Cove_{n_cove:03d}")
    lt.CreateWidthAttr(L)
    lt.CreateHeightAttr(0.03)
    lt.CreateIntensityAttr(ZONES["lighting"]["cove_intensity"])
    lt.CreateColorTemperatureAttr(ZONES["lighting"]["cove_cct"])
    lt.CreateEnableColorTemperatureAttr(True)
    x_ = UsdGeom.Xformable(lt)
    x_.AddTranslateOp().Set(Gf.Vec3d(mid[0], mid[1], z))
    x_.AddRotateZOp().Set(ang)
    x_.AddRotateXOp().Set(180.0)                                 # emit upward onto the coffer ceiling
    n_cove += 1
print(f"cove lights: {n_cove}")

# ------------------------------------------------------------------ collision boxes
for k, (name, ctr, size, ang, cls) in enumerate(COLL):
    cube = UsdGeom.Cube.Define(stage, f"{R}/Collision/{Tf.MakeValidIdentifier(name.replace('-', 'm'))}_{k}")   # m = minus
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*ctr))
    if ang:
        x.AddRotateZOp().Set(float(ang))
    x.AddScaleOp().Set(Gf.Vec3f(*[max(s, 0.002) for s in size]))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    label(cube.GetPrim(), cls)
pm = UsdShade.Material.Define(stage, f"{R}/Looks/PhysicsMaterial")
pma = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
pma.CreateStaticFrictionAttr(0.8)
pma.CreateDynamicFrictionAttr(0.6)
pma.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(f"{R}/Collision")).Bind(pm, UsdShade.Tokens.weakerThanDescendants, "physics")
stage.GetRootLayer().Save()
print(f"collision boxes: {len(COLL)}; wrote {PAY / 'building.usdc'}")
