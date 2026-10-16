"""Stage 3: rebuild the DX Hub as a clean architectural model (scan-to-BIM style).

The Poisson scan mesh follows sensor noise, so walls/ceilings look lumpy. Here the fixed building
elements are re-modelled from the LiDAR measurements as planar geometry:

  floor     one flat polygon at z=0 over the scanned footprint
  walls     2D wall mask (vertical continuity above furniture height) -> polygons -> extruded prisms,
            plus closing walls where the footprint ends without a measured wall
  fixtures  built-in solids (counters, booths, sills): structure with no floor visible beneath,
            extruded to their measured top height
  glass     glass partitions/doors located from the scan (frosted / clear glass + aluminium frames)
  ceiling   piecewise-flat height regions; every region is extruded upward, so steps between levels
            (soffits, coffers) appear automatically; recessed light panels become emissive quads
  objects   the remaining furniture from the scan mesh, with everything coinciding with the shell removed

Writes payloads/structure.usdc, payloads/objects.usdc and WORK/{shell,objects}.ply, WORK/lights.json.
"""
import json

import cv2
import numpy as np
import open3d as o3d
import shapely
import shapely.affinity
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy import ndimage
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from common import GSPLAT, LAS, OUT, WORK, apply, read_gs_ply, read_las

C, ZB, Z0 = 0.05, 0.05, -0.10            # 2D cell size, height bin, lowest bin
LO = np.array([-14.0, -23.0])
NX, NY, NZ = int(28 / C), int(46 / C), int((3.1 - Z0) / ZB)
PAY = OUT / "payloads"
TEX = PAY / "textures"
TEX.mkdir(parents=True, exist_ok=True)
fr = json.load(open(WORK / "frame.json"))
zc = Z0 + ZB * (np.arange(NZ) + 0.5)

# ------------------------------------------------------------------ 1. occupancy + fine ceiling
xyz, _ = read_las(LAS)
P = apply(np.array(fr["T_las_to_world"]), xyz).astype(np.float32)
del xyz
i = np.floor((P[:, 0] - LO[0]) / C).astype(np.int64)
j = np.floor((P[:, 1] - LO[1]) / C).astype(np.int64)
k = np.floor((P[:, 2] - Z0) / ZB).astype(np.int64)
ok = (i >= 0) & (i < NX) & (j >= 0) & (j < NY) & (k >= 0) & (k < NZ)
P, i, j, k = P[ok], i[ok], j[ok], k[ok]
H = np.bincount((i * NY + j) * NZ + k, minlength=NX * NY * NZ).reshape(NX, NY, NZ).astype(np.int32)
occ = H >= 3
hi = np.where(zc > 1.9)[0]
oc = occ[:, :, hi]
ceil_bin = np.where(oc.any(2), zc[hi[0] + oc.shape[2] - 1 - np.argmax(oc[:, :, ::-1], axis=2)], np.nan)
c_at = ceil_bin[i, j]
sel = np.isfinite(c_at) & (np.abs(P[:, 2] - c_at) < 0.06)
idx = i[sel] * NY + j[sel]
s, n = np.bincount(idx, P[sel, 2], NX * NY), np.bincount(idx, minlength=NX * NY)
cz = np.where(n >= 3, s / np.maximum(n, 1), np.nan).reshape(NX, NY)
del P, i, j, k, c_at, sel, idx
print("occupancy + ceiling done")

# ------------------------------------------------------------------ 2. classification
fp = ndimage.binary_opening(np.isfinite(cz), iterations=3)
lab, nl = ndimage.label(fp)
fp = lab == (np.argmax(ndimage.sum(fp, lab, range(1, nl + 1))) + 1)
fp = ndimage.binary_fill_holes(ndimage.binary_closing(fp, iterations=6))

cl = np.where(np.isfinite(cz), cz, np.nanmedian(cz))
ub = (zc[None, None, :] > 1.6) & (zc[None, None, :] < cl[..., None] - 0.12)
covu = (occ & ub).sum(2) / np.maximum(ub.sum(2), 1)
wall = ndimage.binary_closing(covu >= 0.6, iterations=1)
lab, nl = ndimage.label(wall)
wall = np.isin(lab, np.where(ndimage.sum(wall, lab, range(1, nl + 1)) >= 12)[0] + 1)
wall &= ndimage.binary_dilation(fp, iterations=3)
edge = fp & ~ndimage.binary_erosion(fp, iterations=2)
perim = edge & (ndimage.distance_transform_edt(~wall) * C > 0.2)
walls = wall | perim

# Glass walls. LiDAR mostly passes through glass, so these were located by hand from the scan:
# rooms seen only *through* a line (ray pass-through analysis), frame/film returns in elevation
# profiles along the line, and the splat renders. Coordinates are in the world frame (m).
#   kind: "film"    clear glass with a frosted film band (0.45-1.95 m)
#         "frosted" frosted glass
#         "door"    clear glass door (closed; leads to unscanned space)
#         "opening" open doorway (header above, nothing below)
#   top:  glass/door head height; None = up to the local ceiling
GLASS = [
    {"name": "A_RoomPartition", "axis": "x", "c": -0.77,
     "segs": [(-12.45, -5.50, "film", 2.40), (-5.50, -4.40, "opening", 2.40)]},
    {"name": "B_SouthRooms", "axis": "y", "c": -15.195,
     "segs": [(-7.45, -5.00, "frosted", None), (-5.00, -4.05, "door", 2.35), (-4.05, -1.20, "frosted", None),
              (-1.20, -0.15, "door", 2.38), (-0.15, 3.62, "frosted", None)]},
    {"name": "C_CorridorEast", "axis": "x", "c": 5.925,
     "segs": [(-4.35, -1.50, "door", 2.58), (-1.50, -0.35, "frosted", 2.45)]},
]
gx = LO[0] + (np.arange(NX) + 0.5) * C
gy = LO[1] + (np.arange(NY) + 0.5) * C
GX, GY = np.meshgrid(gx, gy, indexing="ij")
glass_mask = np.zeros((NX, NY), bool)
for gw in GLASS:
    n_c, a_c = (GX, GY) if gw["axis"] == "x" else (GY, GX)
    for a0, a1, _, _ in gw["segs"]:
        glass_mask |= (np.abs(n_c - gw["c"]) <= 0.15) & (a_c >= a0 - 0.02) & (a_c <= a1 + 0.02)
walls &= ~glass_mask

floor_seen = (H[:, :, 1] + H[:, :, 2]) >= 2
kmax = np.clip(np.floor((np.minimum(cl - 0.15, 2.2) - Z0) / ZB).astype(int), 0, NZ - 1)
below = occ & (np.arange(NZ)[None, None, :] <= kmax[..., None]) & (zc[None, None, :] > 0.1)
anyb = below.any(2)
top = np.where(anyb, zc[NZ - 1 - np.argmax(below[:, :, ::-1], axis=2)], np.nan)
solid = anyb & ~floor_seen & (top >= 0.35) & ~walls & fp
solid = ndimage.binary_fill_holes(ndimage.binary_closing(ndimage.binary_opening(solid, iterations=1), iterations=2))
lab, nl = ndimage.label(solid)
fixtures = []                                             # (mask, height)
for kk, sl in enumerate(ndimage.find_objects(lab), 1):
    mk = lab == kk
    area = mk.sum() * C * C
    ext = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * C
    if area >= 0.5 and ext >= 1.2:
        fixtures.append((mk & ~walls, float(np.nanmedian(top[mk]))))
fix_any = np.zeros_like(fp)
fix_h = np.zeros(fp.shape)
for mk, h in fixtures:
    fix_any |= mk
    fix_h[mk] = h
print(f"footprint {fp.sum() * C * C:.0f} m2, walls {walls.sum() * C * C:.1f} m2, {len(fixtures)} fixtures")

# ------------------------------------------------------------------ 3. ceiling regions + light panels
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
bigmask = big[rl]
_, nearest = ndimage.distance_transform_edt(~bigmask, return_indices=True)
rlab = rl[tuple(nearest)] * fp
# region heights, then merge neighbours closer than 2.5 cm
ids = np.unique(rlab[rlab > 0])
rh = {r: float(np.nanmedian(np.where(rlab == r, cz, np.nan))) for r in ids}
parent = {r: r for r in ids}


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


for a_, b_ in [(rlab[1:], rlab[:-1]), (rlab[:, 1:], rlab[:, :-1])]:
    pairs = np.unique(np.stack([a_.ravel(), b_.ravel()], 1), axis=0)
    for a, b in pairs:
        if a and b and a != b and abs(rh[a] - rh[b]) < 0.025:
            parent[find(a)] = find(b)
root = np.arange(rlab.max() + 1)
for r in ids:
    root[r] = find(r)
rlab = root[rlab]
ids = np.unique(rlab[rlab > 0])
rh = {r: float(np.nanmedian(np.where(rlab == r, cz, np.nan))) for r in ids}
ceil_map = np.zeros(fp.shape)
for r in ids:
    ceil_map[rlab == r] = rh[r]
H_TOP = max(rh.values()) + 0.02

# Recessed light panels: rectangles that deviate from the surrounding ceiling by > 8 mm
# (background = 1.5 m local median, which follows gentle slopes and ignores panels)
background = ndimage.median_filter(czf, size=31)
near_step = ndimage.binary_dilation(jump | ~fp, iterations=3)
res = np.where(np.isfinite(cz) & (ceil_map > 0) & ~near_step, cz - background, 0.0)
pm = ndimage.binary_opening(np.abs(ndimage.median_filter(res, size=3)) > 0.008, iterations=1)
pl, npl = ndimage.label(pm)
panels = []
for kk, sl in enumerate(ndimage.find_objects(pl), 1):
    mk = (pl[sl] == kk).astype(np.uint8)
    if not (0.12 <= mk.sum() * C * C <= 1.6):
        continue
    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (cu, cv_), (w, h), ang = cv2.minAreaRect(max(cnts, key=len))
    if mk.sum() / max(w * h, 1) < 0.65 or min(w, h) * C < 0.15 or max(w, h) * C > 1.6:
        continue
    if min(ang % 90, 90 - ang % 90) > 5:                  # the building is orthogonal
        continue
    cx, cy = LO[0] + (sl[0].start + cv_ + 0.5) * C, LO[1] + (sl[1].start + cu + 0.5) * C
    # cv2 (u, v) = (col, row) = (world y, world x): a side along image angle a lies along world 90 - a
    panels.append({"x": float(cx), "y": float(cy), "w": float((h + 1) * C), "l": float((w + 1) * C),
                   "ang": float(90.0 - ang)})
print(f"ceiling: {len(ids)} flat regions at {sorted(set(round(v, 2) for v in rh.values()))}, {len(panels)} light panels")


# ------------------------------------------------------------------ 4. polygons and prisms
def world(pts):
    """cv2 contour points (col=j, row=i) -> world xy."""
    pts = np.asarray(pts, float).reshape(-1, 2)
    return np.c_[LO[0] + (pts[:, 1] + 0.5) * C, LO[1] + (pts[:, 0] + 0.5) * C]


def mask_polys(mask, eps=1.0, grow=0.0):
    cnts, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    for a, cnt in enumerate(cnts):
        if hier[a][3] != -1:
            continue
        ext = world(cv2.approxPolyDP(cnt, eps, True))
        if len(ext) < 2:
            continue
        shape = LineString(np.vstack([ext, ext[:1]])).buffer(C / 2, join_style=2)
        if len(ext) >= 3:
            shape = shape.union(Polygon(ext).buffer(0))
        ch = hier[a][2]
        while ch != -1:
            hole = world(cv2.approxPolyDP(cnts[ch], eps, True))
            if len(hole) >= 3:
                shape = shape.difference(Polygon(hole).buffer(0).buffer(-C / 2, join_style=2))
            ch = hier[ch][0]
        if grow:
            shape = shape.buffer(grow, join_style=2)
        shape = shape.simplify(0.01)
        out += [g for g in getattr(shape, "geoms", [shape]) if g.area > 1e-4]
    return out


class MeshBuilder:
    def __init__(self):
        self.V, self.F = [], []
        self.n = 0

    def add(self, V, F):
        self.V.append(np.asarray(V, float))
        self.F.append(np.asarray(F, np.int64) + self.n)
        self.n += len(V)

    def cap(self, poly, z, up=True):
        tris = shapely.constrained_delaunay_triangles(poly)
        for t in tris.geoms:
            p = np.asarray(t.exterior.coords)[:3]
            ccw = (p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1]) - (p[1, 1] - p[0, 1]) * (p[2, 0] - p[0, 0]) > 0
            if ccw != up:
                p = p[::-1]
            self.add(np.c_[p, np.full(3, z)], [[0, 1, 2]])

    def sides(self, poly, z0, z1):
        poly = orient(poly, 1.0)                       # exterior CCW, holes CW -> outward normals
        for ring in [poly.exterior, *poly.interiors]:
            p = np.asarray(ring.coords)[:-1]
            q = np.roll(p, -1, 0)
            m = len(p)
            V = np.vstack([np.c_[p, np.full(m, z0)], np.c_[q, np.full(m, z0)],
                           np.c_[q, np.full(m, z1)], np.c_[p, np.full(m, z1)]])
            a = np.arange(m)
            self.add(V, np.r_[np.c_[a, a + m, a + 2 * m], np.c_[a, a + 2 * m, a + 3 * m]])

    def prism(self, poly, z0, z1, top=True, bottom=False):
        self.sides(poly, z0, z1)
        if top:
            self.cap(poly, z1, up=True)
        if bottom:
            self.cap(poly, z0, up=False)

    def arrays(self):
        if not self.V:
            return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
        return np.vstack(self.V), np.vstack(self.F)


parts = {}
b = MeshBuilder()
for poly in mask_polys(fp, eps=1.0, grow=0.02):
    b.cap(poly, 0.0, up=True)
parts["Floor"] = b.arrays()
b = MeshBuilder()
for poly in mask_polys(walls, eps=1.0):
    b.prism(poly, 0.0, H_TOP, top=True)
parts["Walls"] = b.arrays()
b = MeshBuilder()
for mk, h in fixtures:
    for poly in mask_polys(mk, eps=1.0):
        b.prism(poly, 0.0, h, top=True)
parts["Fixtures"] = b.arrays()
b = MeshBuilder()
for r in ids:
    for poly in mask_polys(rlab == r, eps=1.0, grow=0.02):
        b.prism(poly, rh[r], H_TOP + 0.01, top=False, bottom=True)
for poly in mask_polys(fp, eps=1.0, grow=0.05):          # backstop: hides any crack between regions
    b.cap(poly, H_TOP + 0.01, up=False)
parts["Ceiling"] = b.arrays()
b = MeshBuilder()
lights = []
for pnl in panels:
    i0, j0 = int((pnl["x"] - LO[0]) / C), int((pnl["y"] - LO[1]) / C)
    if not (0 <= i0 < NX and 0 <= j0 < NY) or ceil_map[i0, j0] == 0:
        continue
    z = ceil_map[i0, j0] - 0.005
    rect = shapely.affinity.rotate(shapely.geometry.box(-pnl["l"] / 2, -pnl["w"] / 2, pnl["l"] / 2, pnl["w"] / 2),
                                   pnl["ang"], origin=(0, 0))
    rect = shapely.affinity.translate(rect, pnl["x"], pnl["y"])
    b.cap(rect, z, up=False)
    lights.append({**pnl, "z": float(z)})
parts["LightPanels"] = b.arrays()


def box(bld, axis, c, a0, a1, z0, z1, depth):
    """Axis-aligned box along a wall line: [a0, a1] along it, `depth` across it."""
    if axis == "x":
        r = shapely.geometry.box(c - depth / 2, a0, c + depth / 2, a1)
    else:
        r = shapely.geometry.box(a0, c - depth / 2, a1, c + depth / 2)
    bld.prism(r, z0, z1, top=True, bottom=True)


GLASS_T, FRAME_W, FRAME_D, RAIL_H, MAX_PANE = 0.012, 0.05, 0.06, 0.04, 1.5
gb = {k: MeshBuilder() for k in ["Clear", "Frosted", "Frames"]}
hb = MeshBuilder()                                           # headers above doors/openings (wall material)
for gw in GLASS:
    ax_, c0 = gw["axis"], gw["c"]
    for a0, a1, kind, top in gw["segs"]:
        am = (a0 + a1) / 2
        i0 = int(((c0 if ax_ == "x" else am) - LO[0]) / C)
        j0 = int(((am if ax_ == "x" else c0) - LO[1]) / C)
        local_ceil = float(ceil_map[np.clip(i0, 0, NX - 1), np.clip(j0, 0, NY - 1)]) or H_TOP
        ztop = local_ceil if top is None else top
        if ztop < local_ceil - 0.01:
            box(hb, ax_, c0, a0, a1, ztop, H_TOP, 0.10)
        if kind == "opening":
            continue
        # frame: bottom + top rails, stiles at both ends, intermediate mullions every <= 1.5 m
        box(gb["Frames"], ax_, c0, a0, a1, 0.0, RAIL_H, FRAME_D)
        box(gb["Frames"], ax_, c0, a0, a1, ztop - RAIL_H, ztop, FRAME_D)
        n_p = max(1, int(np.ceil((a1 - a0) / MAX_PANE))) if kind != "door" else 1
        edges = np.linspace(a0, a1, n_p + 1)
        for e in edges:
            box(gb["Frames"], ax_, c0, max(a0, e - FRAME_W / 2), min(a1, e + FRAME_W / 2), RAIL_H, ztop - RAIL_H, FRAME_D)
        for p0, p1 in zip(edges[:-1], edges[1:]):
            g0, g1 = p0 + FRAME_W / 2, p1 - FRAME_W / 2
            if kind == "film":                                # clear / frosted film band / clear
                box(gb["Clear"], ax_, c0, g0, g1, RAIL_H, 0.45, GLASS_T)
                box(gb["Frosted"], ax_, c0, g0, g1, 0.45, 1.95, GLASS_T)
                box(gb["Clear"], ax_, c0, g0, g1, 1.95, ztop - RAIL_H, GLASS_T)
            else:
                box(gb["Frosted" if kind == "frosted" else "Clear"], ax_, c0, g0, g1, RAIL_H, ztop - RAIL_H, GLASS_T)
for k, bld in gb.items():
    parts[f"Glass{k}"] = bld.arrays()
V_h, F_h = hb.arrays()
Vw, Fw = parts["Walls"]
parts["Walls"] = (np.vstack([Vw, V_h]), np.vstack([Fw, F_h + len(Vw)]))
json.dump({"panels": lights, "H_TOP": H_TOP}, open(WORK / "lights.json", "w"), indent=1)
for kname, (V, F) in parts.items():
    print(f"  {kname}: {len(F):,} tris")

# ------------------------------------------------------------------ 5. colours
# Per-Gaussian DC colours are not the composited colour (overlapping semi-transparent splats), so a
# fixed architectural palette matched by eye to the splat renders is used instead (sRGB).
col_floor = np.array([0.74, 0.73, 0.70])      # light grey porcelain tiles
col_wall = np.array([0.88, 0.87, 0.84])       # off-white paint
col_ceil = np.array([0.93, 0.93, 0.92])       # white ceiling board
col_fix = [np.array([0.90, 0.89, 0.87])] * len(fixtures)   # white counters / built-ins


def lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


# procedural 60 cm floor tiles in the measured floor colour (4 x 4 tiles per texture)
rng = np.random.default_rng(7)
T, tile = 1024, 256
tex = np.ones((T, T, 3)) * col_floor
for a in range(4):
    for c2 in range(4):
        tex[a * tile:(a + 1) * tile, c2 * tile:(c2 + 1) * tile] *= 1 + rng.normal(0, 0.025)
tex *= 1 + rng.normal(0, 0.012, (T, T, 1))
g = np.zeros((T, T), bool)
for a in range(5):
    g[max(a * tile - 2, 0):a * tile + 2, :] = True
    g[:, max(a * tile - 2, 0):a * tile + 2] = True
tex[g] *= 0.72
Image.fromarray((np.clip(tex, 0, 1) * 255).astype(np.uint8)).save(TEX / "floor_tiles.png")

# ------------------------------------------------------------------ 6. objects = scan mesh minus shell
scan = o3d.io.read_triangle_mesh(str(WORK / "mesh_visual.ply"))
SV, SF = np.asarray(scan.vertices), np.asarray(scan.triangles)
scan.compute_triangle_normals()
SN = np.asarray(scan.triangle_normals)
cen = SV[SF].mean(1)
ti = np.clip(((cen[:, 0] - LO[0]) / C).astype(int), 0, NX - 1)
tj = np.clip(((cen[:, 1] - LO[1]) / C).astype(int), 0, NY - 1)
walls_d = ndimage.binary_dilation(walls | glass_mask, iterations=2)
fix_d = ndimage.binary_dilation(fix_any, iterations=1)
fix_hd = ndimage.grey_dilation(fix_h, size=3)
fp_d = ndimage.binary_dilation(fp, iterations=1)
ceil_d = ndimage.grey_dilation(ceil_map, size=5)
drop = (~fp_d[ti, tj]) | walls_d[ti, tj] \
    | (fix_d[ti, tj] & (cen[:, 2] < fix_hd[ti, tj] + 0.04)) \
    | (cen[:, 2] < 0.03) | ((cen[:, 2] < 0.08) & (np.abs(SN[:, 2]) > 0.85)) \
    | ((ceil_d[ti, tj] > 0) & (cen[:, 2] > ceil_map[ti, tj] - 0.06)) | (cen[:, 2] > H_TOP - 0.1)
obj = o3d.geometry.TriangleMesh(scan)
obj.remove_triangles_by_mask(drop)
obj.remove_unreferenced_vertices()
tc, tn, _ = obj.cluster_connected_triangles()
tc, tn = np.asarray(tc), np.asarray(tn)
tz = np.asarray(obj.vertices)[np.asarray(obj.triangles)].mean(1)[:, 2]
comp_z = ndimage.median(tz, tc, np.arange(len(tn)))           # median height of each component
# drop small fragments and anything living up at the ceiling (fixture housings, pendant blobs)
obj.remove_triangles_by_mask((tn[tc] < 400) | (np.asarray(comp_z)[tc] > 1.9))
obj.remove_unreferenced_vertices()
col_keep = np.asarray(obj.vertex_colors).copy()
obj = obj.filter_smooth_taubin(number_of_iterations=8)
obj.vertex_colors = o3d.utility.Vector3dVector(col_keep)
obj.compute_vertex_normals()
o3d.io.write_triangle_mesh(str(WORK / "objects.ply"), obj)
print(f"objects: {len(obj.triangles):,} tris kept of {len(SF):,}")

# shell for collision
shell = MeshBuilder()
for kname in ["Floor", "Walls", "Fixtures", "Ceiling", "GlassClear", "GlassFrosted", "GlassFrames"]:
    shell.add(*parts[kname])
SVs, SFs = shell.arrays()
o3d.io.write_triangle_mesh(str(WORK / "shell.ply"),
                           o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(SVs), o3d.utility.Vector3iVector(SFs)))


# ------------------------------------------------------------------ 7. USD payloads
def new_stage(path):
    st = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    return st


def material(st, path, colour, rough=0.8, emissive=None, texture=None):
    m = UsdShade.Material.Define(st, path)
    sh = UsdShade.Shader.Define(st, f"{path}/PreviewSurface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    if texture:
        st_r = UsdShade.Shader.Define(st, f"{path}/STReader")
        st_r.CreateIdAttr("UsdPrimvarReader_float2")
        st_r.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tx = UsdShade.Shader.Define(st, f"{path}/Texture")
        tx.CreateIdAttr("UsdUVTexture")
        tx.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tx.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tx.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tx.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_r.ConnectableAPI(), "result")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tx.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*lin(colour).tolist()))
    if emissive is not None:
        sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    m.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return m


def mesh(st, path, V, F, mat=None, st_uv=None, colours=None):
    me = UsdGeom.Mesh.Define(st, path)
    me.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(V, np.float32)))
    me.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
    me.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(F, np.int32).ravel()))
    me.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*np.min(V, 0).tolist()), Gf.Vec3f(*np.max(V, 0).tolist())]))
    me.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    me.CreateDoubleSidedAttr(True)
    if st_uv is not None:
        pv = UsdGeom.PrimvarsAPI(me).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(st_uv, np.float32)))
    if colours is not None:
        me.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.asarray(colours, np.float32)))
    if mat is not None:
        UsdShade.MaterialBindingAPI.Apply(me.GetPrim()).Bind(mat)
    return me


stg = new_stage(PAY / "structure.usdc")
rootp = UsdGeom.Xform.Define(stg, "/Structure")
stg.SetDefaultPrim(rootp.GetPrim())
L = "/Structure/Looks"
m_floor = material(stg, f"{L}/FloorTiles", col_floor, rough=0.35, texture="./textures/floor_tiles.png")
m_wall = material(stg, f"{L}/WallPaint", col_wall, rough=0.85)
m_ceil = material(stg, f"{L}/CeilingPaint", col_ceil, rough=0.9)
m_panel = material(stg, f"{L}/LightPanel", [1, 1, 1], rough=0.5, emissive=(1.0, 0.98, 0.95))
V, F = parts["Floor"]
mesh(stg, "/Structure/Floor", V, F, m_floor, st_uv=V[:, :2] / 2.4)
mesh(stg, "/Structure/Walls", *parts["Walls"], m_wall)
mesh(stg, "/Structure/Ceiling", *parts["Ceiling"], m_ceil)
if len(parts["LightPanels"][1]):
    mesh(stg, "/Structure/LightPanels", *parts["LightPanels"], m_panel)


def glass_material(st, path, frosting, opacity):
    """OmniGlass (MDL, used by Omniverse RTX) with a UsdPreviewSurface fallback for other renderers."""
    m = UsdShade.Material.Define(st, path)
    mdl = UsdShade.Shader.Define(st, f"{path}/OmniGlass")
    mdl.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    mdl.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    mdl.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.97, 0.99, 0.99))
    mdl.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.5)
    mdl.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(frosting)
    mdl.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    out = mdl.CreateOutput("out", Sdf.ValueTypeNames.Token)
    m.CreateSurfaceOutput("mdl").ConnectToSource(out)
    m.CreateVolumeOutput("mdl").ConnectToSource(out)
    m.CreateDisplacementOutput("mdl").ConnectToSource(out)
    ps = UsdShade.Shader.Define(st, f"{path}/PreviewSurface")
    ps.CreateIdAttr("UsdPreviewSurface")
    ps.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.9, 0.95, 0.96))
    ps.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    ps.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(max(frosting, 0.02))
    ps.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    m.CreateSurfaceOutput().ConnectToSource(ps.ConnectableAPI(), "surface")
    return m


UsdGeom.Xform.Define(stg, "/Structure/Glass")
m_clear = glass_material(stg, f"{L}/GlassClear", frosting=0.0, opacity=0.15)
m_frost = glass_material(stg, f"{L}/GlassFrosted", frosting=0.35, opacity=0.55)
m_frame = UsdShade.Material.Define(stg, f"{L}/FrameAluminium")
fs = UsdShade.Shader.Define(stg, f"{L}/FrameAluminium/PreviewSurface")
fs.CreateIdAttr("UsdPreviewSurface")
fs.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.035, 0.035, 0.04))
fs.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.6)
fs.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
m_frame.CreateSurfaceOutput().ConnectToSource(fs.ConnectableAPI(), "surface")
for k, mt in [("Clear", m_clear), ("Frosted", m_frost), ("Frames", m_frame)]:
    if len(parts[f"Glass{k}"][1]):
        mesh(stg, f"/Structure/Glass/{k}", *parts[f"Glass{k}"], mt)

# The two U-shaped tables in the middle of the hall are named so the building-only export can keep them.
U_TABLE_BOXES = [(-1.0, 4.0, 2.0, 12.5), (-1.0, 4.0, -4.5, 0.5)]   # (xmin, xmax, ymin, ymax)
UsdGeom.Xform.Define(stg, "/Structure/Fixtures")
for n_, ((mk, h), colr) in enumerate(zip(fixtures, col_fix)):
    bb = MeshBuilder()
    for poly in mask_polys(mk, eps=1.0):
        bb.prism(poly, 0.0, h, top=True)
    Vf, Ff = bb.arrays()
    if len(Ff):
        cx_, cy_ = (Vf[:, 0].min() + Vf[:, 0].max()) / 2, (Vf[:, 1].min() + Vf[:, 1].max()) / 2
        is_u = any(x0 <= cx_ <= x1 and y0 <= cy_ <= y1 for x0, x1, y0, y1 in U_TABLE_BOXES)
        nm = f"{'UTable' if is_u else 'Fixture'}_{n_:02d}"
        mesh(stg, f"/Structure/Fixtures/{nm}", Vf, Ff, material(stg, f"{L}/{nm}", colr, rough=0.6))
stg.GetRootLayer().Save()

sto = new_stage(PAY / "objects.usdc")
ro = UsdGeom.Xform.Define(sto, "/Objects")
sto.SetDefaultPrim(ro.GetPrim())
mo = UsdShade.Material.Define(sto, "/Objects/Looks/ScanVertexColor")
sh = UsdShade.Shader.Define(sto, "/Objects/Looks/ScanVertexColor/PreviewSurface")
sh.CreateIdAttr("UsdPreviewSurface")
rd = UsdShade.Shader.Define(sto, "/Objects/Looks/ScanVertexColor/DisplayColorReader")
rd.CreateIdAttr("UsdPrimvarReader_float3")
rd.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rd.ConnectableAPI(), "result")
sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
mo.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
mesh(sto, "/Objects/Furniture", np.asarray(obj.vertices), np.asarray(obj.triangles), mo,
     colours=np.asarray(obj.vertex_colors))
sto.GetRootLayer().Save()
print("wrote structure.usdc / objects.usdc")
