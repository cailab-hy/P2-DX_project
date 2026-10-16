"""Stage 9a (v3): geometric validation of the built USD against the LiDAR (no Isaac needed).

  * wall faces vs LiDAR wall-band points (1.6-2.3 m): median / p95 distance
  * 2D free-space IoU: model free space (footprint minus walls/columns/glass/closed doors) vs measured free space
  * off-axis walls (must be measured diagonals only) and wall pieces shorter than 0.15 m
  * door clear widths (opening minus jamb linings)
Writes DX_Hub_Building_no_obj_v1/validation/geometry_metrics.json
"""
import json

import numpy as np
from pxr import Usd, UsdGeom, UsdSemantics
from scipy import ndimage
from scipy.spatial import cKDTree

from common import LAS, OUT, WORK, apply, read_las

DST = OUT.parent / "DX_Hub_Building_no_obj_v1"
(DST / "validation").mkdir(exist_ok=True)
CFG = OUT.parent / "config"
PLAN = json.load(open(CFG / "dx_hub_no_obj_v1_plan.json"))
st = Usd.Stage.Open(str(DST / "payloads" / "building.usdc"))
R = "/DX_Hub_Building"


def cls_of(p):
    a = UsdSemantics.LabelsAPI(p, "class").GetLabelsAttr()
    return list(a.Get())[0] if a and a.Get() else None


# collision boxes carry the solid geometry in a simple form (centre, size, rotZ)
boxes = []
for p in st.GetPrimAtPath(f"{R}/Collision").GetChildren():
    if p.GetTypeName() != "Cube":
        continue
    xf = UsdGeom.Xformable(p)
    t = s = None
    rz = 0.0
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t = np.array(op.Get())
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            s = np.array(op.Get())
        elif op.GetOpType() == UsdGeom.XformOp.TypeRotateZ:
            rz = float(op.Get())
    boxes.append({"name": p.GetName(), "cls": cls_of(p), "c": t, "s": s, "rz": rz})
# solids that cross the measured band (1.6-2.3 m): the same height as the LiDAR reference
KINDS = ("wall", "column", "glass_partition", "window_glass", "door", "window_sill")
lo_z = lambda b: b["c"][2] - b["s"][2] / 2
hi_z = lambda b: b["c"][2] + b["s"][2] / 2
touch = [b for b in boxes if b["cls"] in KINDS and lo_z(b) < 2.3 and hi_z(b) > 1.6]     # for face distances
solid = [b for b in boxes if b["cls"] in KINDS and lo_z(b) < 1.6 and hi_z(b) > 2.3]     # for free space

xyz, _ = read_las(LAS)
P = apply(np.array(PLAN["frame"]["T_las_to_world"]), xyz[::4]).astype(np.float64)
del xyz
band = P[(P[:, 2] > 1.6) & (P[:, 2] < 2.3)]


def box_surface_samples(b, step=0.02):
    """Points on the vertical faces of a (possibly rotated) box, in the 1.6-2.3 m band."""
    sx, sy, _ = b["s"]
    cx, cy, _ = b["c"]
    a = np.radians(b["rz"])
    R2 = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    pts = []
    for (u0, v0, u1, v1) in [(-sx / 2, -sy / 2, sx / 2, -sy / 2), (sx / 2, -sy / 2, sx / 2, sy / 2),
                             (sx / 2, sy / 2, -sx / 2, sy / 2), (-sx / 2, sy / 2, -sx / 2, -sy / 2)]:
        L = np.hypot(u1 - u0, v1 - v0)
        n = max(2, int(L / step))
        q = np.c_[np.linspace(u0, u1, n), np.linspace(v0, v1, n)]
        pts.append(q @ R2.T + [cx, cy])
    return np.vstack(pts)


OPN = json.load(open(CFG / "dx_hub_no_obj_v1_openings.json"))
in_window = np.zeros(len(band), bool)                       # window bays are measured separately (blind plane)
win_d = []
for w in OPN["windows"]:
    k = 0 if w["axis"] == "x" else 1
    m = (np.abs(band[:, k] - w["glass_plane_u"]) < 0.3) & (band[:, 1 - k] > w["a0"]) & (band[:, 1 - k] < w["a1"])
    in_window |= m
    win_d.append(np.abs(band[m, k] - w["glass_plane_u"]))
wall_like = [b for b in touch if b["cls"] in ("wall", "column", "door", "glass_partition")]
surf = np.vstack([box_surface_samples(b) for b in wall_like])
tree = cKDTree(surf)
# LiDAR wall-band points that belong to walls: within 10 cm of the model (furniture / clutter excluded)
d, _ = tree.query(band[~in_window, :2], distance_upper_bound=0.10, workers=-1)
on = np.isfinite(d)
wd = np.concatenate(win_d)
wd = wd[wd < 0.10]
metrics = {"wall_face_to_LAS_median_m": float(np.median(d[on])), "wall_face_to_LAS_p95_m": float(np.percentile(d[on], 95)),
           "wall_band_points_within_10cm_frac": float(on.mean()),
           "window_blind_plane_to_LAS_median_m": float(np.median(wd)), "window_blind_plane_to_LAS_p95_m": float(np.percentile(wd, 95))}

# free-space IoU (2.5 cm raster)
RC, LO = 0.025, np.array([-14.0, -23.0])
ras = np.load(WORK / "v3_rasters.npz")
fp, meas_free = ras["fp_hr"], ras["meas_free"]
model_occ = np.zeros_like(fp)
for b in solid:
    hull = box_surface_samples(b, step=0.01)
    lo_ = np.floor((hull.min(0) - LO) / RC).astype(int)
    hi_ = np.ceil((hull.max(0) - LO) / RC).astype(int)
    if b["rz"] == 0.0:
        model_occ[max(lo_[0], 0):hi_[0], max(lo_[1], 0):hi_[1]] = True
    else:
        ij = np.floor((hull - LO) / RC).astype(int)
        tmp = np.zeros_like(fp)
        tmp[np.clip(ij[:, 0], 0, fp.shape[0] - 1), np.clip(ij[:, 1], 0, fp.shape[1] - 1)] = True
        model_occ |= ndimage.binary_fill_holes(ndimage.binary_dilation(tmp, iterations=1))
model_free = fp & ~model_occ
metrics["free_space_IoU"] = float((model_free & meas_free).sum() / (model_free | meas_free).sum())

# off-axis walls and short pieces
metrics["off_axis_walls"] = [f"{b['name']}: {b['rz']:.1f} deg, {b['s'][0]:.2f} m" for b in boxes
                             if b["cls"] == "wall" and abs(((b["rz"] + 45) % 90) - 45) > 0.01]
short = []
for p in st.GetPrimAtPath(f"{R}/Walls").GetChildren():
    V = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
    ext = V.max(0) - V.min(0)
    along = max(ext[0], ext[1])
    if along < 0.15 and not p.GetName().startswith(("jamb_fill", "header")):
        short.append(f"{p.GetName()} ({along * 100:.1f} cm)")
metrics["wall_pieces_shorter_than_0.15m"] = short

# door clear widths
widths = []
for d_ in st.GetPrimAtPath(f"{R}/Openings/Doors").GetChildren():
    w = d_.GetCustomDataByKey("width")
    widths.append({"door": d_.GetName(), "opening_m": round(w, 3), "clear_m": round(w - 2 * 0.015, 3),
                   "height_m": round(d_.GetCustomDataByKey("height"), 3)})
for g in st.GetPrimAtPath(f"{R}/Glass").GetChildren():
    for p in g.GetChildren():
        if p.GetName().startswith("Door_"):
            V = np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())
            ext = V.max(0) - V.min(0)
            widths.append({"door": f"{g.GetName()}/{p.GetName()}", "opening_m": round(float(max(ext[0], ext[1])) + 0.05, 3),
                           "clear_m": round(float(max(ext[0], ext[1])), 3), "height_m": round(float(V[:, 2].max()), 3),
                           "type": p.GetCustomDataByKey("door_type")})
metrics["door_clear_widths"] = widths
json.dump(metrics, open(DST / "validation" / "geometry_metrics.json", "w"), indent=1)
print(json.dumps({k: v for k, v in metrics.items() if k != "door_clear_widths"}, indent=1))
print("doors:", [(w["door"], w["clear_m"]) for w in widths])
