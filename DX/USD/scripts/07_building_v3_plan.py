"""Stage 7a (v3): measure a 2D vector floor plan of the DX Hub from the LiDAR map.

Outputs USD/config/dx_hub_no_obj_v1_plan.json (human-editable) and evidence images in
USD/DX_Hub_Building_no_obj_v1/plan/. Geometry is measured, not traced from a raster:

  1. points between 1.6 m and 2.3 m (above furniture, below the lowest ceiling 2.44 m), 2 cm voxels
  2. per-point 2D normals (PCA), oriented toward the scanner pose at capture time
  3. axis-facing points -> face lines: peaks of the point distribution across the axis, split along the
     line where the points stop (gaps > 0.25 m), kept only if they cover the 1.6-2.3 m band (walls, not
     tall furniture)
  4. faces of opposite orientation 4-45 cm apart = both sides of one wall (measured thickness);
     single faces get a default thickness behind the face (marked "estimated")
  5. corners closed by extending/trimming wall ends onto perpendicular walls, collinear pieces merged,
     pieces < 0.15 m dropped; compact wall clusters become columns
  6. points not explained by axis walls -> RANSAC lines (diagonal walls) and circles (arcs, round columns)
  7. U-shaped counters fitted as two straight arms + a circular arc from their front faces
  8. rooms = free space split at door-sized gaps
"""
import json
import time

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import v3_geom
from common import LAS, OUT, TRAJ, WORK, apply, read_las

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)
CFG = OUT.parent / "config"
FIG = OUT.parent / "DX_Hub_Building_no_obj_v1" / "plan"
CFG.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
fr = json.load(open(WORK / "frame.json"))
OVR = json.load(open(OUT.parent / "config" / "dx_hub_no_obj_v1_overrides.json"))
T = np.array(fr["T_las_to_world"])

Z_LO, Z_HI = 1.6, 2.3                  # wall band: above furniture, below the lowest ceiling (2.44 m)
C = 0.05                               # raster cell for footprint/rooms
LO = np.array([-14.0, -23.0])
NX, NY = int(28 / C), int(46 / C)

# ------------------------------------------------------------------ load + footprint
xyz, rec = read_las(LAS)
P = apply(T, xyz).astype(np.float32)
t_all = np.asarray(rec["t"])
del xyz
traj, trec = read_las(TRAJ)
traj = apply(T, traj)
tt = np.asarray(trec["t"])
o = np.argsort(tt)
tt, traj = tt[o], traj[o]
log(f"loaded {len(P):,} points")

# footprint = observed ceiling, as in v2 (03_structure.py)
ii = np.floor((P[:, 0] - LO[0]) / C).astype(np.int64)
jj = np.floor((P[:, 1] - LO[1]) / C).astype(np.int64)
ok = (ii >= 0) & (ii < NX) & (jj >= 0) & (jj < NY)
ceil_obs = np.zeros((NX, NY), np.int32)
hi = ok & (P[:, 2] > 2.35)
np.add.at(ceil_obs, (ii[hi], jj[hi]), 1)
floor_obs = np.zeros((NX, NY), np.int32)
fl = ok & (np.abs(P[:, 2]) < 0.05)
np.add.at(floor_obs, (ii[fl], jj[fl]), 1)
fp = ndimage.binary_opening(ceil_obs >= 3, iterations=3)
lab, nl = ndimage.label(fp)
fp = lab == (np.argmax(ndimage.sum(fp, lab, range(1, nl + 1))) + 1)
fp = ndimage.binary_fill_holes(ndimage.binary_closing(fp, iterations=6))
fp_near = ndimage.binary_dilation(fp, iterations=12)          # +0.6 m

# ------------------------------------------------------------------ wall band, voxels, normals
# ray subsample for see-through tests (windows / glass): every 25th point at 0.3-2.3 m
rs = np.arange(0, len(P), 25)
rs = rs[(P[rs, 2] > 0.3) & (P[rs, 2] < 2.3)]
RAY_P = P[rs].astype(np.float64)
RAY_S = np.stack([np.interp(t_all[rs], tt, traj[:, k]) for k in range(3)], 1)
# lower band (counter fronts) for fixture fitting
low = ok & (P[:, 2] > 0.3) & (P[:, 2] < 0.9)
LOW = P[low][::2].astype(np.float64)
TOPS = P[ok & (P[:, 2] > 0.9) & (P[:, 2] < 1.3)][::2].astype(np.float64)
LOWTOP = P[ok & (P[:, 2] > 0.8) & (P[:, 2] < 1.3)].astype(np.float64)
LOWTOP_F = P[ok & (P[:, 2] > 0.3) & (P[:, 2] < 0.75)].astype(np.float64)
ALLZ = P[ok & (P[:, 2] > 0.05) & (P[:, 2] < 2.5)][::3].astype(np.float64)

band = ok & (P[:, 2] > Z_LO) & (P[:, 2] < Z_HI)
band[band] = fp_near[ii[band], jj[band]]
S, ts = P[band].astype(np.float64), t_all[band]
del P, t_all
v = np.floor(S / 0.02).astype(np.int64)
v -= v.min(0)
key = (v[:, 0] * (v[:, 1].max() + 1) + v[:, 1]) * (v[:, 2].max() + 1) + v[:, 2]
_, first = np.unique(key, return_index=True)
S, ts = S[first], ts[first]
sensor = np.stack([np.interp(ts, tt, traj[:, k]) for k in range(3)], 1)
log(f"wall band: {len(S):,} voxels")

tree2 = cKDTree(S[:, :2])
nbr = tree2.query_ball_point(S[:, :2], r=0.06, workers=-1, return_sorted=False)
N2 = np.zeros((len(S), 2))
lin = np.zeros(len(S))
for a in range(0, len(S), 200_000):                         # batched 2D PCA
    for q, idx in enumerate(nbr[a:a + 200_000]):
        if len(idx) < 6:
            continue
        X = S[idx, :2] - S[idx, :2].mean(0)
        w, U = np.linalg.eigh(X.T @ X)
        N2[a + q] = U[:, 0]
        lin[a + q] = 1 - w[0] / max(w[1], 1e-12)
del nbr
to_sensor = sensor[:, :2] - S[:, :2]
flip = np.einsum("ij,ij->i", N2, to_sensor) < 0
N2[flip] *= -1
log("normals done")

COS_AX = np.cos(np.radians(10))
is_x = (np.abs(N2[:, 0]) > COS_AX) & (lin > 0.8)            # faces of walls running along Y (plane x = const)
is_y = (np.abs(N2[:, 1]) > COS_AX) & (lin > 0.8)
off_axis = (~is_x) & (~is_y) & (lin > 0.8)


# ------------------------------------------------------------------ face lines
def face_lines(u, vv, z, min_pts=120):
    """Peaks of the across-line coordinate u, split into runs along vv. Returns list of faces."""
    if len(u) == 0:
        return []
    bins = np.arange(u.min() - 0.05, u.max() + 0.06, 0.01)
    h, e = np.histogram(u, bins)
    hs = ndimage.gaussian_filter1d(h.astype(float), 1.0)
    peaks = [k for k in range(1, len(hs) - 1) if hs[k] >= hs[k - 1] and hs[k] > hs[k + 1] and h[max(k - 2, 0):k + 3].sum() >= min_pts]
    centres = e[np.array(peaks, int)] + 0.005 if peaks else np.zeros(0)
    if not len(centres):
        return []
    # assign each point to its nearest peak (within 2 cm)
    d = np.abs(u[:, None] - centres[None, :])
    near = d.argmin(1)
    near[d.min(1) > 0.02] = -1
    faces = []
    for pk in range(len(centres)):
        m = near == pk
        if m.sum() < min_pts:
            continue
        uu, vv_, zz = u[m], vv[m], z[m]
        order = np.argsort(vv_)
        uu, vv_, zz = uu[order], vv_[order], zz[order]
        gaps = np.where(np.diff(vv_) > 0.25)[0]
        for seg in np.split(np.arange(len(vv_)), gaps + 1):
            if len(seg) < 40:
                continue
            v0, v1 = vv_[seg[0]], vv_[seg[-1]]
            if v1 - v0 < 0.15:
                continue
            # vertical coverage of the band, measured in 0.3 m pieces along the face
            zc = np.floor((zz[seg] - Z_LO) / 0.05).astype(int)
            pc = np.floor((vv_[seg] - v0) / 0.3).astype(int)
            occ = np.zeros((pc.max() + 1, int((Z_HI - Z_LO) / 0.05) + 1), bool)
            occ[pc, np.clip(zc, 0, occ.shape[1] - 1)] = True
            cover = occ[:, :-1].mean()
            # straightness: fitted angle must be within 3 deg of the axis
            A = np.c_[vv_[seg], np.ones(len(seg))]
            slope = np.linalg.lstsq(A, uu[seg], rcond=None)[0][0]
            faces.append({"u": float(np.median(uu[seg])), "v0": float(v0), "v1": float(v1), "n": int(len(seg)),
                          "cover": float(cover), "angle_deg": float(np.degrees(np.arctan(slope))),
                          "resid": float(np.median(np.abs(uu[seg] - np.median(uu[seg]))))})
    return faces


faces = []
for axis, m_ax, ui, vi in [("x", is_x, 0, 1), ("y", is_y, 1, 0)]:
    for sgn in (+1, -1):
        m = m_ax & (np.sign(N2[:, ui]) == sgn)
        for f in face_lines(S[m, ui], S[m, vi], S[m, 2]):
            f.update(axis=axis, sign=sgn)
            faces.append(f)
log(f"raw faces: {len(faces)}")
np.savez(WORK / "v3_band.npz", S=S, N2=N2, is_x=is_x, is_y=is_y, off_axis=off_axis, fp=fp,
         floor_obs=floor_obs, ceil_obs=ceil_obs)
json.dump(faces, open(WORK / "v3_faces_raw.json", "w"))

# ------------------------------------------------------------------ faces -> walls
WALL_COVER = 0.6                       # faces must cover >= 60 % of the 1.6-2.3 m band
T_INT, T_EXT = 0.12, 0.25              # default thickness for single-sided walls (estimated)
wall_faces = [f for f in faces if f["cover"] >= WALL_COVER and abs(f["angle_deg"]) <= 3.0]
weak_faces = [f for f in faces if 0.3 <= f["cover"] < WALL_COVER and abs(f["angle_deg"]) <= 3.0]


def merge_collinear(fs, du=0.02, gap=0.30):
    out = []
    for axis in "xy":
        for sgn in (+1, -1):
            grp = sorted([f for f in fs if f["axis"] == axis and f["sign"] == sgn], key=lambda f: (f["u"], f["v0"]))
            used = [False] * len(grp)
            for a, fa in enumerate(grp):
                if used[a]:
                    continue
                cur = dict(fa)
                changed = True
                while changed:
                    changed = False
                    for b, fb in enumerate(grp):
                        if used[b] or b == a or abs(fb["u"] - cur["u"]) > du:
                            continue
                        if fb["v0"] <= cur["v1"] + gap and fb["v1"] >= cur["v0"] - gap:
                            w_a, w_b = cur["v1"] - cur["v0"], fb["v1"] - fb["v0"]
                            cur["u"] = (cur["u"] * w_a + fb["u"] * w_b) / (w_a + w_b)
                            cur["v0"], cur["v1"] = min(cur["v0"], fb["v0"]), max(cur["v1"], fb["v1"])
                            cur["n"] += fb["n"]
                            used[b] = changed = True
                used[a] = True
                out.append(cur)
    return out


wall_faces = merge_collinear(wall_faces)


def outside(x, y):
    i, j = int((x - LO[0]) / C), int((y - LO[1]) / C)
    return not (0 <= i < NX and 0 <= j < NY and fp[i, j])


# Pair opposite faces into two-sided walls. A face with normal +u bounds a wall body on its -u side.
walls = []                              # rect: axis, c (centre across), t (thickness), v0, v1, source
remaining = [dict(f, rest=[(f["v0"], f["v1"])]) for f in wall_faces]
pos = [f for f in remaining if f["sign"] > 0]
neg = [f for f in remaining if f["sign"] < 0]
for fp_ in pos:                         # body on the -u side of fp_, i.e. a matching neg face at smaller u
    for fn in sorted(neg, key=lambda f: fp_["u"] - f["u"]):
        if fn["axis"] != fp_["axis"]:
            continue
        thick = fp_["u"] - fn["u"]
        if not (0.04 <= thick <= 0.45):
            continue
        for (a0, a1) in list(fp_["rest"]):
            for (b0, b1) in list(fn["rest"]):
                o0, o1 = max(a0, b0), min(a1, b1)
                if o1 - o0 < 0.15:
                    continue
                walls.append({"axis": fp_["axis"], "c": (fp_["u"] + fn["u"]) / 2, "t": thick, "v0": o0, "v1": o1,
                              "thickness": "measured"})
                fp_["rest"].remove((a0, a1))
                fp_["rest"] += [r for r in [(a0, o0), (o1, a1)] if r[1] - r[0] >= 0.15]
                fn["rest"].remove((b0, b1))
                fn["rest"] += [r for r in [(b0, o0), (o1, b1)] if r[1] - r[0] >= 0.15]
                break
RAY_D = RAY_P - RAY_S
RAY_L = np.linalg.norm(RAY_D[:, :2], axis=1)


def rays_through(axis, u, a0, a1, beyond=1.5):
    """Rays (sensor -> point) that cross the line {axis = u, other in [a0, a1]} at 0.3-2.3 m height and end
    more than `beyond` metres past it: evidence of windows / glass in that wall."""
    k = 0 if axis == "x" else 1
    den = RAY_D[:, k]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_ = (u - RAY_S[:, k]) / den
    hit = (s_ > 0) & (s_ < 1)
    vcross = RAY_S[:, 1 - k] + s_ * RAY_D[:, 1 - k]
    zcross = RAY_S[:, 2] + s_ * RAY_D[:, 2]
    hit &= (vcross > a0) & (vcross < a1) & (zcross > 0.3) & (zcross < 2.3) & ((1 - s_) * RAY_L > beyond)
    return int(hit.sum())


for f in remaining:                     # single-sided remainders
    for (a0, a1) in f["rest"]:
        if a1 - a0 < 0.15:
            continue
        n_through = rays_through(f["axis"], f["u"], a0, a1)
        t_ = T_EXT if n_through / (a1 - a0) >= 20 else T_INT     # windows => exterior wall
        walls.append({"axis": f["axis"], "c": f["u"] - f["sign"] * t_ / 2, "t": t_, "v0": a0, "v1": a1,
                      "thickness": "estimated_exterior" if t_ == T_EXT else "estimated_interior",
                      "face_u": f["u"], "face_sign": f["sign"]})
log(f"walls before cleanup: {len(walls)}")


def merge_walls(ws, dc=0.03, gap=0.30):
    out, used = [], [False] * len(ws)
    order = sorted(range(len(ws)), key=lambda k: (ws[k]["axis"], ws[k]["c"], ws[k]["v0"]))
    for a in order:
        if used[a]:
            continue
        cur = dict(ws[a])
        used[a] = True
        changed = True
        while changed:
            changed = False
            for b in order:
                wb = ws[b]
                if used[b] or wb["axis"] != cur["axis"]:
                    continue
                # same line: centre lines close, or one wall's footprint contains the other's
                fd = OVR.get("wall_merge", {}).get("max_face_diff_m", 0.08)
                same = abs(wb["c"] - cur["c"]) <= dc or \
                    (abs((wb["c"] - wb["t"] / 2) - (cur["c"] - cur["t"] / 2)) <= fd and
                     abs((wb["c"] + wb["t"] / 2) - (cur["c"] + cur["t"] / 2)) <= fd)
                if same and wb["v0"] <= cur["v1"] + gap and wb["v1"] >= cur["v0"] - gap:
                    lo_ = min(cur["c"] - cur["t"] / 2, wb["c"] - wb["t"] / 2)
                    hi_ = max(cur["c"] + cur["t"] / 2, wb["c"] + wb["t"] / 2)
                    meas = cur["thickness"] == "measured" or wb["thickness"] == "measured"
                    if hi_ - lo_ <= 0.45:
                        cur["c"], cur["t"] = (lo_ + hi_) / 2, hi_ - lo_
                    cur["v0"], cur["v1"] = min(cur["v0"], wb["v0"]), max(cur["v1"], wb["v1"])
                    cur["thickness"] = "measured" if meas else cur["thickness"]
                    used[b] = changed = True
        out.append(cur)
    return out


walls = merge_walls(walls)


def close_corners(ws, reach=0.35):
    """Extend or trim each wall end onto a perpendicular wall within `reach` (far face, so they overlap)."""
    for w in ws:
        for end in ("v0", "v1"):
            ve = w[end]
            best = None
            for p in ws:
                if p is w or p["axis"] == w["axis"]:
                    continue
                # p runs along the other axis: its across-coordinate p["c"] is along w's v axis
                if not (p["v0"] - reach <= w["c"] <= p["v1"] + reach):
                    continue
                d = p["c"] - ve
                if abs(d) <= reach + p["t"] / 2:
                    if best is None or abs(d) < abs(best[0]):
                        best = (d, p)
            if best is None:
                continue
            d, p = best
            direction = -1 if end == "v0" else +1        # outward direction of this end
            w[end] = p["c"] + direction * p["t"] / 2     # meet the far face of p
            # and make p reach w as well
            if w["c"] < p["v0"]:
                p["v0"] = min(p["v0"], w["c"] - w["t"] / 2)
            if w["c"] > p["v1"]:
                p["v1"] = max(p["v1"], w["c"] + w["t"] / 2)
    return ws


walls = close_corners(walls)
walls = merge_walls(walls)
walls = [w for w in walls if w["v1"] - w["v0"] >= 0.15]
log(f"walls after cleanup: {len(walls)}")
json.dump({"walls": walls, "weak_faces": weak_faces}, open(WORK / "v3_walls_tmp.json", "w"), indent=1)

# ------------------------------------------------------------------ columns (compact wall clusters)
RC = 0.025
def raster_rects(rects, shape_lo=LO, res=RC):
    nx, ny = int(28 / res), int(46 / res)
    img = np.zeros((nx, ny), bool)
    for w in rects:
        if w["axis"] == "x":
            x0, x1, y0, y1 = w["c"] - w["t"] / 2, w["c"] + w["t"] / 2, w["v0"], w["v1"]
        else:
            x0, x1, y0, y1 = w["v0"], w["v1"], w["c"] - w["t"] / 2, w["c"] + w["t"] / 2
        i0, i1 = int((x0 - shape_lo[0]) / res), int(np.ceil((x1 - shape_lo[0]) / res))
        j0, j1 = int((y0 - shape_lo[1]) / res), int(np.ceil((y1 - shape_lo[1]) / res))
        img[max(i0, 0):i1, max(j0, 0):j1] = True
    return img


wall_img = raster_rects(walls)
lab, nl = ndimage.label(ndimage.binary_dilation(wall_img, iterations=2))
columns, col_members = [], set()
for k, sl in enumerate(ndimage.find_objects(lab), 1):
    wx, wy = (sl[0].stop - sl[0].start) * RC, (sl[1].stop - sl[1].start) * RC
    if wx <= 1.3 and wy <= 1.3:
        x0, y0 = LO[0] + sl[0].start * RC + 2 * RC, LO[1] + sl[1].start * RC + 2 * RC
        x1, y1 = LO[0] + sl[0].stop * RC - 2 * RC, LO[1] + sl[1].stop * RC - 2 * RC
        if min(x1 - x0, y1 - y0) < 0.25:                     # thin pieces = window frames / scan-edge debris (Q5)
            continue
        columns.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "shape": "rect"})
        for q, w in enumerate(walls):
            cx_ = w["c"] if w["axis"] == "x" else (w["v0"] + w["v1"]) / 2
            cy_ = (w["v0"] + w["v1"]) / 2 if w["axis"] == "x" else w["c"]
            if x0 - 0.05 <= cx_ <= x1 + 0.05 and y0 - 0.05 <= cy_ <= y1 + 0.05:
                col_members.add(q)
walls = [w for q, w in enumerate(walls) if q not in col_members]
log(f"columns: {len(columns)}, walls: {len(walls)}")

# ------------------------------------------------------------------ off-axis lines / arcs (RANSAC)
def rect_dist(Pxy, rects):
    d = np.full(len(Pxy), np.inf)
    for w in rects:
        if w["axis"] == "x":
            dx = np.maximum(np.abs(Pxy[:, 0] - w["c"]) - w["t"] / 2, 0)
            dy = np.maximum(np.maximum(w["v0"] - Pxy[:, 1], Pxy[:, 1] - w["v1"]), 0)
        else:
            dy = np.maximum(np.abs(Pxy[:, 1] - w["c"]) - w["t"] / 2, 0)
            dx = np.maximum(np.maximum(w["v0"] - Pxy[:, 0], Pxy[:, 0] - w["v1"]), 0)
        d = np.minimum(d, np.hypot(dx, dy))
    return d


col_rects = [{"axis": "x", "c": (c["x0"] + c["x1"]) / 2, "t": c["x1"] - c["x0"], "v0": c["y0"], "v1": c["y1"]}
             for c in columns]
resid_pts = S[off_axis & (rect_dist(S[:, :2], walls + col_rects) > 0.08)]
rng = np.random.default_rng(0)


def band_cover(pts):
    zc = np.floor((pts[:, 2] - Z_LO) / 0.05).astype(int)
    return len(np.unique(zc[(zc >= 0) & (zc < int((Z_HI - Z_LO) / 0.05))])) / int((Z_HI - Z_LO) / 0.05)


diag_walls, arcs = [], []
R = resid_pts.copy()
for _ in range(30):                                          # lines
    if len(R) < 200:
        break
    best = None
    for _ in range(400):
        a, b = R[rng.choice(len(R), 2, replace=False), :2]
        d = b - a
        L = np.linalg.norm(d)
        if L < 0.3:
            continue
        nrm = np.array([-d[1], d[0]]) / L
        inl = np.abs((R[:, :2] - a) @ nrm) < 0.015
        if best is None or inl.sum() > best[0].sum():
            best = (inl, a, d / L, nrm)
    inl, a, dirv, nrm = best
    if inl.sum() < 250:
        break
    tpar = (R[inl, :2] - a) @ dirv
    order = np.argsort(tpar)
    runs = np.split(order, np.where(np.diff(tpar[order]) > 0.25)[0] + 1)
    for run in runs:
        pts = R[inl][run]
        if len(pts) < 150:
            continue
        c0 = pts[:, :2].mean(0)
        _, _, Vt_ = np.linalg.svd(pts[:, :2] - c0)
        dv = Vt_[0]
        tp = (pts[:, :2] - c0) @ dv
        ang = np.degrees(np.arctan2(dv[1], dv[0])) % 180
        off = min(ang % 90, 90 - ang % 90)
        if tp.max() - tp.min() >= 0.5 and band_cover(pts) >= WALL_COVER and off > 3:
            diag_walls.append({"p0": (c0 + dv * tp.min()).tolist(), "p1": (c0 + dv * tp.max()).tolist(),
                               "t": T_INT, "thickness": "estimated_interior", "angle_deg": float(ang),
                               "n": int(len(pts))})
    R = R[~inl]
for _ in range(20):                                          # circles (arcs / round columns)
    if len(R) < 200:
        break
    best = None
    for _ in range(600):
        p3 = R[rng.choice(len(R), 3, replace=False), :2]
        A = np.c_[2 * p3, np.ones(3)]
        bb = (p3 ** 2).sum(1)
        try:
            cx_, cy_, k_ = np.linalg.solve(A, bb)
        except np.linalg.LinAlgError:
            continue
        r_ = np.sqrt(k_ + cx_ ** 2 + cy_ ** 2)
        if not (0.15 <= r_ <= 6):
            continue
        inl = np.abs(np.hypot(R[:, 0] - cx_, R[:, 1] - cy_) - r_) < 0.015
        if best is None or inl.sum() > best[0].sum():
            best = (inl, cx_, cy_, r_)
    inl, cx_, cy_, r_ = best
    if inl.sum() < 200:
        break
    pts = R[inl]
    th = np.degrees(np.arctan2(pts[:, 1] - cy_, pts[:, 0] - cx_))
    hcount, _ = np.histogram(th, bins=36, range=(-180, 180))
    span = (hcount > 0).sum() * 10
    if band_cover(pts) >= WALL_COVER and span >= 30:
        arcs.append({"cx": float(cx_), "cy": float(cy_), "r": float(r_), "span_deg": int(span), "n": int(len(pts)),
                     "kind": "round_column" if span >= 270 and r_ < 0.6 else "curved_wall"})
    R = R[~inl]
log(f"off-axis: {len(diag_walls)} straight, {len(arcs)} arcs; {len(R)} unexplained band points")


# ------------------------------------------------------------------ user decisions (overrides)
# 1) diagonal pieces that belong to one continuous wall -> one wall, thickness from the paired faces
for ov in OVR.get("diagonal_full_walls", []):
    grp = [d for d in diag_walls if abs(((d["angle_deg"] - ov["angle_deg"] + 90) % 180) - 90) < 3]
    near = np.array(ov["near"], float)
    dirv = np.array([np.cos(np.radians(ov["angle_deg"])), np.sin(np.radians(ov["angle_deg"]))])
    nrm = np.array([-dirv[1], dirv[0]])
    grp = [d for d in grp if abs((np.add(d["p0"], d["p1"]) / 2 - near) @ nrm) < 0.5]
    if not grp:
        continue
    offs = np.array([(np.add(d["p0"], d["p1"]) / 2 - near) @ nrm for d in grp])
    ts = np.concatenate([[(np.array(d["p0"]) - near) @ dirv, (np.array(d["p1"]) - near) @ dirv] for d in grp])
    o_lo, o_hi = offs.min(), offs.max()
    thick = o_hi - o_lo if 0.04 <= o_hi - o_lo <= 0.45 else T_INT
    mid_off = (o_lo + o_hi) / 2 if 0.04 <= o_hi - o_lo <= 0.45 else offs.mean() - np.sign(offs.mean()) * T_INT / 2
    p0 = near + dirv * ts.min() + nrm * mid_off
    p1 = near + dirv * ts.max() + nrm * mid_off
    diag_walls = [d for d in diag_walls if d not in grp]
    diag_walls.append({"name": ov["name"], "p0": p0.tolist(), "p1": p1.tolist(), "t": float(thick),
                       "thickness": "measured" if 0.04 <= o_hi - o_lo <= 0.45 else "estimated_interior",
                       "angle_deg": ov["angle_deg"], "decision": ov["note"]})
    log(f"override: {ov['name']} merged {len(grp)} pieces -> {np.linalg.norm(p1 - p0):.2f} m, t={thick:.3f}")

# 1b) stubs / columns the user confirmed do not exist
for ov in OVR.get("remove_diagonal_walls", []):
    near = np.array(ov["near"], float)
    diag_walls = [d for d in diag_walls if not (abs(((d["angle_deg"] - ov["angle_deg"] + 90) % 180) - 90) < 3
                                                and np.linalg.norm((np.add(d["p0"], d["p1"]) / 2) - near) < 0.8)]
for ov in OVR.get("remove_columns", []):
    x_, y_ = ov["near"]
    columns = [c for c in columns if not (c["x0"] - 0.3 <= x_ <= c["x1"] + 0.3 and c["y0"] - 0.3 <= y_ <= c["y1"] + 0.3)]

for ov in OVR.get("solid_boxes", []):
    (bx0, bx1), (by0, by1) = ov["x"], ov["y"]
    def inside_box(w):
        x0, x1, y0, y1 = (w["c"] - w["t"] / 2, w["c"] + w["t"] / 2, w["v0"], w["v1"]) if w["axis"] == "x" else \
            (w["v0"], w["v1"], w["c"] - w["t"] / 2, w["c"] + w["t"] / 2)
        return x0 >= bx0 - 0.08 and x1 <= bx1 + 0.08 and y0 >= by0 - 0.08 and y1 <= by1 + 0.08
    walls = [w for w in walls if not inside_box(w)]
    columns.append({"x0": bx0, "x1": bx1, "y0": by0, "y1": by1, "shape": "rect", "name": ov["name"]})

# 2) walls the user confirmed as ordinary interior walls (see-through test was fooled)
for ov in OVR.get("force_interior_walls", []):
    for w in walls:
        if w["axis"] == ov["axis"] and ov["c_range"][0] <= w["c"] <= ov["c_range"][1] \
                and w["v1"] > ov["v_range"][0] and w["v0"] < ov["v_range"][1] and w["thickness"] == "estimated_exterior":
            w["t"] = T_INT
            w["c"] = w["face_u"] - w["face_sign"] * T_INT / 2
            w["thickness"] = "estimated_interior"
            w["decision"] = ov["note"]

# 2b) walls the user confirmed as demolished
for ov in OVR.get("remove_walls", []):
    before = len(walls)
    walls = [w for w in walls if not (w["axis"] == ov["axis"] and ov["c_range"][0] <= w["c"] <= ov["c_range"][1]
                                      and w["v0"] >= ov["v_range"][0] - 0.05 and w["v1"] <= ov["v_range"][1] + 0.05)]
    log(f"override: removed {before - len(walls)} walls ({ov['note'][:40]}...)")

# 3) extra glass lines: exact line position = peak of all points (0.05-2.5 m) across the approximate line
extra_glass = []
for ov in OVR.get("extra_glass_lines", []):
    k = 0 if ov["axis"] == "x" else 1
    if "c" in ov:                                            # line fixed from measured neighbouring wall faces
        c_ = float(ov["c"])
    else:
        m = (np.abs(ALLZ[:, k] - ov["approx_c"]) < 0.4) & (ALLZ[:, 1 - k] > ov["a0"]) & (ALLZ[:, 1 - k] < ov["a1"])
        h_, e_ = np.histogram(ALLZ[m, k], bins=np.arange(ov["approx_c"] - 0.4, ov["approx_c"] + 0.401, 0.005))
        c_ = float(e_[np.argmax(ndimage.gaussian_filter1d(h_.astype(float), 1))] + 0.0025)
    extra_glass.append({"name": ov["name"], "axis": ov["axis"], "c": c_, "a0": ov["a0"], "a1": ov["a1"],
                        "kind": ov["kind"], "decision": ov["note"]})
    log(f"override: glass line {ov['name']} measured at {ov['axis']}={c_:.3f}")

# 4) fixed blocks (closed built-ins): footprint from front faces, top from the dominant height
fixed_blocks = []
for ov in OVR.get("fixed_blocks", []):
    if "parts" in ov:                                        # measured by hand on the LAS (bar_measure)
        fixed_blocks.append({"name": ov["name"], "parts": ov["parts"], "showcase": ov.get("showcase"),
                             "backsplash": ov.get("backsplash"), "decision": ov["note"]})
        continue
    x0, x1, y0, y1 = ov["search_box"]
    q = ALLZ[(ALLZ[:, 0] > x0) & (ALLZ[:, 0] < x1) & (ALLZ[:, 1] > y0) & (ALLZ[:, 1] < y1) & (ALLZ[:, 2] < 1.6)]
    h_, e_ = np.histogram(q[:, 2], bins=np.arange(0.5, 1.6, 0.01))
    top_z = float(e_[np.argmax(h_)] + 0.005)
    body = q[q[:, 2] < top_z + 0.05]
    bx = [float(np.percentile(body[:, 0], 1)), float(np.percentile(body[:, 0], 99))]
    by = [float(np.percentile(body[:, 1], 1)), float(np.percentile(body[:, 1], 99))]
    zmax = float(np.percentile(q[:, 2], 99.5))
    fixed_blocks.append({"name": ov["name"], "x": bx, "y": by, "top_z": top_z, "max_z": zmax, "decision": ov["note"]})
    log(f"override: block {ov['name']} x {bx[0]:.2f}..{bx[1]:.2f} y {by[0]:.2f}..{by[1]:.2f} top {top_z:.2f} (max {zmax:.2f})")

# ------------------------------------------------------------------ counters (outline -> lines + arcs)
def rounded_u(xL, xR, y_end, rc, opening, len_L, len_R, armL=None, armR=None):
    """Primitives of one U face: left arm, corner arc, end line, corner arc, right arm.
    The curved end spans [xL, xR]; arms may sit at a different x (armL/armR) with a short step where the
    curve meets them. `len_*` = arm length measured from `y_end` in the opening direction."""
    s_ = opening
    rcL, rcR = (rc, rc) if np.isscalar(rc) else rc
    rcL, rcR = min(rcL, (xR - xL) / 2), min(rcR, (xR - xL) / 2)
    armL = xL if armL is None else armL
    armR = xR if armR is None else armR
    ycL, ycR = y_end + s_ * rcL, y_end + s_ * rcR
    prims = []
    if len_L > rcL + 0.01:
        prims.append({"type": "line", "p0": [armL, y_end + s_ * len_L], "p1": [armL, ycL]})
    if abs(armL - xL) > 0.01:
        prims.append({"type": "line", "p0": [armL, ycL], "p1": [xL, ycL]})
    a0, a1 = (180.0, 270.0) if s_ > 0 else (180.0, 90.0)
    prims.append({"type": "arc", "cx": xL + rcL, "cy": ycL, "r": rcL, "a0_deg": a0, "a1_deg": a1})
    if xR - xL - rcL - rcR > 1e-3:
        prims.append({"type": "line", "p0": [xL + rcL, y_end], "p1": [xR - rcR, y_end]})
    a0, a1 = (270.0, 360.0) if s_ > 0 else (90.0, 0.0)
    prims.append({"type": "arc", "cx": xR - rcR, "cy": ycR, "r": rcR, "a0_deg": a0, "a1_deg": a1})
    if abs(armR - xR) > 0.01:
        prims.append({"type": "line", "p0": [xR, ycR], "p1": [armR, ycR]})
    if len_R > rcR + 0.01:
        prims.append({"type": "line", "p0": [armR, ycR], "p1": [armR, y_end + s_ * len_R]})
    return prims


def face_x(F, x_lo, x_hi, y_lo, y_hi, pick):
    m = (F[:, 0] > x_lo) & (F[:, 0] < x_hi) & (F[:, 1] > y_lo) & (F[:, 1] < y_hi)
    h_, e_ = np.histogram(F[m, 0], bins=np.arange(x_lo, x_hi, 0.005))
    strong = np.where(h_ >= 0.3 * h_.max())[0]
    k = strong.min() if pick == "min" else strong.max()
    win = (F[m, 0] > e_[k] - 0.015) & (F[m, 0] < e_[k] + 0.02)
    return float(np.median(F[m, 0][win]))


def fit_u_counter(name, box, opening, zones):
    """Measured 'rounded U' counter. Each face (outer/inner) = curved end [x_left, x_right] with corner radius
    + arms at x_arm_left/right. zones: arm windows (x ranges, y range clear of columns), the x window used to
    find the closed ends."""
    x0, x1, y0, y1 = box
    F = LOWTOP_F[(LOWTOP_F[:, 0] > x0) & (LOWTOP_F[:, 0] < x1) & (LOWTOP_F[:, 1] > y0) & (LOWTOP_F[:, 1] < y1), :2]
    (ex0, ex1) = zones["end_x"]
    arm = {"outer": [None, None], "inner": [None, None]}         # None = no straight arm on that side
    for k_, side in enumerate(("left", "right")):
        z_ = zones.get(side)
        if z_ is None:
            continue
        (zx0, zx1), (zy0, zy1) = z_
        arm["outer"][k_] = face_x(F, zx0, zx1, zy0, zy1, "min" if side == "left" else "max")
        arm["inner"][k_] = face_x(F, zx0, zx1, zy0, zy1, "max" if side == "left" else "min")
    ys = F[(F[:, 0] > ex0) & (F[:, 0] < ex1), 1]
    h_, e_ = np.histogram(ys, bins=np.arange(y0, y1, 0.005))
    ctr = e_[:-1] + 0.0025
    strong = np.where(h_ >= 10)[0]                         # absolute: the far outer face can be sparse
    y_end = {"outer": float(ctr[min(strong, key=lambda k: ctr[k] * opening)])}
    dist = (ctr - y_end["outer"]) * opening
    win = (dist > 0.4) & (dist < 1.5)
    y_end["inner"] = float(ctr[win][np.argmax(h_[win])])
    faces = {}
    # pass 1: curved ends (outer first; the inner width follows from the outer one)
    for face in ("outer", "inner"):
        aL, aR = arm[face]
        ye = y_end[face]
        known = [a_ for a_ in (aL, aR) if a_ is not None]
        if face == "outer":
            half = (max(known) - min(known)) / 2 if len(known) == 2 else 1.0
            for _ in range(4):
                band = F[((F[:, 1] - ye) * opening > 0.7 * half) & ((F[:, 1] - ye) * opening < 1.0 * half)]
                sel = band[(band[:, 0] > (aL if aL is not None else x0) - 0.4)
                           & (band[:, 0] < (aR if aR is not None else x1) + 0.4)]
                xL, xR = float(np.percentile(sel[:, 0], 1)), float(np.percentile(sel[:, 0], 99))
                if len(known) == 2:
                    break
                half = (xR - xL) / 2
        else:
            o = faces["outer"]
            half = (o["x_right"] - o["x_left"]) / 2 - (ye - y_end["outer"]) * opening
            band = F[((F[:, 1] - ye) * opening > 0.7 * half) & ((F[:, 1] - ye) * opening < 1.0 * half)]
            sel = band[(band[:, 0] > o["x_left"] + 0.3) & (band[:, 0] < o["x_right"] - 0.3)]
            xL, xR = float(np.percentile(sel[:, 0], 2)), float(np.percentile(sel[:, 0], 98))
        # per-side corner radius (and, for a side without an arm, the face x) by robust fit to the points
        mid = (xL + xR) / 2
        res_side = {}
        for side in ("left", "right"):
            sgn = -1 if side == "left" else 1
            x_fixed = (aL if side == "left" else aR)
            xs_try = [xL if side == "left" else xR] if x_fixed is not None else \
                np.arange(*(sorted([xL if side == "left" else xR, mid + sgn * 0.3])), 0.02)
            zone = F[((F[:, 1] - ye) * opening > -0.05) & ((F[:, 1] - ye) * opening < (xR - xL) / 2 + 0.05)
                     & ((F[:, 0] - mid) * sgn > 0) & ((F[:, 0] - mid) * sgn < (xR - xL) / 2 + 0.08)]
            if len(zone) > 3000:
                zone = zone[rng.choice(len(zone), 3000, replace=False)]
            best = None
            for xs in xs_try:
                for rc in np.arange(0.05, abs(xs - mid) + 0.001, 0.02):
                    xa, xb = (xs, 2 * mid - xs) if side == "left" else (2 * mid - xs, xs)
                    P = v3_geom.sample(rounded_u(xa, xb, ye, rc, opening, 0.0, 0.0), max_deg=2.0)
                    dense = np.vstack([np.linspace(P[q], P[q + 1], 20) for q in range(len(P) - 1)])
                    score = np.mean(np.minimum(cKDTree(dense).query(zone)[0], 0.03))   # robust to clutter
                    if best is None or score < best[0]:
                        best = (score, rc, xs)
            res_side[side] = best
        xL, xR = res_side["left"][2], res_side["right"][2]
        faces[face] = {"x_left": float(xL), "x_right": float(xR), "x_arm_left": float(xL if aL is None else aL),
                       "x_arm_right": float(xR if aR is None else aR), "y_end": ye,
                       "corner_r_left": float(res_side["left"][1]), "corner_r_right": float(res_side["right"][1]),
                       "fit_mad_m": float((res_side["left"][0] + res_side["right"][0]) / 2)}
    # pass 2: arm lengths from the inner faces (free of walls/columns); outer arms end at the same end caps
    inner_len = []
    for k_, side in enumerate(("left", "right")):
        d = faces["inner"]
        L = d[f"corner_r_{side}"]
        if zones.get(side) is not None:
            xf = d["x_arm_left"] if side == "left" else d["x_arm_right"]
            q = F[np.abs(F[:, 0] - xf) < 0.015]
            t_ = np.sort((q[:, 1] - d["y_end"]) * opening)
            t_ = np.r_[L, t_[t_ >= L - 0.05]]
            gaps = np.where(np.diff(t_) > 1.2)[0]            # runs may pass a column (< 1.2 m)
            L = float(t_[gaps[0]] if len(gaps) else t_[-1])
        inner_len.append(L)
    shift = (y_end["inner"] - y_end["outer"]) * opening
    for k_, side in enumerate(("left", "right")):
        faces["inner"][f"arm_len_{side}"] = inner_len[k_]
        faces["outer"][f"arm_len_{side}"] = inner_len[k_] + shift if zones.get(side) is not None \
            else faces["outer"][f"corner_r_{side}"]
    tops = LOWTOP[(LOWTOP[:, 0] > x0) & (LOWTOP[:, 0] < x1) & (LOWTOP[:, 1] > y0) & (LOWTOP[:, 1] < y1)]
    h_, e_ = np.histogram(tops[:, 2], bins=np.arange(0.8, 1.3, 0.005))
    hs = ndimage.gaussian_filter1d(h_.astype(float), 1.5)
    pk = [k for k in range(1, len(hs) - 1) if hs[k] >= hs[k - 1] and hs[k] > hs[k + 1] and hs[k] > 0.25 * hs.max()]
    return {"name": name, "type": "u_counter", "opening": "+y" if opening > 0 else "-y",
            "top_levels_z": [float(e_[k] + 0.0025) for k in pk], "top_thickness": 0.03,
            "outer": faces["outer"], "inner": faces["inner"]}


counters = [
    # zones: per arm (x window, y window clear of columns/walls); None = that side has no straight arm
    fit_u_counter("U_table_north", (-0.9, 3.9, 2.4, 11.9), +1,
                  {"left": ((-0.9, 0.5), (5.7, 8.7)), "right": ((2.5, 3.9), (5.7, 8.7)), "end_x": (1.2, 1.8)}),
    fit_u_counter("U_table_south", (-0.9, 3.9, -3.85, 0.25), -1,
                  {"left": None, "right": ((2.5, 3.9), (-3.6, -2.2)), "end_x": (1.2, 1.8)}),
]
for c_ in counters:
    for f in ("outer", "inner"):
        d = c_[f]
        log(f"counter {c_['name']} {f}: curve x [{d['x_left']:.3f}, {d['x_right']:.3f}] arms x [{d['x_arm_left']:.3f}, "
            f"{d['x_arm_right']:.3f}] end y {d['y_end']:.3f} corner r {d['corner_r_left']:.2f}/{d['corner_r_right']:.2f} arms "
            f"{d['arm_len_left']:.2f}/{d['arm_len_right']:.2f} m, fit {d['fit_mad_m'] * 1000:.1f} mm")
    log(f"   top levels {[round(z, 3) for z in c_['top_levels_z']]}")

# ------------------------------------------------------------------ glass partitions (kept from v2, measured there)
GLASS = [
    {"name": "A_RoomPartition", "axis": "x", "c": -0.77, "a0": -12.45, "a1": -4.40},
    {"name": "B_SouthRooms", "axis": "y", "c": -15.195, "a0": -7.45, "a1": 3.62},
    {"name": "C_CorridorEast", "axis": "x", "c": 5.925, "a0": -4.35, "a1": -0.35},
]
GLASS += extra_glass
glass_rects = [{"axis": g["axis"], "c": g["c"], "t": 0.06, "v0": g["a0"], "v1": g["a1"]} for g in GLASS]


def cut_by_glass(ws):
    """Walls overlapping a glass partition line are cut back (glass replaces them)."""
    out = []
    for w in ws:
        pieces = [(w["v0"], w["v1"])]
        for g in glass_rects:
            if g["axis"] != w["axis"] or abs(g["c"] - w["c"]) > (w["t"] / 2 + 0.12):
                continue
            nxt = []
            for a0, a1 in pieces:
                if a1 <= g["v0"] or a0 >= g["v1"]:
                    nxt.append((a0, a1))
                    continue
                nxt += [r for r in [(a0, g["v0"]), (g["v1"], a1)] if r[1] - r[0] >= 0.15]
            pieces = nxt
        out += [dict(w, v0=a0, v1=a1) for a0, a1 in pieces]
    return out


walls = cut_by_glass(walls)


def regularize_lines(ws, max_diff, max_gap):
    """Adjacent collinear walls whose faces differ by <= max_diff get common faces (length-weighted median,
    measured faces x3), so walls and the doors set into them line up (user request 2026-10-08)."""
    n = len(ws)
    par = list(range(n))

    def fnd(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    lo_ = [w["c"] - w["t"] / 2 for w in ws]
    hi_ = [w["c"] + w["t"] / 2 for w in ws]
    for a in range(n):
        for b in range(a + 1, n):
            wa, wb = ws[a], ws[b]
            if wa["axis"] != wb["axis"]:
                continue
            gap = max(wa["v0"], wb["v0"]) - min(wa["v1"], wb["v1"])
            if gap > max_gap or abs(lo_[a] - lo_[b]) > max_diff or abs(hi_[a] - hi_[b]) > max_diff:
                continue
            par[fnd(a)] = fnd(b)
    groups = {}
    for a in range(n):
        groups.setdefault(fnd(a), []).append(a)
    changed = 0
    for g in groups.values():
        if len(g) < 2:
            continue
        w8 = np.array([(ws[a]["v1"] - ws[a]["v0"]) * (3 if ws[a]["thickness"] == "measured" else 1) for a in g])
        def wmed(v):
            o = np.argsort(v)
            cw = np.cumsum(w8[o])
            return float(np.asarray(v)[o][np.searchsorted(cw, cw[-1] / 2)])
        L, H = wmed([lo_[a] for a in g]), wmed([hi_[a] for a in g])
        for a in g:
            ws[a]["c"], ws[a]["t"] = (L + H) / 2, H - L
            ws[a]["regularized"] = True
            changed += 1
    return ws, changed


RG = OVR.get("line_regularization", {})
if RG:
    walls, n_reg = regularize_lines(walls, RG["max_face_diff_m"], RG["max_gap_m"])
    log(f"line regularization: {n_reg} walls aligned")

# ------------------------------------------------------------------ rooms: free space split at door-sized gaps
obst = raster_rects(walls + col_rects + glass_rects)
for dw in diag_walls:
    p0, p1 = np.array(dw["p0"]), np.array(dw["p1"])
    for t_ in np.linspace(0, 1, int(np.linalg.norm(p1 - p0) / 0.01) + 1):
        q = p0 + t_ * (p1 - p0)
        i_, j_ = int((q[0] - LO[0]) / RC), int((q[1] - LO[1]) / RC)
        obst[max(i_ - 2, 0):i_ + 3, max(j_ - 2, 0):j_ + 3] = True
fp_hr = ndimage.zoom(fp.astype(np.uint8), C / RC, order=0).astype(bool)[:obst.shape[0], :obst.shape[1]]
free = fp_hr & ~obst
closed = ~ndimage.binary_dilation(~free, iterations=int(0.55 / RC))      # closes gaps < 1.1 m (doors)
seeds, nseed = ndimage.label(closed)
sizes = ndimage.sum(closed, seeds, range(1, nseed + 1)) * RC * RC
keep_ids = np.where(sizes >= 2.0)[0] + 1
seeds = np.where(np.isin(seeds, keep_ids), seeds, 0)
_, (ni, nj) = ndimage.distance_transform_edt(seeds == 0, return_indices=True)
room_lab = np.where(free, seeds[ni, nj], 0)
rooms = []
for rid in keep_ids:
    m = room_lab == rid
    cnts, _ = cv2.findContours(m.T.astype(np.uint8).copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    c = cv2.approxPolyDP(c, 2, True)[:, 0, :].astype(float)
    poly = np.c_[LO[0] + (c[:, 0] + 0.5) * RC, LO[1] + (c[:, 1] + 0.5) * RC]
    ij = np.argwhere(m)
    cx_, cy_ = LO + (ij.mean(0) + 0.5) * RC
    rooms.append({"id": f"R{len(rooms) + 1:02d}", "name": "", "area_m2": float(m.sum() * RC * RC),
                  "centroid": [float(cx_), float(cy_)], "polygon": poly.round(3).tolist()})
log(f"rooms: {len(rooms)}")

# ------------------------------------------------------------------ metrics
def face_dist(Pxy, rects):
    """Distance of points to the nearest wall *face* (rectangle boundary)."""
    d = np.full(len(Pxy), np.inf)
    for w in rects:
        if w["axis"] == "x":
            along = np.clip(Pxy[:, 1], w["v0"], w["v1"])
            d_face = np.minimum(np.abs(Pxy[:, 0] - (w["c"] - w["t"] / 2)), np.abs(Pxy[:, 0] - (w["c"] + w["t"] / 2)))
            d = np.minimum(d, np.hypot(d_face, Pxy[:, 1] - along))
        else:
            along = np.clip(Pxy[:, 0], w["v0"], w["v1"])
            d_face = np.minimum(np.abs(Pxy[:, 1] - (w["c"] - w["t"] / 2)), np.abs(Pxy[:, 1] - (w["c"] + w["t"] / 2)))
            d = np.minimum(d, np.hypot(d_face, Pxy[:, 0] - along))
    return d


axis_pts = S[(is_x | is_y)]
dfa = face_dist(axis_pts[:, :2], walls + col_rects)
on_wall = dfa < 0.10
meas = [w for w in walls if w["thickness"] == "measured"]
metrics = {
    "wall_face_to_LAS_median_m": float(np.median(dfa[on_wall])),
    "wall_face_to_LAS_p95_m": float(np.percentile(dfa[on_wall], 95)),
    "wall_band_points_explained_frac": float(on_wall.mean()),
    "n_walls": len(walls), "n_walls_measured_thickness": len(meas), "n_columns": len(columns),
    "n_diagonal_walls": len(diag_walls), "n_arcs": len(arcs),
    "n_wall_pieces_shorter_than_0.15m": int(sum(w["v1"] - w["v0"] < 0.15 for w in walls)),
    "off_axis_walls": [f"{d['angle_deg']:.1f} deg, {np.linalg.norm(np.subtract(d['p1'], d['p0'])):.2f} m" for d in diag_walls],
}
# 2D free-space IoU: plan free space vs measured free space (observed footprint minus LAS wall-band occupancy)
occ_las = np.zeros_like(obst)
ii_, jj_ = ((S[:, 0] - LO[0]) / RC).astype(int), ((S[:, 1] - LO[1]) / RC).astype(int)
okk = (ii_ >= 0) & (ii_ < obst.shape[0]) & (jj_ >= 0) & (jj_ < obst.shape[1])
cnt = np.zeros(obst.shape, np.int32)
np.add.at(cnt, (ii_[okk], jj_[okk]), 1)
occ_las = ndimage.binary_closing(cnt >= 4, iterations=1)
meas_free = fp_hr & ~occ_las
inter = (free & meas_free).sum()
metrics["free_space_IoU"] = float(inter / (free | meas_free).sum())
log("metrics: " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}))

# ------------------------------------------------------------------ write plan JSON
plan = {
    "_doc": "DX Hub v3 measured floor plan (world frame: Z-up, metres, floor z=0). Edit values and rerun 07b/08 to "
            "regenerate. Walls are axis-aligned rectangles: axis 'x' = wall runs along Y at x=c; v0..v1 = extent "
            "along the wall; t = thickness ('measured' from both faces, else estimated default).",
    "frame": {"T_las_to_world": fr["T_las_to_world"], "T_gsplat_to_world": fr["T_gsplat_to_world"]},
    "params": {"wall_band_z": [Z_LO, Z_HI], "wall_cover_min": WALL_COVER, "default_thickness_interior": T_INT,
               "default_thickness_exterior": T_EXT, "exterior_rule": ">= 20 rays/m pass through the face (windows)"},
    "walls": [{"id": f"W{k:03d}", **{q: (round(v, 4) if isinstance(v, float) else v) for q, v in w.items()}}
              for k, w in enumerate(sorted(walls, key=lambda w: (w["axis"], w["c"], w["v0"])))],
    "diagonal_walls": diag_walls,
    "columns": [{"id": f"C{k:02d}", **{q: (round(v, 4) if isinstance(v, float) else v) for q, v in c.items()}}
                for k, c in enumerate(columns)],
    "arcs": arcs,
    "glass_partitions": GLASS,
    "counters": counters,
    "fixed_blocks": fixed_blocks,
    "rooms": rooms,
    "weak_faces_glass_or_partial_candidates": [{q: (round(v, 4) if isinstance(v, float) else v) for q, v in f.items()}
                                                for f in weak_faces],
    "metrics": metrics,
}
json.dump(plan, open(CFG / "dx_hub_no_obj_v1_plan.json", "w"), indent=1)
np.savez(WORK / "v3_rasters.npz", obst=obst, free=free, meas_free=meas_free, room_lab=room_lab, fp_hr=fp_hr)
log(f"wrote {CFG / 'dx_hub_no_obj_v1_plan.json'}")
