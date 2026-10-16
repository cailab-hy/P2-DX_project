"""Stage 7b (v3): measure openings (doors, windows, glass panels) on every wall face of the v3 plan.

For each face line (collinear wall faces merged across gaps <= 3 m) an elevation image is built from the LiDAR
points lying on the face plane (+-2.5 cm): along-line x height, 2 cm cells. Openings are empty rectangles:

  door / passage : empty from the floor up to a head (1.9-2.5 m) with wall above (header), width 0.6-2.6 m
  window         : wall below (sill 0.4-1.3 m), empty up to a head, light passes through (rays cross it)

A door's state is read behind the face: a planar leaf 3-30 cm behind it = closed (recess measured); rays
passing through = open / open passage. Doors leading to unscanned space are flagged always_closed (user
decision 2026-10-07). Glass lines (A-E) are split into panels at their vertical frames.

Writes USD/config/dx_hub_no_obj_v1_openings.json and figures in USD/DX_Hub_Building_no_obj_v1/openings/.
"""
import json
import time

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import LAS, OUT, TRAJ, WORK, apply, read_las  # noqa: E402

T0 = time.time()
log = lambda m: print(f"[{time.time() - T0:6.1f}s] {m}", flush=True)
CFG = OUT.parent / "config"
FIG = OUT.parent / "DX_Hub_Building_no_obj_v1" / "openings"
FIG.mkdir(parents=True, exist_ok=True)
PLAN = json.load(open(CFG / "dx_hub_no_obj_v1_plan.json"))
OVR = json.load(open(CFG / "dx_hub_no_obj_v1_overrides.json"))
T = np.array(PLAN["frame"]["T_las_to_world"])
G = 0.02                                   # elevation cell
ZMAX = 2.8

xyz, rec = read_las(LAS)
P = apply(T, xyz).astype(np.float32)
t_all = np.asarray(rec["t"])
del xyz
traj, trec = read_las(TRAJ)
traj = apply(T, traj)
tt = np.asarray(trec["t"])
o = np.argsort(tt)
tt, traj = tt[o], traj[o]
keep = (P[:, 2] > -0.05) & (P[:, 2] < ZMAX) & (P[:, 0] > -15) & (P[:, 0] < 15) & (P[:, 1] > -24) & (P[:, 1] < 24)
P, t_all = P[keep], t_all[keep]
rs = np.arange(0, len(P), 10)
RAY_P = P[rs].astype(np.float64)
RAY_S = np.stack([np.interp(t_all[rs], tt, traj[:, k]) for k in range(3)], 1)
RAY_D = RAY_P - RAY_S
del t_all
log(f"points {len(P):,}, rays {len(RAY_P):,}")

# footprint (scanned space) for "leads to unscanned space"
fp = np.load(WORK / "v3_rasters.npz")["fp_hr"]
RC, LO = 0.025, np.array([-14.0, -23.0])


def scanned(x, y):
    i, j = int((x - LO[0]) / RC), int((y - LO[1]) / RC)
    return 0 <= i < fp.shape[0] and 0 <= j < fp.shape[1] and bool(fp[i, j])


# ------------------------------------------------------------------ face lines
faces = []
for w in PLAN["walls"]:
    if w["thickness"] == "measured":
        faces += [(w["axis"], w["c"] - w["t"] / 2, -1, w["v0"], w["v1"], w["id"]),
                  (w["axis"], w["c"] + w["t"] / 2, +1, w["v0"], w["v1"], w["id"])]
    else:
        faces.append((w["axis"], w["face_u"], w["face_sign"], w["v0"], w["v1"], w["id"]))
lines = []
for axis in "xy":
    for sgn in (-1, 1):
        grp = sorted([f for f in faces if f[0] == axis and f[2] == sgn], key=lambda f: (f[1], f[3]))
        used = [False] * len(grp)
        for a, f in enumerate(grp):
            if used[a]:
                continue
            cur = {"axis": axis, "sign": sgn, "u": f[1], "v0": f[3], "v1": f[4], "walls": [f[5]]}
            used[a] = True
            changed = True
            while changed:
                changed = False
                for b, g in enumerate(grp):
                    if used[b] or abs(g[1] - cur["u"]) > 0.04:
                        continue
                    if g[3] <= cur["v1"] + 3.0 and g[4] >= cur["v0"] - 3.0:
                        cur["v0"], cur["v1"] = min(cur["v0"], g[3]), max(cur["v1"], g[4])
                        cur["walls"].append(g[5])
                        used[b] = changed = True
            if cur["v1"] - cur["v0"] >= 0.6:
                lines.append(cur)
log(f"face lines: {len(lines)}")


SUB = {"P": P}


def use_subset(axis, u, v0, v1, reach=0.7):
    """Restrict the point set to a slab around one line (speeds up the many elevation queries)."""
    k = 0 if axis == "x" else 1
    m = (np.abs(P[:, k] - u) < reach) & (P[:, 1 - k] > v0 - 0.5) & (P[:, 1 - k] < v1 + 0.5)
    SUB["P"] = P[m]


def elevation(axis, u, v0, v1, half=0.025, offset=0.0):
    """Occupancy image (along x z) of points within +-half of the plane u+offset."""
    k = 0 if axis == "x" else 1
    Q = SUB["P"]
    m = (np.abs(Q[:, k] - (u + offset)) < half) & (Q[:, 1 - k] > v0) & (Q[:, 1 - k] < v1)
    na, nz = int(np.ceil((v1 - v0) / G)), int(ZMAX / G)
    img = np.zeros((na, nz), np.int32)
    ia = ((Q[m, 1 - k] - v0) / G).astype(int)
    iz = np.clip((Q[m, 2] / G).astype(int), 0, nz - 1)
    np.add.at(img, (np.clip(ia, 0, na - 1), iz), 1)
    return img


def depth_map(axis, u, v0, v1, half=0.10):
    """Mean depth offset (m) of points within +-half of the face, per (along, z) cell; NaN where empty."""
    k = 0 if axis == "x" else 1
    Q = SUB["P"]
    m = (np.abs(Q[:, k] - u) < half) & (Q[:, 1 - k] > v0) & (Q[:, 1 - k] < v1)
    na, nz = int(np.ceil((v1 - v0) / G)), int(ZMAX / G)
    s_ = np.zeros((na, nz))
    n_ = np.zeros((na, nz))
    ia = np.clip(((Q[m, 1 - k] - v0) / G).astype(int), 0, na - 1)
    iz = np.clip((Q[m, 2] / G).astype(int), 0, nz - 1)
    np.add.at(s_, (ia, iz), Q[m, k] - u)
    np.add.at(n_, (ia, iz), 1)
    return np.where(n_ >= 2, s_ / np.maximum(n_, 1), np.nan)


def flush_doors(D, v0, min_step=0.005):
    """Door leaves almost flush with the wall. The depth profile (median offset over 0.3-1.8 m) is cut at
    sharp steps (>= min_step within 8 cm); a segment 0.65-2.2 m wide whose depth differs from the wall just
    outside it, and whose difference stops at a head height of 1.9-2.45 m (header above), is a door.
    Adjacent leaves (double doors with slightly different depths) are merged. Returns [(a0, a1, head, dev)]."""
    zs = slice(int(0.3 / G), int(1.8 / G))
    with np.errstate(all="ignore"):
        prof = np.nanmedian(D[:, zs], axis=1)
    prof = ndimage.median_filter(np.nan_to_num(prof, nan=np.nanmedian(prof)), size=3)
    step = np.abs(np.roll(prof, -2) - np.roll(prof, 2))
    step[:3] = step[-3:] = 0
    edges = [k for k in range(2, len(prof) - 2) if step[k] >= min_step and step[k] == step[k - 2:k + 3].max()]
    edges = [0] + edges + [len(prof)]
    segs = [(edges[q], edges[q + 1]) for q in range(len(edges) - 1)]
    out = []
    q = 0
    while q < len(segs):
        a0, a1 = segs[q]
        w = (a1 - a0) * G
        if not (0.55 <= w <= 2.2):
            q += 1
            continue
        # merge a neighbouring leaf of similar width/sign (double door) when the sum stays <= 2.2 m
        cand = [(a0, a1)]
        if q + 1 < len(segs):
            b0, b1 = segs[q + 1]
            if 0.55 <= (b1 - b0) * G <= 1.2 and (b1 - a0) * G <= 2.2 and np.sign(np.median(prof[a0:a1])) == np.sign(np.median(prof[b0:b1])):
                cand.append((a0, b1))
        found = None
        for c0, c1 in cand[::-1]:                               # prefer the merged (double) version
            if c0 < 5 or c1 > len(prof) - 5:
                continue
            # edge steps per height row: depth just inside minus just outside each edge. Local differences
            # are immune to the slow range/incidence bias across the wall.
            with np.errstate(all="ignore"):
                sL = np.nanmedian(D[c0 + 1:c0 + 4, :], axis=0) - np.nanmedian(D[c0 - 4:c0 - 1, :], axis=0)
                sR = np.nanmedian(D[c1 - 4:c1 - 1, :], axis=0) - np.nanmedian(D[c1 + 1:c1 + 4, :], axis=0)
            body = slice(int(0.3 / G), int(1.8 / G))
            sgn_ = np.sign(np.nanmedian(np.r_[sL[body], sR[body]]))
            okL = np.nanmean((sL[body] * sgn_) >= 0.003)
            okR = np.nanmean((sR[body] * sgn_) >= 0.003)
            if min(okL, okR) < 0.6:
                continue
            both = np.nanmax(np.abs(np.vstack([sL, sR])), axis=0)
            head = None
            for zi in range(int(1.85 / G), int(2.5 / G)):
                w3 = both[zi:zi + 3]
                if np.all(np.isfinite(w3)) and np.all(w3 < 0.002):
                    head = zi * G
                    break
            low_ = np.r_[sL[int(0.05 / G):int(0.3 / G)], sR[int(0.05 / G):int(0.3 / G)]] * sgn_
            if head is None or not (1.9 <= head <= 2.45) or np.nanmean(low_ >= 0.002) < 0.4:
                continue
            found = (v0 + c0 * G, v0 + c1 * G, head, float(sgn_ * np.nanmedian(np.r_[sL[body], sR[body]] * sgn_)))
            break
        if found:
            out.append(found)
            q += 2 if found[1] - found[0] > (a1 - a0) * G + 1e-6 else 1
        else:
            q += 1
    return out


def detect_windows(axis, u, sgn, v0, v1):
    """Strip windows on a face whose far side is outside the scan: a sill wall 5-70 cm into the room with a
    ledge on top, the glass/blind plane at the face above it. Returns a list of bays."""
    k = 0 if axis == "x" else 1
    Q = SUB["P"]
    d = (Q[:, k] - u) * sgn                                   # depth into the room
    along = Q[:, 1 - k]
    low = (d > 0.05) & (d < 0.7) & (Q[:, 2] > 0.1) & (Q[:, 2] < 0.7) & (along > v0) & (along < v1)
    if low.sum() < 500:
        return []
    h_, e_ = np.histogram(d[low], bins=np.arange(0.05, 0.7, 0.01))
    d_s = float(e_[np.argmax(h_)] + 0.005)
    sill_pts = low & (np.abs(d - d_s) < 0.02)
    na = int(np.ceil((v1 - v0) / 0.05))
    ia = np.clip(((along - v0) / 0.05).astype(int), 0, na - 1)
    sill_col = np.bincount(ia[sill_pts], minlength=na) >= 8
    near_face = (np.abs(d) < 0.04) & (Q[:, 2] > 1.0) & (Q[:, 2] < 2.4) & (along > v0) & (along < v1)
    glass_col = np.bincount(ia[near_face], minlength=na) >= 8
    bay = ndimage.binary_closing(sill_col & glass_col, iterations=2)
    lab, n = ndimage.label(bay)
    out = []
    for kk in range(1, n + 1):
        idx = np.where(lab == kk)[0]
        a0, a1 = v0 + idx.min() * 0.05, v0 + (idx.max() + 1) * 0.05
        if a1 - a0 < 0.4:
            continue
        inb = (along > a0) & (along < a1)
        sz = Q[sill_pts & inb, 2]
        sill_top = float(np.percentile(sz, 98)) if len(sz) > 20 else None
        ledge = inb & (d > 0.02) & (d < d_s - 0.02) & (Q[:, 2] > 0.5) & (Q[:, 2] < 1.3)
        if ledge.sum() > 50:
            hz, ez = np.histogram(Q[ledge, 2], bins=np.arange(0.5, 1.3, 0.005))
            sill_top = float(ez[np.argmax(hz)] + 0.0025)
        gp = inb & (np.abs(d) < 0.04)
        head = float(np.percentile(Q[gp, 2], 99.5)) if gp.sum() > 50 else None
        mull = []
        if sill_top is not None:                              # transom panes above the sill; frames between
            band = gp & (Q[:, 2] > sill_top + 0.05) & (Q[:, 2] < sill_top + 0.3)
            nb = int(np.ceil((a1 - a0) / 0.02))
            occ = np.bincount(np.clip(((along[band] - a0) / 0.02).astype(int), 0, nb - 1), minlength=nb) >= 3
            occ = ndimage.binary_opening(occ, iterations=1)
            l2, n2 = ndimage.label(occ)
            for q in range(1, n2 + 1):
                ii = np.where(l2 == q)[0]
                if (ii.max() - ii.min() + 1) * 0.02 <= 0.25:
                    mull.append(round(a0 + (ii.mean() + 0.5) * 0.02, 3))
        out.append({"a0": round(a0, 3), "a1": round(a1, 3), "width": round(a1 - a0, 3),
                    "sill_face_depth": round(d_s, 3), "sill_z": round(sill_top, 3) if sill_top else None,
                    "head_z": round(head, 3) if head else None, "mullions": mull,
                    "rays_through": rays_through_rect(axis, u, a0, a1, (sill_top or 0.8) + 0.05, 2.3)})
    return out


def rays_through_rect(axis, u, a0, a1, z0, z1, beyond=0.5):
    k = 0 if axis == "x" else 1
    with np.errstate(divide="ignore", invalid="ignore"):
        s_ = (u - RAY_S[:, k]) / RAY_D[:, k]
    v = RAY_S[:, 1 - k] + s_ * RAY_D[:, 1 - k]
    z = RAY_S[:, 2] + s_ * RAY_D[:, 2]
    L = np.linalg.norm(RAY_D[:, :2], axis=1)
    hit = (s_ > 0) & (s_ < 1) & (v > a0) & (v < a1) & (z > z0) & (z < z1) & ((1 - s_) * L > beyond)
    return int(hit.sum())


# ------------------------------------------------------------------ openings on each face line
openings = []
windows = []
strips = []
for li, L_ in enumerate(lines):
    axis, u, sgn, v0, v1 = L_["axis"], L_["u"], L_["sign"], L_["v0"] - 0.3, L_["v1"] + 0.3
    use_subset(axis, u, v0, v1)
    img = elevation(axis, u, v0, v1) > 0
    far_side = [scanned(*((u - sgn * 0.8, a) if axis == "x" else (a, u - sgn * 0.8))) for a in np.linspace(v0 + 0.3, v1 - 0.3, 9)]
    if not any(far_side):
        for bay in detect_windows(axis, u, sgn, v0, v1):
            if bay["rays_through"] / max(bay["width"], 0.1) < 5 and not bay["mullions"]:
                continue                                         # no light and no frames: not a window
            windows.append({"line": li, "axis": axis, "glass_plane_u": round(u, 4), "room_side": sgn, **bay,
                            "walls": L_["walls"]})
    for fa0, fa1, head, dev in flush_doors(depth_map(axis, u, v0, v1), v0):
        far = u - sgn * 0.8
        mid = (fa0 + fa1) / 2
        fx, fy = (far, mid) if axis == "x" else (mid, far)
        openings.append({"line": li, "axis": axis, "face_u": round(u, 4), "face_sign": sgn, "a0": round(fa0, 3),
                         "a1": round(fa1, 3), "width": round(fa1 - fa0, 3), "z0": 0.0, "z1": round(head, 3),
                         "kind": "door", "rays_through": 0, "rays_per_m2": 0.0,
                         "leaf_recess_m": round(-sgn * dev, 3), "leaf_fill": 1.0, "leads_to_scanned_space": scanned(fx, fy),
                         "walls": L_["walls"], "state_in_scan": "closed", "detected_by": "flush_leaf_depth",
                         "leaf_type": "double_swing" if fa1 - fa0 > 1.3 else "single_swing",
                         "always_closed": not scanned(fx, fy), "hinge": "unknown (not measurable when closed; set by hand)"})
    img = ndimage.binary_closing(img, structure=np.ones((3, 3)))           # fill 2 cm sampling holes
    na, nz = img.shape
    # per column: largest empty run in z within [0, 2.5]
    zlo = np.full(na, np.nan)
    zhi = np.full(na, np.nan)
    for a in range(na):
        col = img[a, : int(2.5 / G)]
        if col.sum() < 3:                                    # face not observed at all here
            continue
        empty = ~col
        lab, n = ndimage.label(empty)
        if n == 0:
            continue
        sizes = ndimage.sum(empty, lab, range(1, n + 1))
        k = np.argmax(sizes) + 1
        idx = np.where(lab == k)[0]
        if sizes[k - 1] * G >= 0.8:
            zlo[a], zhi[a] = idx.min() * G, (idx.max() + 1) * G
    # group neighbouring columns with similar empty runs
    a = 0
    while a < na:
        if np.isnan(zlo[a]):
            a += 1
            continue
        b = a
        while b + 1 < na and not np.isnan(zlo[b + 1]) and abs(zlo[b + 1] - np.nanmedian(zlo[a:b + 1])) < 0.12 \
                and abs(zhi[b + 1] - np.nanmedian(zhi[a:b + 1])) < 0.12:
            b += 1
        width = (b - a + 1) * G
        lo_, hi_ = float(np.median(zlo[a:b + 1])), float(np.median(zhi[a:b + 1]))
        A0, A1 = v0 + a * G, v0 + (b + 1) * G
        kind = None
        if lo_ <= 0.12 and 1.85 <= hi_ <= 2.55 and 0.6 <= width <= 2.6:
            kind = "door"
        elif 0.35 <= lo_ <= 1.3 and hi_ - lo_ >= 0.6 and width >= 0.4:
            kind = "window"
        if kind:
            mid = (A0 + A1) / 2
            behind = -sgn                                    # the wall body / far side is opposite the normal
            # leaf behind the face (closed door): occupancy of planes 3-30 cm behind, inside the opening
            leaf = []
            for off in np.arange(0.03, 0.31, 0.01):
                e2 = elevation(axis, u, A0 + 0.05, A1 - 0.05, half=0.006, offset=behind * off)
                zz = slice(int((lo_ + 0.1) / G), int((hi_ - 0.1) / G))
                leaf.append((off, (e2[:, zz] > 0).mean() if e2.size else 0))
            best_off, best_frac = max(leaf, key=lambda q: q[1])
            through = rays_through_rect(axis, u, A0 + 0.05, A1 - 0.05, lo_ + 0.05, hi_ - 0.05)
            far = u + behind * 0.8
            fx, fy = (far, mid) if axis == "x" else (mid, far)
            leads_scanned = scanned(fx, fy)
            op = {"line": li, "detected_by": "empty_rectangle", "axis": axis, "face_u": round(u, 4), "face_sign": sgn,
                  "a0": round(A0, 3), "a1": round(A1, 3),
                  "width": round(width, 3), "z0": round(lo_, 3), "z1": round(hi_, 3), "kind": kind,
                  "rays_through": through, "rays_per_m2": round(through / max(width * (hi_ - lo_), 1e-3), 1),
                  "leaf_recess_m": round(float(best_off), 3) if best_frac > 0.35 else None,
                  "leaf_fill": round(float(best_frac), 2), "leads_to_scanned_space": leads_scanned,
                  "walls": L_["walls"]}
            if kind == "door":
                op["state_in_scan"] = "closed" if best_frac > 0.35 else ("open" if through > 20 else "unknown")
                op["leaf_type"] = "double_swing" if width > 1.3 else "single_swing"
                op["always_closed"] = not leads_scanned
                op["hinge"] = "unknown (not measurable when closed; set by hand)"
            openings.append(op)
        a = b + 1
    strips.append((li, L_, img, v0))
log(f"raw openings: {len(openings)}")
for ex in OVR.get("extra_doors", []):                          # user-added doors (position measured on site data)
    openings.append({"line": -1, "detected_by": "user (extra_doors override)", "kind": "door", "axis": ex["axis"],
                     "face_u": ex["face_u"], "face_sign": ex["face_sign"], "a0": ex["a0"], "a1": ex["a1"],
                     "width": round(ex["a1"] - ex["a0"], 3), "z0": ex["z0"], "z1": ex["z1"], "rays_through": 0,
                     "rays_per_m2": 0.0, "leaf_recess_m": None, "leaf_fill": 1.0, "leads_to_scanned_space": True,
                     "walls": [], "state_in_scan": ex["state_in_scan"], "leaf_type": ex["leaf_type"],
                     "always_closed": False, "hinge": "unknown", "note": ex["note"]})

# the two faces of one wall see the same opening: merge (same axis, |u| within 0.45 m, overlapping)
merged = []
for op in sorted(openings, key=lambda q: -(q["a1"] - q["a0"])):
    dup = next((m for m in merged if m["axis"] == op["axis"] and abs(m["face_u"] - op["face_u"]) < 0.45
                and min(m["a1"], op["a1"]) - max(m["a0"], op["a0"]) > 0.5 * min(m["width"], op["width"])
                and m["kind"] == op["kind"]), None)
    if dup is None:
        merged.append(dict(op, faces=[op["face_u"]]))
    else:
        dup["faces"].append(op["face_u"])
        if op.get("leaf_recess_m") and not dup.get("leaf_recess_m"):
            dup.update({k: op[k] for k in ("leaf_recess_m", "leaf_fill", "state_in_scan") if k in op})
openings = merged
# windows must let light through (otherwise: furniture shadow / unobserved face)
openings = [o for o in openings if o["kind"] == "door"]
# user decisions on detected doors (matched by location, not by id)
filled = []
for ov in OVR.get("door_overrides", []):
    mt = ov["match"]
    for o in list(openings):
        if o["axis"] == mt["axis"] and abs(o["face_u"] - mt["u"]) < 0.3 and abs((o["a0"] + o["a1"]) / 2 - mt["a_mid"]) < mt["tol"]:
            openings.remove(o)
            if ov["action"] == "fill_wall":
                filled.append({**{k: o[k] for k in ("axis", "face_u", "a0", "a1", "z1")}, "note": ov["note"]})
            log(f"override: door at {o['axis']}={o['face_u']:.2f} {o['a0']:.2f}..{o['a1']:.2f} -> {ov['action']}")
if OVR.get("unscanned_doors", {}).get("action") == "fill_wall":      # policy: private / unscanned rooms -> wall
    for o in list(openings):
        if not o.get("leads_to_scanned_space", True):
            openings.remove(o)
            filled.append({**{k: o[k] for k in ("axis", "face_u", "a0", "a1", "z1")}, "note": OVR["unscanned_doors"]["note"]})
            log(f"policy: door at {o['axis']}={o['face_u']:.2f} {o['a0']:.2f}..{o['a1']:.2f} leads to unscanned space -> wall")
for rp in OVR.get("door_replace", []):                       # one double door in one frame (user)
    old_ = [o for o in openings if o["axis"] == rp["axis"] and abs(o["face_u"] - rp["u"]) < 0.3
            and min(o["a1"], rp["a1"]) - max(o["a0"], rp["a0"]) > 0]
    base = old_[0] if old_ else {"line": -1, "face_u": rp["u"], "face_sign": 1, "walls": [], "rays_through": 0,
                                 "rays_per_m2": 0.0, "leaf_recess_m": None, "leaf_fill": 1.0}
    for o in old_:
        openings.remove(o)
    openings.append({**base, "axis": rp["axis"], "a0": rp["a0"], "a1": rp["a1"], "width": round(rp["a1"] - rp["a0"], 3),
                     "z0": 0.0, "z1": rp["z1"], "kind": "door", "leaf_type": rp["leaf_type"], "state_in_scan": "closed",
                     "leads_to_scanned_space": True, "always_closed": False, "hinge": "unknown",
                     "detected_by": "user (door_replace)", "note": rp["note"]})
    log(f"override: {len(old_)} door(s) -> one {rp['leaf_type']} {rp['a0']}..{rp['a1']}")
hinge_rule = OVR.get("hinge_rule", "")
for o in openings:
    if str(o.get("hinge", "")).startswith("unknown"):
        o["hinge"] = "estimated (rule): " + hinge_rule
for k, op in enumerate(sorted(openings, key=lambda q: (q["kind"], q["axis"], q["face_u"], q["a0"]))):
    op["id"] = f"{'D' if op['kind'] == 'door' else 'N'}{k:03d}"
openings = sorted(openings, key=lambda q: q["id"])
# --- extend detected bays to every wall piece on the same exterior line (furniture hides parts of the sill)
wall_by_id = {w["id"]: w for w in PLAN["walls"]}
col_spans = [(c["x0"], c["x1"], c["y0"], c["y1"]) for c in PLAN["columns"]]
ext = []
lines_with_bays = {}
for w in sorted(windows, key=lambda q: (q["axis"], q["glass_plane_u"])):   # one facade = glass planes within 15 cm
    key = next((k for k in lines_with_bays if k[0] == w["axis"] and abs(k[1] - w["glass_plane_u"]) < 0.15), None)
    lines_with_bays.setdefault(key or (w["axis"], w["glass_plane_u"]), []).append(w)
for (axis, _), bays in lines_with_bays.items():
    gu = float(np.median([b["glass_plane_u"] for b in bays]))
    sill = float(np.median([b["sill_z"] for b in bays if b["sill_z"]]))
    head = float(np.median([b["head_z"] for b in bays if b["head_z"]]))
    depth = float(np.median([b["sill_face_depth"] for b in bays]))
    side = bays[0]["room_side"]
    sp = [d for b in bays for d in np.diff(b["mullions"]) if 1.0 <= d <= 2.0]
    pitch = float(np.median(sp)) if sp else None
    pieces = sorted([w for w in PLAN["walls"] if w["axis"] == axis and abs(w.get("face_u", w["c"]) - gu) < 0.15],
                    key=lambda w: w["v0"])
    for w in pieces:
        a0, a1 = w["v0"], w["v1"]
        if a1 - a0 < 0.4:
            continue
        ext.append({"axis": axis, "glass_plane_u": round(gu, 4), "room_side": side, "a0": round(a0, 3), "a1": round(a1, 3),
                    "width": round(a1 - a0, 3), "sill_face_depth": round(depth, 3), "sill_z": round(sill, 3),
                    "head_z": round(head, 3), "mullion_pitch": round(pitch, 3) if pitch else None,
                    "walls": [w["id"]], "source": "measured line, extended over the wall piece"})
# --- envelope windows where no wall was detected (user-confirmed locations, dimensions measured here)
for ov in OVR.get("envelope_windows", []):
    use_subset(ov["axis"], ov["approx_c"], ov["a0"], ov["a1"], reach=1.0)
    k = 0 if ov["axis"] == "x" else 1
    Q = SUB["P"]
    inr = (Q[:, 1 - k] > ov["a0"] + 0.1) & (Q[:, 1 - k] < ov["a1"] - 0.1)
    up = inr & (Q[:, 2] > 0.9) & (Q[:, 2] < 2.3) & (np.abs(Q[:, k] - ov["approx_c"]) < 0.45)
    h_, e_ = np.histogram(Q[up, k], bins=np.arange(ov["approx_c"] - 0.45, ov["approx_c"] + 0.451, 0.01))
    gu = float(e_[np.argmax(h_)] + 0.005)
    d = (Q[:, k] - gu) * ov["room_side"]
    low = inr & (Q[:, 2] > 0.1) & (Q[:, 2] < 0.7) & (d > 0.05) & (d < 0.8)
    h2, e2 = np.histogram(d[low], bins=np.arange(0.05, 0.8, 0.01))
    depth = float(e2[np.argmax(h2)] + 0.005) if low.sum() > 200 else 0.3
    sz = Q[low & (np.abs(d - depth) < 0.02), 2]
    sill = float(np.percentile(sz, 98)) if len(sz) > 50 else 0.8
    gp = inr & (np.abs(Q[:, k] - gu) < 0.04)
    head = float(np.percentile(Q[gp, 2], 99.5)) if gp.sum() > 50 else None
    ext.append({"axis": ov["axis"], "glass_plane_u": round(gu, 4), "room_side": ov["room_side"], "a0": ov["a0"],
                "a1": ov["a1"], "width": round(ov["a1"] - ov["a0"], 3), "sill_face_depth": round(depth, 3),
                "sill_z": round(sill, 3), "head_z": round(head, 3) if head else None, "mullion_pitch": None,
                "walls": [], "source": f"override '{ov['name']}' ({ov['note']}); dimensions measured"})
    log(f"envelope window {ov['name']}: glass {ov['axis']}={gu:.3f}, sill face depth {depth:.2f}, sill {sill:.2f}, head {head}")
dedup = []
for w in sorted(ext, key=lambda q: 0 if q["source"].startswith("override") else 1):
    span = [w["a0"], w["a1"]]
    for o in dedup:
        if o["axis"] == w["axis"] and abs(o["glass_plane_u"] - w["glass_plane_u"]) < 0.3:
            if o["a0"] <= span[0] < o["a1"]:
                span[0] = o["a1"]
            if o["a0"] < span[1] <= o["a1"]:
                span[1] = o["a0"]
            if o["a0"] <= span[0] and span[1] <= o["a1"]:
                span = [0, 0]
    if span[1] - span[0] >= 0.3:
        dedup.append(dict(w, a0=round(span[0], 3), a1=round(span[1], 3), width=round(span[1] - span[0], 3)))
detected_bays = windows
windows = dedup
for k, w in enumerate(sorted(windows, key=lambda q: (q["axis"], q["glass_plane_u"], q["a0"]))):
    w["id"] = f"N{k:03d}"
windows = sorted(windows, key=lambda q: q["id"])
log(f"openings: {len(openings)} doors, {len(windows)} window bays")

# ------------------------------------------------------------------ glass lines: panels between vertical frames
glass = []
for g in PLAN["glass_partitions"]:
    use_subset(g["axis"], g["c"], g["a0"], g["a1"])
    img = elevation(g["axis"], g["c"], g["a0"] - 0.1, g["a1"] + 0.1, half=0.05) > 0
    # frames are thin vertical members filled over the full height, including the clear-glass zones
    zones = np.r_[np.arange(int(0.1 / G), int(0.4 / G)), np.arange(int(1.97 / G), int(2.2 / G))]
    col_fill = img[:, zones].mean(1)
    lab, n = ndimage.label(col_fill > 0.6)
    posts = [float(g["a0"] - 0.1 + (np.mean(np.where(lab == k)[0]) + 0.5) * G) for k in range(1, n + 1)
             if (lab == k).sum() * G <= 0.12]
    posts = [p for p in posts if g["a0"] + 0.3 <= p <= g["a1"] - 0.3]
    merged_posts = []
    for p_ in sorted(posts):                                    # posts closer than 40 cm are one frame
        if merged_posts and p_ - merged_posts[-1] < 0.4:
            merged_posts[-1] = (merged_posts[-1] + p_) / 2
        else:
            merged_posts.append(p_)
    posts = merged_posts
    k_ = 0 if g["axis"] == "x" else 1
    Qg = SUB["P"]
    on = (np.abs(Qg[:, k_] - g["c"]) < 0.03) & (Qg[:, 1 - k_] > g["a0"]) & (Qg[:, 1 - k_] < g["a1"]) & (Qg[:, 2] > 1.9)
    zz = np.sort(Qg[on, 2])
    ceil_line = float(np.percentile(zz, 99)) if len(zz) > 50 else None
    # frosted band: rows (heights) where the film returns points along most of the line
    fill = img[:, : int(2.6 / G)].mean(0)
    rows = np.where(fill[: int(2.3 / G)] > 0.35)[0]
    rows = rows[rows * G > 0.25]
    band = None
    if len(rows):
        runs = np.split(rows, np.where(np.diff(rows) > 2)[0] + 1)
        r = max(runs, key=len)
        band = [round(r.min() * G, 2), round((r.max() + 1) * G, 2)]
    # head: the clear glass above the frosted band ends where the top rail / header starts
    head = ceil_line
    if band is not None:
        for zi in range(int((band[1] + 0.08) / G), len(fill)):
            if fill[zi] > 0.5:
                head = zi * G
                break
    lay = OVR["glass_door_layout"].get(g["name"], {"doors": [], "max_pane": 1.5})
    if "head_z" in lay:                                        # measured by hand on the LAS elevation
        head = lay["head_z"]
    doors_ = lay["doors"]
    if doors_ == "all_panels":
        cuts = posts if len(posts) >= 2 else []
        edges = [g["a0"]] + cuts + [g["a1"]]
        segs = []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            n_ = max(1, int(round((e1 - e0) / lay["max_pane"])))
            segs += [(e0 + q * (e1 - e0) / n_, e0 + (q + 1) * (e1 - e0) / n_) for q in range(n_)]
        panels = [{"a0": round(a, 3), "a1": round(b, 3), "type": "sliding_door"} for a, b in segs]
    else:
        opens_ = lay.get("openings", [])
        wpan_ = lay.get("wall_panels", [])
        edges = sorted(set([g["a0"], g["a1"]] + [x for d_ in doors_ + opens_ + wpan_ for x in d_]))
        panels = []
        for e0, e1 in zip(edges[:-1], edges[1:]):
            if any(abs(e0 - d_[0]) < 1e-6 and abs(e1 - d_[1]) < 1e-6 for d_ in doors_):
                panels.append({"a0": e0, "a1": e1, "type": "swing_door"})
                continue
            if any(abs(e0 - d_[0]) < 1e-6 and abs(e1 - d_[1]) < 1e-6 for d_ in opens_):
                panels.append({"a0": e0, "a1": e1, "type": "open_doorway"})
                continue
            if any(abs(e0 - d_[0]) < 1e-6 and abs(e1 - d_[1]) < 1e-6 for d_ in wpan_):
                panels.append({"a0": e0, "a1": e1, "type": "wall_panel"})
                continue
            n_ = max(1, int(np.ceil((e1 - e0) / lay["max_pane"])))
            panels += [{"a0": round(e0 + q * (e1 - e0) / n_, 3), "a1": round(e0 + (q + 1) * (e1 - e0) / n_, 3),
                        "type": "fixed"} for q in range(n_)]
    glass.append({**g, "frames_at": [round(p, 3) for p in posts], "head_z": round(head, 3) if head else None,
                  "frosted_band_z": band, "panels": panels,
                  "pattern": "clear top / frosted band / clear bottom (REF-C1..C3)", "layout_source": lay})
    widths = [p_["a1"] - p_["a0"] for p_ in panels]
    log(f"glass {g['name']}: {len(panels)} panels {np.round(widths, 2).tolist()}, head {head}, frosted band {band}")

json.dump({"_doc": "Openings measured on the v3 plan faces (07b). Edit 'kind', 'leaf_type', 'hinge', 'always_closed' "
                   "by hand if needed; geometry is in world metres (a0/a1 along the wall, z0/z1 heights).",
           "doors": [o for o in openings if o["kind"] == "door"],
           "windows": windows,
           "window_bays_detected": detected_bays,
           "doors_filled_with_wall": filled,
           "glass_lines": glass}, open(CFG / "dx_hub_no_obj_v1_openings.json", "w"), indent=1)
log(f"wrote {CFG / 'dx_hub_no_obj_v1_openings.json'}")

# ------------------------------------------------------------------ figures: elevation strips per face line
for li, L_, img, v0 in strips:
    ops = [o for o in openings if li == o["line"] or (o["axis"] == L_["axis"] and abs(o["face_u"] - L_["u"]) < 0.05)]
    if not ops:
        continue
    fig, ax = plt.subplots(figsize=(max(6, img.shape[0] * G * 1.3), 3.6))
    ax.imshow(img.T, origin="lower", cmap="gray_r", extent=[v0, v0 + img.shape[0] * G, 0, ZMAX], aspect="equal")
    for o in ops:
        c = "red" if o["kind"] == "door" else "deepskyblue"
        ax.add_patch(Rectangle((o["a0"], o["z0"]), o["a1"] - o["a0"], o["z1"] - o["z0"], fill=False, ec=c, lw=2))
        lbl = o["id"] + (f" {o.get('state_in_scan', '')}" if o["kind"] == "door" else "")
        ax.text(o["a0"], o["z1"] + 0.05, lbl, color=c, fontsize=8)
    ax.set_title(f"line {li}: {L_['axis']}={L_['u']:.3f} (normal {'+' if L_['sign'] > 0 else '-'}{L_['axis']}), "
                 f"walls {', '.join(L_['walls'][:6])}", fontsize=9)
    ax.set_xlabel("along (m)")
    plt.tight_layout()
    plt.savefig(FIG / f"line_{li:03d}.png", dpi=55)
    plt.close()
log("figures done")
