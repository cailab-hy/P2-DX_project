"""Stage 7c (v3): measure ceiling lights from the LiDAR return intensity on the ceiling.

Light diffusers (acrylic) reflect differently from the acoustic tiles / painted ceiling, so every recessed panel
shows as a crisp rectangle on the intensity map (CEIL map = mean intensity of points within 5 cm of the ceiling).
  1. candidates: dark/bright residual segmentation (2 scales) + standard-size template matching (both
     polarities), each required to sit on a flat ceiling (ring height range < 4 cm) with full ring coverage and
     to show a same-sign contrast step on >= 3 of its 4 edges
  2. rows/columns: for every confirmed panel, the same size is searched along its row and column
  3. manual review: config/dx_hub_no_obj_v1_ceiling_lights_edits.json (remove / add / to_hvac)
  4. refinement: position +-0.12 m and size in {300, 600} x {600, 900, 1200} mm, maximising the edge contrast
  5. downlights: small round blobs on the charcoal ceilings (corridors, hall band) + 1D search along corridor axes
  6. HVAC cassettes: v2 height-based squares (~0.84 m) + edits
Writes config/dx_hub_no_obj_v1_ceiling_lights.json. The exploratory candidate files come from the scripts in the
scripts/exploratory/ (ceilmap.py, ceil2.py -> WORK/ceil_*.npy; detect_lights*.py -> WORK/lights_v3b.json).
"""
import json

import numpy as np
from scipy import ndimage

from common import OUT, WORK

CFG = OUT.parent / "config"
ED = json.load(open(CFG / "dx_hub_no_obj_v1_ceiling_lights_edits.json"))
Z = json.load(open(CFG / "dx_hub_no_obj_v1_zones.json"))
I = np.load(WORK / "ceil_intensity.npy")
cz = np.load(WORK / "ceil_fine.npy")
G, LO = 0.025, np.array([-14.0, -23.0])
valid = np.isfinite(I)
Ism = np.where(valid, ndimage.median_filter(np.where(valid, I, np.nanmedian(I)), size=3), np.nan)
D = json.load(open(WORK / "lights_v3b.json"))


def ring_flat(x0, x1, y0, y1, m=0.2):
    k0, k1 = int((x0 - m - LO[0]) / 0.05), int((x1 + m - LO[0]) / 0.05)
    l0, l1 = int((y0 - m - LO[1]) / 0.05), int((y1 + m - LO[1]) / 0.05)
    if k0 < 0 or l0 < 0 or k1 >= cz.shape[0] or l1 >= cz.shape[1]:
        return 9, 0
    hz = cz[k0:k1, l0:l1]
    ring = np.ones(hz.shape, bool)
    ring[int((x0 - LO[0]) / 0.05) - k0:int((x1 - LO[0]) / 0.05) + 1 - k0,
         int((y0 - LO[1]) / 0.05) - l0:int((y1 - LO[1]) / 0.05) + 1 - l0] = False
    v = hz[ring]
    f = np.isfinite(v).mean()
    v = v[np.isfinite(v)]
    return (float(np.percentile(v, 90) - np.percentile(v, 10)) if v.size > 20 else 9), f


def med(xa, xb, ya, yb):
    i0, i1 = int((xa - LO[0]) / G), int(np.ceil((xb - LO[0]) / G))
    j0, j1 = int((ya - LO[1]) / G), int(np.ceil((yb - LO[1]) / G))
    if i0 < 0 or j0 < 0 or i1 >= I.shape[0] or j1 >= I.shape[1]:
        return np.nan
    v = Ism[i0:i1, j0:j1]
    v = v[np.isfinite(v)]
    return np.median(v) if v.size > 3 else np.nan


def edge_score(x, y, sx, sy, d=0.06, inset=0.05):
    x0, x1, y0, y1 = x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2
    s = []
    for ia, ib, oa, ob in [(x0 + inset, x0 + inset + d, x0 - inset - d, x0 - inset),
                           (x1 - inset - d, x1 - inset, x1 + inset, x1 + inset + d)]:
        i_, o_ = med(ia, ib, y0 + 0.1, y1 - 0.1), med(oa, ob, y0 + 0.1, y1 - 0.1)
        s.append(i_ / o_ - 1 if o_ else np.nan)
    for ia, ib, oa, ob in [(y0 + inset, y0 + inset + d, y0 - inset - d, y0 - inset),
                           (y1 - inset - d, y1 - inset, y1 + inset, y1 + inset + d)]:
        i_, o_ = med(x0 + 0.1, x1 - 0.1, ia, ib), med(x0 + 0.1, x1 - 0.1, oa, ob)
        s.append(i_ / o_ - 1 if o_ else np.nan)
    s = np.array(s)
    if np.isnan(s).sum() > 1:
        return -1
    sg = np.sign(np.nanmedian(s))
    if np.nansum(s * sg > 0.12) < 3:
        return -1
    return float(np.nanmedian(np.abs(s)))


def near(p, q, r):
    return np.hypot(p["x"] - q[0], p["y"] - q[1]) < r


panels = [dict(p) for p in D["panels"]]
hvac = [dict(h) for h in D["hvac"]]
for rm in ED["remove"]:
    panels = [p for p in panels if not near(p, rm["near"], rm["r"])]
for th in ED["to_hvac"]:
    hit = [p for p in panels if near(p, th["near"], th["r"])]
    panels = [p for p in panels if p not in hit]
    hvac += [dict(kind="hvac_840", x=round(p["x"], 3), y=round(p["y"], 3), sx=0.84, sy=0.84, src="edit") for p in hit]
for ad in ED["add"]:
    panels.append(dict(kind="panel", x=ad["near"][0], y=ad["near"][1], sx=ad["size"][0], sy=ad["size"][1], src="edit"))

# refinement: position +-0.12 m and standard size, maximising the edge contrast (longer panels preferred on ties,
# the hall / room panels seen in real_03 are 300 x 1200); detections that cannot be refined keep their size
SIZES = [(0.3, 1.2), (0.3, 0.9), (0.3, 0.6), (0.6, 0.6), (0.6, 1.2)]
out = []
for p in panels:
    horiz = p["sx"] >= p["sy"]
    best = None
    sizes_ = [(min(p["sx"], p["sy"]), max(p["sx"], p["sy"]))] if p.get("src") == "edit" else SIZES   # edits keep their size
    for s_, l_ in sizes_:
        sx, sy = ((l_, s_) if horiz else (s_, l_)) if s_ != l_ else (s_, s_)
        for dx in np.arange(-0.12, 0.121, 0.025):
            for dy in np.arange(-0.12, 0.121, 0.025):
                x, y = p["x"] + dx, p["y"] + dy
                hf, f = ring_flat(x - sx / 2, x + sx / 2, y - sy / 2, y + sy / 2)
                if hf > 0.045 or f < 0.8:
                    continue
                sc = edge_score(x, y, sx, sy)
                if sc <= 0:
                    continue
                key = sc + 0.06 * max(sx, sy)
                if best is None or key > best[0]:
                    best = (key, x, y, sx, sy, sc)
    if best is None:
        out.append(dict(x=p["x"], y=p["y"], sx=p["sx"], sy=p["sy"], edge=0.0, src=p.get("src", "") + " (unrefined)"))
        continue
    _, x, y, sx, sy, sc = best
    out.append(dict(x=round(float(x), 3), y=round(float(y), 3), sx=sx, sy=sy, edge=round(sc, 3), src=p.get("src", "")))
# de-duplicate after refinement
fin = []
for p in sorted(out, key=lambda p: -p["edge"]):
    if all(not (abs(p["x"] - q["x"]) < (p["sx"] + q["sx"]) / 2 - 0.05 and abs(p["y"] - q["y"]) < (p["sy"] + q["sy"]) / 2 - 0.05)
           for q in fin):
        fin.append(p)
RC_ = ED.get("row_consistency")
if RC_:
    import collections as _c
    for p in fin:
        if p["sx"] == p["sy"]:
            continue
        horiz = p["sx"] > p["sy"]
        row = [q for q in fin if q["sx"] != q["sy"] and (q["sx"] > q["sy"]) == horiz
               and abs((q["y"] if horiz else q["x"]) - (p["y"] if horiz else p["x"])) < RC_["centre_tol_m"]
               and abs((q["x"] if horiz else q["y"]) - (p["x"] if horiz else p["y"])) < RC_["max_span_m"]]
        if len(row) >= 3:
            (sx, sy), n = _c.Counter((q["sx"], q["sy"]) for q in row).most_common(1)[0]
            if n >= 2 and (sx, sy) != (p["sx"], p["sy"]):
                p["sx"], p["sy"], p["row_fixed"] = sx, sy, True
for p in fin:
    p["size_mm"] = f"{int(round(min(p['sx'], p['sy']) * 1000))}x{int(round(max(p['sx'], p['sy']) * 1000))}"

# downlights on charcoal ceilings: blobs + 1D search along corridor axes
bg = ndimage.median_filter(np.where(valid, I, np.nanmedian(I)), size=17)
rel = np.where(valid, ndimage.median_filter(np.where(valid, I, np.nanmedian(I)), size=2) / np.maximum(bg, 5), 1)
charcoal_boxes = [z["box"] for z in Z["zones"] if "box" in z and Z["zone_finish"].get(z["name"], {}).get("ceiling") == "ceiling_gray"]
charcoal_boxes += [[3.62, 6.2, -15.0, 21.5], [6.0, 13.8, -14.75, -13.15]]          # NS / E corridors (dark ceiling, real_03)
low = ndimage.zoom((np.where(np.isfinite(cz), cz, 9) <= Z["hall_ceiling_rule"]["gray_below_m"]).astype(np.uint8), 2, order=0).astype(bool)
low = low[:I.shape[0], :I.shape[1]]


def is_charcoal(x, y):
    if any(b[0] <= x <= b[1] and b[2] <= y <= b[3] for b in charcoal_boxes):
        return True
    i, j = int((x - LO[0]) / G), int((y - LO[1]) / G)
    return 0 <= i < low.shape[0] and 0 <= j < low.shape[1] and bool(low[i, j])


dl = []
for pol, mask in (("dark", rel < 0.6), ("bright", rel > 1.6)):
    lab, n = ndimage.label(mask & valid)
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        mk = lab[sl] == k
        a = mk.sum() * G * G
        h, w = mk.shape
        if not (0.006 <= a <= 0.05) or not (0.08 <= max(h, w) * G <= 0.26) or min(h, w) / max(h, w) < 0.6:
            continue
        ij = np.argwhere(mk).mean(0)
        x, y = LO[0] + (sl[0].start + ij[0] + 0.5) * G, LO[1] + (sl[1].start + ij[1] + 0.5) * G
        i0, j0 = int((x - 0.3 - LO[0]) / G), int((y - 0.3 - LO[1]) / G)
        if valid[i0:i0 + 24, j0:j0 + 24].mean() < 0.97 or not is_charcoal(x, y):
            continue
        dl.append((round(float(x), 3), round(float(y), 3)))
# corridors: 1D profile along the centre line
for axis, c, a0, a1 in [("y", 4.87, -21.3, 21.0), ("x", -14.0, 6.3, 13.5)]:
    if axis == "x":
        continue
    xs = np.arange(a0, a1, G)
    prof = np.array([np.nanmin([abs(1 - (rel[int((c + dx - LO[0]) / G), int((t - LO[1]) / G)])) for dx in (-0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075)])
                     if valid[int((c - LO[0]) / G), int((t - LO[1]) / G)] else 0 for t in xs])
    score = np.array([max(abs(1 - rel[int((c + dx - LO[0]) / G), int((t - LO[1]) / G)]) for dx in (-0.05, 0, 0.05)) for t in xs])
    pk = [k for k in range(4, len(xs) - 4) if score[k] > 0.55 and score[k] == score[k - 4:k + 5].max()]
    for k in pk:
        y = float(xs[k])
        if all(np.hypot(c - q[0], y - q[1]) > 0.9 for q in dl):
            dl.append((c, round(y, 3)))
# merge duplicates (< 0.5 m) and complete the corridor row at its measured pitch (dark corridor ceilings return
# weak LiDAR intensity, so some downlights are missed; real_03 t3-9 shows an unbroken row)
merged_ = []
for q in sorted(dl):
    if all(np.hypot(q[0] - m_[0], q[1] - m_[1]) >= 0.5 for m_ in merged_):
        merged_.append(q)
dl = merged_
line = sorted(q[1] for q in dl if abs(q[0] - 4.87) < 0.15)
gaps_ = np.diff(line)
pitch = float(np.median(gaps_[(gaps_ > 0.8) & (gaps_ < 1.6)])) if len(gaps_) else 1.2
for y0_, y1_ in zip(line[:-1], line[1:]):
    n_ = int(round((y1_ - y0_) / pitch))
    for k_ in range(1, n_):
        dl.append((4.87, round(y0_ + k_ * (y1_ - y0_) / n_, 3)))
print(f"corridor downlight pitch {pitch:.2f} m")

# alignment (user 2026-10-08): straight rows / columns, corridor downlights on the centre line at an even pitch
AL = ED["alignment"]
PL = json.load(open(CFG / "dx_hub_no_obj_v1_plan.json"))
OP = json.load(open(CFG / "dx_hub_no_obj_v1_openings.json"))
barriers = [((w["c"], w["v0"]), (w["c"], w["v1"])) if w["axis"] == "x" else ((w["v0"], w["c"]), (w["v1"], w["c"])) for w in PL["walls"]]
barriers += [((g["c"], g["a0"]), (g["c"], g["a1"])) if g["axis"] == "x" else ((g["a0"], g["c"]), (g["a1"], g["c"])) for g in OP["glass_lines"]]
barriers += [(tuple(d["p0"]), tuple(d["p1"])) for d in PL["diagonal_walls"]]


def crosses(a, b):
    def orient(p, q, r):
        return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
    return any(orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0 for c, d in barriers)


def dist_to_barrier(q):
    d_ = []
    for (ax_, ay_), (bx_, by_) in barriers:
        v = np.array([bx_ - ax_, by_ - ay_])
        t = np.clip(np.dot(np.array(q) - (ax_, ay_), v) / max(np.dot(v, v), 1e-9), 0, 1)
        d_.append(np.hypot(*(np.array(q) - (ax_, ay_) - t * v)))
    return min(d_)


on_wall = [q for q in dl if dist_to_barrier(q) < AL["min_wall_dist_m"]]
print(f"  dropped {len(on_wall)} downlight blobs on a wall line: {on_wall}")
dl = [dict(x=q[0], y=q[1]) for q in dl if q not in on_wall]
cl_boxes = [c["box"] for c in AL["corridor_lines"]]
items = [("panel", p) for p in fin] + [("dl", q) for q in dl
                                       if not any(b[0] <= q["x"] <= b[1] and b[2] <= q["y"] <= b[3] for b in cl_boxes)]
for p in fin + dl:
    p["x_meas"], p["y_meas"] = p["x"], p["y"]


def orient_of(kind, p):
    return "dl" if kind == "dl" else ("h" if p["sx"] > p["sy"] else "v" if p["sx"] < p["sy"] else "s")


n_snapped = 0
for cross, along in (("y", "x"), ("x", "y")):                     # rows, then columns
    n_ = len(items)
    adj = [[] for _ in range(n_)]
    for i in range(n_):
        for j in range(i + 1, n_):
            (ki, pi), (kj, pj) = items[i], items[j]
            oi, oj = orient_of(ki, pi), orient_of(kj, pj)
            if "dl" not in (oi, oj) and oi != oj:
                continue
            if abs(pi[cross] - pj[cross]) < AL["tol_m"] and abs(pi[along] - pj[along]) < AL["max_gap_m"] \
                    and not crosses((pi["x"], pi["y"]), (pj["x"], pj["y"])):
                adj[i].append(j)
                adj[j].append(i)
    seen = set()
    for s in range(n_):
        if s in seen or not adj[s]:
            continue
        comp, stack = [], [s]
        seen.add(s)
        while stack:
            k = stack.pop()
            comp.append(k)
            for m in adj[k]:
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        vals = [items[k][1][cross] for k in comp]
        ref = [items[k][1][cross] for k in comp if items[k][0] == "panel"] or vals
        c_ = round(float(np.median(ref)), 3)
        out_ = [k for k in comp if abs(items[k][1][cross] - c_) >= AL["tol_m"]]
        if out_:
            print(f"  align: {cross}={c_}: left as measured {[(round(items[k][1]['x'], 2), round(items[k][1]['y'], 2)) for k in out_]}")
        for k in comp:
            if k not in out_:
                n_snapped += abs(items[k][1][cross] - c_) > 0.005
                items[k][1][cross] = c_
for cl in AL["corridor_lines"]:
    b = cl["box"]
    on = sorted([q for q in dl if b[0] <= q["x"] <= b[1] and b[2] <= q["y"] <= b[3]], key=lambda q: q["y"])
    rest = [q for q in dl if q not in on]
    ys = []
    for q in on:                                                   # merge near-duplicates once on the line
        if ys and q["y"] - ys[-1][-1] < cl["merge_m"]:
            ys[-1].append(q["y"])
        else:
            ys.append([q["y"]])
    ys = [float(np.mean(g)) for g in ys]
    runs = [[ys[0]]]
    for y_ in ys[1:]:
        (runs[-1].append(y_) if y_ - runs[-1][-1] <= cl["split_gap_m"] else runs.append([y_]))
    new = []
    for r in runs:
        n_ = max(1, int(round((r[-1] - r[0]) / pitch)))
        new += [dict(x=cl["c"], y=round(r[0] + k * (r[-1] - r[0]) / n_, 3), on_line=cl["name"]) for k in range(n_ + 1)] \
            if len(r) > 1 else [dict(x=cl["c"], y=round(r[0], 3), on_line=cl["name"])]
    print(f"  {cl['name']}: {len(on)} measured downlights -> {len(new)} on x={cl['c']} in {len(runs)} runs")
    dl = rest + new
print(f"alignment: {n_snapped} coordinates snapped to rows / columns")


# drop downlights that collide with panels / hvac
def free(q):
    return all(not (abs(q["x"] - p["x"]) < p["sx"] / 2 + 0.15 and abs(q["y"] - p["y"]) < p["sy"] / 2 + 0.15) for p in fin) and \
        all(np.hypot(q["x"] - h["x"], q["y"] - h["y"]) > 0.6 for h in hvac)


dl = [q for q in dl if free(q)]
json.dump({"_doc": "ceiling lights measured on the LiDAR intensity map (07c_ceiling_lights.py); aligned to rows / "
                   "columns (x_meas / y_meas = measured centre)",
           "panels": fin, "hvac": hvac, "downlights": dl},
          open(CFG / "dx_hub_no_obj_v1_ceiling_lights.json", "w"), indent=1)
import collections
print(f"panels {len(fin)} {dict(collections.Counter(p['size_mm'] for p in fin))}, hvac {len(hvac)}, downlights {len(dl)}")
