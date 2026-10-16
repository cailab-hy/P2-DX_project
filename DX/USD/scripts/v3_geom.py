"""Geometry helpers for the no_obj_v1 building model: decompose measured 2D outlines into straight lines and
circular arcs (least squares, max deviation tolerance), and sample them back into clean polygons."""
import cv2
import numpy as np
from scipy import ndimage


def fit_line(P):
    c = P.mean(0)
    _, _, Vt = np.linalg.svd(P - c)
    d = Vt[0]
    n = np.array([-d[1], d[0]])
    return c, d, np.abs((P - c) @ n).max()


def fit_circle(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(1), rcond=None)
    cx, cy = sol[:2]
    r = np.sqrt(max(sol[2] + cx ** 2 + cy ** 2, 1e-12))
    return cx, cy, r, np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r).max()


def mask_contour(mask, res, lo, smooth=3.0, step=0.01):
    """Outer contour of a binary mask (indexed [i=x, j=y]) as a closed, evenly resampled polyline (m)."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8).T.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=len)[:, 0, :].astype(float)            # (col=x index, row=y index)
    P = np.c_[lo[0] + (c[:, 0] + 0.5) * res, lo[1] + (c[:, 1] + 0.5) * res]
    P = np.c_[ndimage.gaussian_filter1d(P[:, 0], smooth, mode="wrap"), ndimage.gaussian_filter1d(P[:, 1], smooth, mode="wrap")]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([P, P[:1]]), axis=0), axis=1))]
    s = np.arange(0, seg[-1], step)
    closed = np.vstack([P, P[:1]])
    return np.c_[np.interp(s, seg, closed[:, 0]), np.interp(s, seg, closed[:, 1])]


def decompose(P, tol=0.015, min_arc_r=0.25, max_arc_r=15.0, axis_snap_deg=3.0, closed=True):
    """Greedy split of a polyline into the longest line / arc pieces within `tol`, then a merge pass.
    A closed polyline starts at its sharpest corner so pieces do not straddle it."""
    if closed:
        d1 = np.roll(P, -3, 0) - P
        d0 = P - np.roll(P, 3, 0)
        ang = np.abs(np.arctan2(d0[:, 0] * d1[:, 1] - d0[:, 1] * d1[:, 0], (d0 * d1).sum(1)))
        P = np.roll(P, -int(np.argmax(ang)), 0)
        P = np.vstack([P, P[:1]])
    prims, i = [], 0
    while i < len(P) - 1:
        best = None
        for kind in ("line", "arc"):
            lo_, hi_ = i + 2, len(P)
            ok_j = None
            j = i + 3
            while j <= hi_:                                 # grow while it fits
                Q = P[i:j]
                if kind == "line":
                    dev = fit_line(Q)[2]
                else:
                    if j - i < 6:
                        j += 1
                        continue
                    cx, cy, r, dev = fit_circle(Q)
                    if not (min_arc_r <= r <= max_arc_r):
                        dev = np.inf
                if dev <= tol:
                    ok_j = j
                    j += max(1, (j - i) // 8)
                else:
                    break
            if ok_j is not None:
                L = ok_j - i
                if best is None or L > best[1] * (1.15 if kind == "arc" else 1.0):   # prefer lines on ties
                    best = (kind, L, ok_j)
        if best is None:
            best = ("line", 2, i + 2)
        kind, _, j = best
        Q = P[i:j]
        if kind == "line":
            c, d, dev = fit_line(Q)
            t = (Q - c) @ d
            p0, p1 = c + d * t.min(), c + d * t.max()
            ang_ = np.degrees(np.arctan2(d[1], d[0])) % 180
            off = min(ang_ % 90, 90 - ang_ % 90)
            if off <= axis_snap_deg:                         # Manhattan snap around the midpoint
                mid = (p0 + p1) / 2
                if min(ang_, 180 - ang_) <= axis_snap_deg:   # along x
                    p0, p1 = np.array([p0[0], mid[1]]), np.array([p1[0], mid[1]])
                else:
                    p0, p1 = np.array([mid[0], p0[1]]), np.array([mid[0], p1[1]])
            prims.append({"type": "line", "p0": p0.tolist(), "p1": p1.tolist(), "dev": float(dev), "_ij": (i, j)})
        else:
            cx, cy, r, dev = fit_circle(Q)
            a = np.unwrap(np.arctan2(Q[:, 1] - cy, Q[:, 0] - cx))
            prims.append({"type": "arc", "cx": float(cx), "cy": float(cy), "r": float(r),
                          "a0_deg": float(np.degrees(a[0])), "a1_deg": float(np.degrees(a[-1])), "dev": float(dev),
                          "_ij": (i, j)})
        i = j - 1
    prims = merge_runs(P, prims, tol, min_arc_r, max_arc_r, axis_snap_deg)
    for p in prims:
        p.pop("_ij", None)
    return merge_small(prims)


def make_prim(P, i, j, kind, axis_snap_deg):
    Q = P[i:j]
    if kind == "line":
        c, d, dev = fit_line(Q)
        t = (Q - c) @ d
        p0, p1 = c + d * t.min(), c + d * t.max()
        ang_ = np.degrees(np.arctan2(d[1], d[0])) % 180
        if min(ang_ % 90, 90 - ang_ % 90) <= axis_snap_deg:
            mid = (p0 + p1) / 2
            if min(ang_, 180 - ang_) <= axis_snap_deg:
                p0, p1 = np.array([p0[0], mid[1]]), np.array([p1[0], mid[1]])
            else:
                p0, p1 = np.array([mid[0], p0[1]]), np.array([mid[0], p1[1]])
        return {"type": "line", "p0": p0.tolist(), "p1": p1.tolist(), "dev": float(dev), "_ij": (i, j)}
    cx, cy, r, dev = fit_circle(Q)
    a = np.unwrap(np.arctan2(Q[:, 1] - cy, Q[:, 0] - cx))
    return {"type": "arc", "cx": float(cx), "cy": float(cy), "r": float(r), "a0_deg": float(np.degrees(a[0])),
            "a1_deg": float(np.degrees(a[-1])), "dev": float(dev), "_ij": (i, j)}


def merge_runs(P, prims, tol, rmin, rmax, snap):
    """Second pass: replace the longest run of consecutive pieces that fits one line or one arc."""
    out, k = [], 0
    while k < len(prims):
        best = (k, prims[k])
        for m in range(k + 1, min(k + 40, len(prims))):
            i, j = prims[k]["_ij"][0], prims[m]["_ij"][1]
            Q = P[i:j]
            if fit_line(Q)[2] <= tol:
                best = (m, make_prim(P, i, j, "line", snap))
                continue
            cx, cy, r, dev = fit_circle(Q)
            if dev <= tol and rmin <= r <= rmax:
                best = (m, make_prim(P, i, j, "arc", snap))
        out.append(best[1])
        k = best[0] + 1
    return out


def merge_small(prims, min_len=0.04):
    """Drop sub-4 cm slivers between pieces (corner rounding of the raster)."""
    out = []
    for p in prims:
        L = length(p)
        if L < min_len and out:
            continue
        out.append(p)
    return out


def length(p):
    if p["type"] == "line":
        return float(np.linalg.norm(np.subtract(p["p1"], p["p0"])))
    return float(abs(np.radians(p["a1_deg"] - p["a0_deg"])) * p["r"])


def sample(prims, max_deg=3.0):
    """Primitives -> polygon vertices; arcs split every <= max_deg degrees."""
    pts = []
    for p in prims:
        if p["type"] == "line":
            pts += [p["p0"], p["p1"]]
        else:
            n = max(2, int(np.ceil(abs(p["a1_deg"] - p["a0_deg"]) / max_deg)) + 1)
            a = np.radians(np.linspace(p["a0_deg"], p["a1_deg"], n))
            pts += np.c_[p["cx"] + p["r"] * np.cos(a), p["cy"] + p["r"] * np.sin(a)].tolist()
    pts = np.array(pts)
    keep = np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-3]
    return pts[keep]
