"""Stage 7b figures: plan with openings, and one evidence crop per door (face occupancy + depth offset)."""
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from common import LAS, OUT, apply, read_las  # noqa: E402

CFG = OUT.parent / "config"
FIG = OUT.parent / "DX_Hub_Building_no_obj_v1" / "openings"
PLAN = json.load(open(CFG / "dx_hub_no_obj_v1_plan.json"))
OPN = json.load(open(CFG / "dx_hub_no_obj_v1_openings.json"))
xyz, _ = read_las(LAS)
P = apply(np.array(PLAN["frame"]["T_las_to_world"]), xyz[::2]).astype(np.float32)
del xyz
band = P[(P[:, 2] > 1.6) & (P[:, 2] < 2.3)]


def seg(axis, u, a0, a1):
    return ([u, u], [a0, a1]) if axis == "x" else ([a0, a1], [u, u])


fig, ax = plt.subplots(figsize=(15, 22))
ax.hist2d(band[:, 0], band[:, 1], bins=(1120, 1840), range=[[-14, 14], [-23, 23]], cmap="gray_r", vmax=6)
for w in PLAN["walls"]:
    r = Rectangle((w["c"] - w["t"] / 2, w["v0"]), w["t"], w["v1"] - w["v0"]) if w["axis"] == "x" else \
        Rectangle((w["v0"], w["c"] - w["t"] / 2), w["v1"] - w["v0"], w["t"])
    r.set(fc="0.35", ec="none", alpha=0.6)
    ax.add_patch(r)
for d in PLAN["diagonal_walls"]:
    ax.plot([d["p0"][0], d["p1"][0]], [d["p0"][1], d["p1"][1]], c="0.35", lw=3)
for c in PLAN["columns"]:
    ax.add_patch(Rectangle((c["x0"], c["y0"]), c["x1"] - c["x0"], c["y1"] - c["y0"], fc="darkgreen", alpha=0.8))
for wnd in OPN["windows"]:
    xs, ys = seg(wnd["axis"], wnd["glass_plane_u"], wnd["a0"], wnd["a1"])
    ax.plot(xs, ys, c="deepskyblue", lw=5, alpha=0.9, solid_capstyle="butt")
    mx, my = np.mean(xs), np.mean(ys)
    ax.text(mx, my, f" {wnd['id']}", color="teal", fontsize=7)
colors = {"fixed": "royalblue", "sliding_door": "magenta", "swing_door": "magenta", "open_doorway": "lime", "wall_panel": "slategray"}
for g in OPN["glass_lines"]:
    for p in g["panels"]:
        xs, ys = seg(g["axis"], g["c"], p["a0"] + 0.02, p["a1"] - 0.02)
        ax.plot(xs, ys, c=colors[p["type"]], lw=4, solid_capstyle="butt")
    xs, ys = seg(g["axis"], g["c"], g["a0"], g["a1"])
    ax.text(np.mean(xs), np.mean(ys), "  " + g["name"].split("_")[0], color="royalblue", fontsize=10, weight="bold")
for d in OPN["doors"]:
    xs, ys = seg(d["axis"], d["face_u"], d["a0"], d["a1"])
    col = "red" if d["always_closed"] else ("orange" if d["state_in_scan"] == "closed" else "limegreen")
    ax.plot(xs, ys, c=col, lw=7, solid_capstyle="butt")
    ax.text(np.mean(xs), np.mean(ys), f" {d['id']}", color=col, fontsize=9, weight="bold")
ax.set_xlim(-14, 14)
ax.set_ylim(-23.5, 23.5)
ax.set_aspect("equal")
ax.set_xticks(range(-14, 15, 2))
ax.set_yticks(range(-23, 24, 2))
ax.grid(alpha=0.25)
ax.set_title("v3 openings. Doors: red = always closed (leads to unscanned space), orange = closed in scan, green = open\n"
             "windows: cyan (glass plane). Glass lines: blue = fixed pane, magenta = glass door, lime = open doorway")
plt.savefig(FIG / "plan_openings.png", dpi=90, bbox_inches="tight")
plt.close()

# per-door evidence: occupancy on the face plane and depth offset
doors = OPN["doors"]
n = len(doors)
fig, axs = plt.subplots(n, 2, figsize=(14, 2.6 * n))
for row, d in zip(np.atleast_2d(axs), doors):
    k = 0 if d["axis"] == "x" else 1
    a0, a1 = d["a0"] - 0.8, d["a1"] + 0.8
    m = (np.abs(P[:, k] - d["face_u"]) < 0.10) & (P[:, 1 - k] > a0) & (P[:, 1 - k] < a1) & (P[:, 2] > 0) & (P[:, 2] < 2.8)
    Q = P[m]
    G = 0.02
    ia = ((Q[:, 1 - k] - a0) / G).astype(int)
    iz = (Q[:, 2] / G).astype(int)
    na, nz = int((a1 - a0) / G) + 1, 141
    occ = np.zeros((na, nz))
    np.add.at(occ, (ia[np.abs(Q[:, k] - d["face_u"]) < 0.025], iz[np.abs(Q[:, k] - d["face_u"]) < 0.025]), 1)
    s_ = np.zeros((na, nz))
    c_ = np.zeros((na, nz))
    np.add.at(s_, (ia, iz), Q[:, k] - d["face_u"])
    np.add.at(c_, (ia, iz), 1)
    D = np.where(c_ > 0, s_ / np.maximum(c_, 1), np.nan)
    row[0].imshow((occ > 0).T, origin="lower", extent=[a0, a1, 0, 2.8], cmap="gray_r", aspect="equal")
    row[1].imshow((D - np.nanmedian(D)).T, origin="lower", extent=[a0, a1, 0, 2.8], cmap="coolwarm", vmin=-0.03,
                  vmax=0.03, aspect="equal")
    for a in row:
        a.add_patch(Rectangle((d["a0"], d["z0"]), d["a1"] - d["a0"], d["z1"] - d["z0"], fill=False, ec="lime", lw=1.5))
        a.tick_params(labelsize=6)
    row[0].set_title(f"{d['id']} {d['axis']}={d['face_u']:.2f} along {d['a0']:.2f}..{d['a1']:.2f}  w {d['width']:.2f} h "
                     f"{d['z1']:.2f}  {d['leaf_type']}, {d['state_in_scan']}, {d['detected_by']}", fontsize=7)
    row[1].set_title("depth offset +-3 cm (red = behind the face)", fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "door_evidence.png", dpi=60)
plt.close()
print("figures written")
