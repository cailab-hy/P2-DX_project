"""Stage 7a figures: plan overlay, v2 comparison and Q1-Q6 evidence sheets for the v3 floor plan."""
import json
import sys

import numpy as np
from PIL import Image
from pxr import Usd, UsdGeom

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MPoly, Rectangle  # noqa: E402

import v3_geom  # noqa: E402
from common import LAS, OUT, WORK, apply, read_las  # noqa: E402

FIG = OUT.parent / "DX_Hub_Building_no_obj_v1" / "plan"
J = json.load(open(OUT.parent / "config" / "dx_hub_no_obj_v1_plan.json"))
FRAMES = sys.argv[1] if len(sys.argv) > 1 else None          # folder with phone frames f_###.jpg
plt.rcParams.update({"font.size": 9})

xyz, _ = read_las(LAS)
P = apply(np.array(J["frame"]["T_las_to_world"]), xyz[::2]).astype(np.float32)
del xyz
band = P[(P[:, 2] > 1.6) & (P[:, 2] < 2.3)]
low = P[(P[:, 2] > 0.3) & (P[:, 2] < 1.2)]

# v2 walls (top caps of the wall prisms)
v2 = Usd.Stage.Open(str(OUT.parent / "DX_Hub_Building" / "DX_Hub_Building.usda"))
m = UsdGeom.Mesh(v2.GetPrimAtPath("/DX_Hub_Building/Visual/Walls"))
V = np.asarray(m.GetPointsAttr().Get())
F = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
top = V[F][:, :, 2].min(1) > V[:, 2].max() - 0.01
v2_tris = V[F[top]][:, :, :2]

src_counter = open(__file__.replace("07_building_v3_plan_figs.py", "07_building_v3_plan.py")).read()
exec(src_counter[src_counter.index("def rounded_u"):src_counter.index("def face_x")])   # rounded_u()


def draw_plan(ax, show_v2=False, v3=True):
    if show_v2:
        for t in v2_tris:
            ax.add_patch(MPoly(t, closed=True, fc="red", ec="none", alpha=0.35))
    if not v3:
        return
    colors = {"measured": "black", "estimated_interior": "tab:blue", "estimated_exterior": "tab:purple"}
    for w in J["walls"]:
        c = colors[w["thickness"]]
        r = Rectangle((w["c"] - w["t"] / 2, w["v0"]), w["t"], w["v1"] - w["v0"]) if w["axis"] == "x" else \
            Rectangle((w["v0"], w["c"] - w["t"] / 2), w["v1"] - w["v0"], w["t"])
        r.set(fc=c, ec=c, alpha=0.75, lw=0.3)
        ax.add_patch(r)
    for c in J["columns"]:
        ax.add_patch(Rectangle((c["x0"], c["y0"]), c["x1"] - c["x0"], c["y1"] - c["y0"], fc="darkgreen", alpha=0.8))
    for d in J["diagonal_walls"]:
        ax.plot([d["p0"][0], d["p1"][0]], [d["p0"][1], d["p1"][1]], c="magenta", lw=3)
    for g in J["glass_partitions"]:
        xs, ys = ([g["c"]] * 2, [g["a0"], g["a1"]]) if g["axis"] == "x" else ([g["a0"], g["a1"]], [g["c"]] * 2)
        ax.plot(xs, ys, c="deepskyblue", lw=3)
    for f in J["weak_faces_glass_or_partial_candidates"]:
        xs, ys = ([f["u"]] * 2, [f["v0"], f["v1"]]) if f["axis"] == "x" else ([f["v0"], f["v1"]], [f["u"]] * 2)
        ax.plot(xs, ys, c="orange", lw=2, ls="--")
    for c in J["counters"]:
        s = 1 if c["opening"] == "+y" else -1
        for face in ("outer", "inner"):
            d = c[face]
            Pp = v3_geom.sample(rounded_u(d["x_left"], d["x_right"], d["y_end"], (d["corner_r_left"], d["corner_r_right"]),
                                          s, d["arm_len_left"], d["arm_len_right"], d["x_arm_left"], d["x_arm_right"]), 3)
            ax.plot(Pp[:, 0], Pp[:, 1], c="saddlebrown", lw=1.5)


# ---------------------------------------------------------------- full plan
fig, ax = plt.subplots(figsize=(15, 22))
ax.hist2d(band[:, 0], band[:, 1], bins=(1120, 1840), range=[[-14, 14], [-23, 23]], cmap="gray_r", vmax=6)
draw_plan(ax)
for r in J["rooms"]:
    ax.add_patch(MPoly(r["polygon"], closed=True, fc="gold", alpha=0.08, ec="goldenrod", lw=0.5))
QS = {"Q1": (-5.2, -2.1), "Q2": (-0.3, 5.05), "Q3": (6.1, 14.3), "Q4": (-13.0, -0.4), "Q5": (4.8, -21.8),
      "Q6": (11.0, -14.6)}
for q, (x, y) in QS.items():
    ax.annotate(q, (x, y), xytext=(x + 1.2, y + 0.8), color="crimson", fontsize=12, weight="bold",
                arrowprops=dict(arrowstyle="->", color="crimson"))
ax.set_xlim(-14, 14)
ax.set_ylim(-23.5, 23.5)
ax.set_aspect("equal")
ax.set_xticks(range(-14, 15, 2))
ax.set_yticks(range(-23, 24, 2))
ax.grid(alpha=0.25)
mt = J["metrics"]
ax.set_title("DX Hub v3 measured plan over LiDAR wall band (1.6-2.3 m)\n"
             "black=wall (measured thickness), blue=interior wall (estimated t=0.12), purple=exterior wall (windows, t=0.25)\n"
             "green=column, magenta=diagonal wall, cyan=glass A/B/C, orange dashed=weak face (glass/partial?), brown=U counters\n"
             f"wall face-LAS median {mt['wall_face_to_LAS_median_m'] * 100:.1f} cm, p95 {mt['wall_face_to_LAS_p95_m'] * 100:.1f} cm, "
             f"free-space IoU {mt['free_space_IoU']:.3f}, pieces<0.15 m: {mt['n_wall_pieces_shorter_than_0.15m']}")
plt.savefig(FIG / "plan_v3_overlay.png", dpi=90, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------- v2 vs v3
fig, ax = plt.subplots(1, 2, figsize=(26, 20))
for a, (t, sv2, sv3) in zip(ax, [("v2 (red) vs v3 (black/colour)", True, True), ("v3 only", False, True)]):
    a.hist2d(band[:, 0], band[:, 1], bins=(1120, 1840), range=[[-14, 14], [-23, 23]], cmap="gray_r", vmax=6)
    draw_plan(a, show_v2=sv2, v3=sv3)
    a.set_xlim(-14, 14)
    a.set_ylim(-23.5, 23.5)
    a.set_aspect("equal")
    a.grid(alpha=0.25)
    a.set_title(t)
plt.savefig(FIG / "plan_v2_vs_v3.png", dpi=70, bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------- Q evidence sheets
def elevation(ax, p0, p1, half=0.08, title=""):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    L = np.linalg.norm(d)
    d /= L
    n = np.array([-d[1], d[0]])
    rel = P[:, :2] - p0
    t = rel @ d
    s = rel @ n
    m_ = (np.abs(s) < half) & (t > -0.2) & (t < L + 0.2) & (P[:, 2] > -0.05) & (P[:, 2] < 2.85)
    ax.hist2d(t[m_], P[m_, 2], bins=(max(int((L + 0.4) / 0.02), 10), 145), cmap="gray_r", vmax=12)
    ax.set_aspect("equal")
    ax.set_xlabel("along line (m)")
    ax.set_ylabel("z (m)")
    ax.axhspan(1.6, 2.3, color="orange", alpha=0.08)
    ax.set_title(title, fontsize=9)


def topview(ax, box, title):
    x0, x1, y0, y1 = box
    mm = (band[:, 0] > x0) & (band[:, 0] < x1) & (band[:, 1] > y0) & (band[:, 1] < y1)
    ml = (low[:, 0] > x0) & (low[:, 0] < x1) & (low[:, 1] > y0) & (low[:, 1] < y1)
    ax.scatter(low[ml, 0], low[ml, 1], s=0.05, c="tan", alpha=0.4)
    ax.scatter(band[mm, 0], band[mm, 1], s=0.08, c="0.25")
    draw_plan(ax, show_v2=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(title, fontsize=9)


def frame(ax, t, note):
    ax.axis("off")
    if FRAMES is None:
        return
    ax.imshow(Image.open(f"{FRAMES}/f_{t:03d}.jpg"))
    ax.set_title(f"phone video t={t}s: {note}", fontsize=9)


Q = [
    ("Q1", "diagonal partition (-6.7,-0.2)->(-3.7,-4.1) + line y=-0.2",
     (-13.5, -2.5, -5, 1.5), [((-6.72, -0.18), (-3.62, -4.08), "diagonal: solid ends, middle without upper-band points"),
                              ((-12.9, -0.2), (-6.7, -0.2), "y=-0.2 line: low wall / glass?")], [(33, "glass partition, frosted CARE film"), (38, "robot room, glass wall on the right")]),
    ("Q2", "small box at (-0.3, 5.05)", (-2, 1.5, 3.5, 7), [((-1.2, 5.05), (0.6, 5.05), "section through the box")],
     [(59, "U-table left end (column?)")]),
    ("Q3", "jagged corridor wall x=6, y 12-16.5", (4.5, 8, 11.5, 17), [((6.05, 11.8), (6.05, 16.8), "east face of corridor x=6.05")],
     [(90, "east corridor: door frames")]),
    ("Q4", "west wall bumps (x=-13)", (-14, -10.5, -8, 21.5), [((-13.15, -13), (-13.15, 21.5), "west wall x=-13.15 elevation")],
     [(52, "west windows between columns"), (78, "column + window sill")]),
    ("Q5", "south corridor end", (2.5, 7.5, -23.5, -19.5), [((3.6, -21.85), (6.2, -21.85), "y=-21.85 across corridor end")],
     [(5, "core corridor: window at the end")]),
    ("Q6", "east corridor jag y=-14.6", (5.5, 14, -16, -12.5), [((6.1, -14.62), (13.8, -14.62), "south face of east corridor")],
     [(92, "east corridor: recessed doors")]),
]
for name, desc, box, lines, frs in Q:
    fig = plt.figure(figsize=(22, 11))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.6, 1.3])
    topview(fig.add_subplot(gs[:, 0]), box, f"{name}: {desc}\n(top view: dark=1.6-2.3 m, tan=0.3-1.2 m, red=v2 walls)")
    for k, (p0, p1, ttl) in enumerate(lines[:2]):
        elevation(fig.add_subplot(gs[k, 1]), p0, p1, title=ttl)
    for k, (t, note) in enumerate(frs[:2]):
        frame(fig.add_subplot(gs[k, 2]), t, note)
    plt.tight_layout()
    plt.savefig(FIG / f"evidence_{name}.png", dpi=60)
    plt.close()
print("figures written to", FIG)
