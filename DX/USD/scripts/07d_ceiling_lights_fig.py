"""Stage 7d (v3): figure of the ceiling lights before / after row-column alignment (07c) on the LiDAR intensity map
-> DX_Hub_Building_no_obj_v1/validation/ceiling_lights_aligned.png
"""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, WORK  # noqa: E402

CFG = OUT.parent / "config"
I = np.load(WORK / "ceil_intensity.npy")
L = json.load(open(CFG / "dx_hub_no_obj_v1_ceiling_lights.json"))
COL = {"300x1200": "lime", "300x900": "magenta", "300x600": "yellow", "600x600": "cyan", "600x1200": "deepskyblue"}

fig, axes = plt.subplots(1, 2, figsize=(22, 18))
for ax, key, title in ((axes[0], "meas", "measured (LiDAR intensity)"), (axes[1], "", "aligned to rows / columns")):
    ax.imshow(np.where(np.isfinite(I), I, np.nan).T, origin="lower", extent=[-14, 14, -23, 23], cmap="gray")
    for p in L["panels"]:
        x, y = (p["x_meas"], p["y_meas"]) if key else (p["x"], p["y"])
        ax.add_patch(plt.Rectangle((x - p["sx"] / 2, y - p["sy"] / 2), p["sx"], p["sy"], fill=False,
                                   ec=COL.get(p["size_mm"], "white"), lw=1.3))
    for h in L["hvac"]:
        ax.add_patch(plt.Rectangle((h["x"] - 0.42, h["y"] - 0.42), 0.84, 0.84, fill=False, ec="orange", lw=1.3))
    for d in L["downlights"]:
        if key and "x_meas" not in d:
            continue                                          # corridor-line downlights have no single measured twin
        x, y = (d["x_meas"], d["y_meas"]) if key else (d["x"], d["y"])
        ax.add_patch(plt.Circle((x, y), 0.13, fill=False, ec="red", lw=1.2))
    if key:
        ax.set_title(title + " - corridor downlights shown after re-spacing only on the right")
    else:
        ax.set_title(title)
        for p in L["panels"] + L["downlights"]:
            if "x_meas" in p and np.hypot(p["x"] - p["x_meas"], p["y"] - p["y_meas"]) > 0.01:
                ax.plot([p["x_meas"], p["x"]], [p["y_meas"], p["y"]], color="yellow", lw=0.8)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-23, 23)
    ax.set_xticks(range(-14, 15, 2))
    ax.set_yticks(range(-23, 24, 2))
    ax.grid(alpha=0.25)
fig.savefig(OUT.parent / "DX_Hub_Building_no_obj_v1/validation/ceiling_lights_aligned.png", dpi=55, bbox_inches="tight")
print("wrote ceiling_lights_aligned.png")
