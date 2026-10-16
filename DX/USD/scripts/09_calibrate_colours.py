"""Stage 9 (v3): calibrate finish albedos from the 2026-10-08 night videos.

Each frame is white-balanced and exposed by a reference patch: either the white acoustic ceiling tile (assumed
neutral, albedo 0.80) or the warm-greige wall albedo already calibrated from the white-referenced frames (chain).
Patch colours are sRGB-decoded to linear, scaled by the per-channel gain, and median-combined per material.
Output: USD/config/dx_hub_no_obj_v1_finishes_calibrated.json (+ swatch image in DX_Hub_Building_no_obj_v1/materials/).
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path("/home/cai/P2&DX_project/P2&DX/USD")
CFG = json.load(open(ROOT / "config" / "dx_hub_no_obj_v1_color_patches.json"))
FR = Path(sys.argv[1])                     # folder with frames_real-01/, frames_real-02/ and photo/
OUT = ROOT / "DX_Hub_Building_no_obj_v1" / "materials"
OUT.mkdir(parents=True, exist_ok=True)


def lin(c):
    c = np.asarray(c, float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def srgb(l):
    l = np.clip(np.asarray(l, float), 0, 1)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * l ** (1 / 2.4) - 0.055)


def patch(img, box):
    x0, y0, x1, y1 = box
    a = lin(np.asarray(img.crop((x0, y0, x1, y1))).reshape(-1, 3))
    lum = a @ [0.2126, 0.7152, 0.0722]
    keep = (lum > np.percentile(lum, 10)) & (lum < np.percentile(lum, 90))      # drop specks / highlights
    return np.median(a[keep], 0)


def open_frame(src):
    kind, name = src.split("/")
    p = FR / (f"frames_{kind}" if kind.startswith("real") else kind) / name
    return Image.open(p).convert("RGB")


LUM = np.array([0.2126, 0.7152, 0.0722])
chroma = {}
for fr in CFG["frames"]:
    img = open_frame(fr["src"])
    for mat, boxes in fr["patches"].items():
        for b in boxes:
            c = patch(img, b)
            chroma.setdefault(mat, []).append(c / max(float(c @ LUM), 1e-6))
res = {}
for mat, spec in CFG["luminance"].items():
    if spec.get("chroma") == "neutral" or (mat not in chroma and "chroma_from" not in spec):
        ch = np.ones(3)
    else:
        ch = np.median(np.array(chroma[spec.get("chroma_from", mat)]), 0)
    a = np.clip(ch / float(ch @ LUM) * spec["Y"], 0.005, 0.95)
    res[mat] = {"linear": [round(float(x), 4) for x in a], "srgb": [round(float(x), 3) for x in srgb(a)],
                "Y": spec["Y"], "how": spec["how"], "n_chroma_samples": len(chroma.get(mat, []))}
json.dump({"_doc": "finish albedos for v3 (09_calibrate_colours.py): chromaticity from the 2026-10-08 night videos, "
                   "luminance anchored (see 'how'); estimates", "materials": res},
          open(ROOT / "config" / "dx_hub_no_obj_v1_finishes_calibrated.json", "w"), indent=1)
sw = Image.new("RGB", (220 * 4, 70 * ((len(res) + 3) // 4)), "white")
d = ImageDraw.Draw(sw)
for k, (m, r) in enumerate(res.items()):
    x, y = (k % 4) * 220, (k // 4) * 70
    d.rectangle([x + 4, y + 4, x + 64, y + 64], fill=tuple(int(c * 255) for c in r["srgb"]))
    d.text((x + 70, y + 10), m, fill=(0, 0, 0))
    d.text((x + 70, y + 30), str(r["srgb"]), fill=(0, 0, 0))
sw.save(OUT / "calibrated_swatches.png")
for m, r in res.items():
    print(f"{m:18s} sRGB {r['srgb']}  linear {r['linear']}  Y={r['Y']}")
