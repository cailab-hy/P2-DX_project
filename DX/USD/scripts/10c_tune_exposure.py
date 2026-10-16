"""Stage 10c (v3): compare render brightness with the 2026-10-08 night videos (same framing) and report the
light scale that matches the median luminance. Usage: python 10c_tune_exposure.py <render_tag> <frames_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

VAL = Path("/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_no_obj_v1/validation")
tag, FR = sys.argv[1], Path(sys.argv[2])
PAIRS = [("eastroom", "real-01", 23), ("care", "real-01", 2), ("core", "real-02", 3), ("hall", "real-02", 57),
         ("sink", "real-02", 74), ("tile", "real-02", 77), ("tv", "real-02", 51), ("northcorr", "real-01", 13)]


def lum(img):
    a = np.asarray(img.convert("RGB").resize((320, 180))).astype(float) / 255
    a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
    return a @ [0.2126, 0.7152, 0.0722]


rows = []
for view, vid, t in PAIRS:
    r = lum(Image.open(VAL / f"_raw_{tag}_v3_{view}.png"))
    v = lum(Image.open(FR / f"frames_{vid}" / f"f_{t:03d}.jpg"))
    rows.append((view, float(np.median(r)), float(np.median(v))))
    print(f"{view:10s} render median Y {np.median(r):.3f}  video {np.median(v):.3f}  ratio {np.median(v) / np.median(r):.2f}")
ratio = float(np.median([b / a for _, a, b in rows]))
print(f"median video/render luminance ratio: {ratio:.2f}  -> scale linear light intensities by ~{ratio:.2f}")
json.dump({"tag": tag, "rows": rows, "scale": ratio}, open(VAL / f"exposure_{tag}.json", "w"), indent=1)
