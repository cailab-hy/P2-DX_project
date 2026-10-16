"""Stage 11b (v3): encode the walkthrough frames (11_walkthrough_v3.py) -> walkthrough/DX_Hub_no_obj_v1_walkthrough.mp4.

Run with the `media` conda env (PyAV): /home/cai/miniforge3/envs/media/bin/python 11b_encode_walkthrough.py
"""
import glob
import json
import os

import av
import numpy as np
from PIL import Image

USD = "/home/cai/P2&DX_project/P2&DX/USD"
CFG = json.load(open(os.environ.get("WALK_CFG", f"{USD}/config/dx_hub_no_obj_v1_walkthrough.json")))
FRAMES = CFG.get("frames_dir", f"{USD}/work/walkthrough_frames")
OUT = CFG.get("video", f"{USD}/DX_Hub_Building_no_obj_v1/walkthrough/DX_Hub_no_obj_v1_walkthrough.mp4")

files = sorted(glob.glob(f"{FRAMES}/f*.jpg"))
W, H = CFG["resolution"]
with av.open(OUT, "w", options={"movflags": "+faststart"}) as out:
    st = out.add_stream("libx264", rate=CFG["fps"])
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "18", "preset": "slow"}
    for f in files:
        out.mux(st.encode(av.VideoFrame.from_ndarray(np.asarray(Image.open(f).convert("RGB")), format="rgb24")))
    out.mux(st.encode())
print(f"{len(files)} frames -> {OUT}")
