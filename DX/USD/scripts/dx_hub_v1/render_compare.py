"""Headless RTX render of the comparison stages (modelled building vs empty building + rescan mesh), same cameras.

    python render_compare.py OUT_DIR STAGE.usda [STAGE.usda ...]   ->  OUT_DIR/<stage>_<camera>.png
"""
import os
import sys

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402

out_dir, stages = sys.argv[1], sys.argv[2:]
os.makedirs(out_dir, exist_ok=True)
for path in stages:
    omni.usd.get_context().open_stage(path)
    for _ in range(60):
        app.update()
    stage = omni.usd.get_context().get_stage()
    cams = [str(p.GetPath()) for p in stage.Traverse() if p.GetTypeName() == "Camera" and str(p.GetPath()).startswith("/World/")]
    only = os.environ.get("CAMS")
    for cam in cams:
        if only and cam.rsplit("/", 1)[1] not in only.split(","):
            continue
        rp = rep.create.render_product(cam, (1280, 720))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb.attach([rp])
        for _ in range(40):  # let the path tracer / textures settle
            app.update()
        rep.orchestrator.step(rt_subframes=16)
        img = np.asarray(rgb.get_data())[..., :3]
        name = f"{os.path.splitext(os.path.basename(path))[0]}_{cam.rsplit('/', 1)[1]}.png"
        Image.fromarray(img.astype(np.uint8)).save(os.path.join(out_dir, name))
        print(f"[render] {name} mean {img.mean():.1f}", flush=True)
        rgb.detach()
        rp.destroy()
app.close()
