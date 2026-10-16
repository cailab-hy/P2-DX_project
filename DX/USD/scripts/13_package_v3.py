"""Stage 13 (v3): delivery zip -> USD/DX_Hub_Building_no_obj_v1.zip (top folder DX_Hub_Building_no_obj_v1/).

Includes the asset (usda, payload, textures), README, plan / opening evidence, calibrated swatches, validation JSON
and the summary images, and the walkthrough video. Leaves out the intermediate review renders (validation/_raw_*,
views_*). NO_VIDEO=1 leaves out the walkthrough mp4. The zip is then extracted to a temp folder and opened with
pxr to check that every layer and texture resolves inside the package.
"""
import os
import tempfile
import zipfile
from pathlib import Path

from pxr import UsdUtils

SRC = Path("/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_no_obj_v1")
OUT = SRC.parent / "DX_Hub_Building_no_obj_v1.zip"
SUMMARY = {"compare_all.png", "ceiling_waves_before_after.png", "ceiling_lights_aligned.png", "ceiling_lights_plan.png", "ceiling_vs_video.png",
           "lighting_variants.png", "materials_vs_video.png"}


def keep(rel):
    top = rel.parts[0]
    if top in ("payloads", "textures", "materials", "plan", "openings") or len(rel.parts) == 1:
        return True
    if top == "validation":
        n = rel.name
        return n.endswith(".json") or n in SUMMARY or (n.startswith("compare_") and n.endswith(".png"))
    if top == "walkthrough":
        return rel.suffix == ".png" or (rel.suffix == ".mp4" and os.environ.get("NO_VIDEO") != "1")
    return False


files = sorted(p for p in SRC.rglob("*") if p.is_file() and keep(p.relative_to(SRC)))
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, Path(SRC.name) / p.relative_to(SRC))
print(f"{len(files)} files, {OUT.stat().st_size / 1e6:.1f} MB -> {OUT}")

with tempfile.TemporaryDirectory() as td:
    zipfile.ZipFile(OUT).extractall(td)
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(f"{td}/{SRC.name}/DX_Hub_Building_no_obj_v1_scene.usda")
    outside = [a for a in [l.realPath for l in layers] + list(assets) if not a.startswith(td)]
    print(f"package check: {len(layers)} layers, {len(assets)} assets, unresolved {unresolved}, outside package {outside}")
    assert not outside and set(unresolved) <= {"OmniPBR.mdl"}, "package is not self-contained"
