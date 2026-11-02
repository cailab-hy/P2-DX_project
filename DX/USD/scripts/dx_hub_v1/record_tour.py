"""Headless RTX tour of the layout's changed spots: slow dolly shots, frames with a caption -> OUT_DIR/f_00000.png (ffmpeg makes the video).

    python record_tour.py STAGE.usda OUT_DIR [FONT.ttf] [MAX_FRAMES]
"""
import os
import sys
import time

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "width": 1280, "height": 720, "renderer": "RaytracedLighting"})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

path, out = sys.argv[1:3]
font = ImageFont.truetype(sys.argv[3], 30) if len(sys.argv) > 3 else ImageFont.load_default()
max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
os.makedirs(out, exist_ok=True)
FPS, SECONDS, SETTLE = 12, 2.5, 12
SHOTS = [
    ('펜던트 조명 27개', (-1.9, 3.2, 1.2), (-8.5, 12.0, 2.2)),
    ('조명 불빛', (-3.0, 7.3, 1.05), (-3.56, 6.58, 1.95)),
    ('소파 배치', (-5.8, 1.9, 2.4), (-8.0, 5.0, 1.08)),
    ('창가 원기둥 소파 6개', (-9.0, 2.2, 1.9), (-10.5, 0.4, 0.25)),
    ('벽걸이 TV, 둥근 테이블, 화분', (-2.75, -0.1, 1.55), (-5.1, -2.05, 1.3)),
    ('BRS120M 4대', (-9.3, 12.6, 1.7), (-11.6, 11.75, 0.72)),
    ('모니터 (서쪽 줄)', (-10.6, 15.6, 1.75), (-12.4, 18.0, 0.9)),
    ('모니터 (사무 구역)', (-9.0, 17.2, 1.75), (-8.9, 20.4, 0.9)),
    ('서랍장', (-7.9, 15.9, 1.6), (-7.63, 17.9, 0.45)),
    ('서랍장, 소화기', (-6.3, 17.1, 1.75), (-4.75, 18.75, 0.45)),
    ('RB-Y1', (-2.0, 17.0, 1.75), (-2.8, 18.95, 0.75)),
    ('AI Worker SG2', (-3.5, 12.6, 1.8), (-3.5, 15.3, 0.9)),
    ('TV와 책상 사이로 옮긴 의자', (-4.3, 12.2, 1.7), (-4.25, 14.5, 0.5)),
    ('책상과 모니터', (-1.4, 12.9, 1.85), (-1.4, 15.0, 0.85)),
    ('카운터 앞 화분, 꽃 화분', (-2.8, 9.2, 1.75), (-0.8, 10.0, 0.7)),
    ('소화기, 화분', (-2.7, 5.9, 1.6), (-0.9, 5.15, 0.35)),
    ('LG CLOi 4대', (-3.0, 0.9, 1.65), (-0.45, -0.65, 0.55)),
    ('마음AI 에이든 3대', (0.7, 0.55, 1.75), (0.9, 2.6, 0.55)),
    ('RBM-S100a 2대', (4.6, -0.2, 1.7), (3.95, -1.95, 0.6)),
]


def camera_matrix(eye, aim):
    f = aim - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, (0, 0, 1))
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return Gf.Matrix4d(*r, 0, *u, 0, *(-f), 0, *eye, 1)


omni.usd.get_context().open_stage(path)
for _ in range(60):
    app.update()
stage = omni.usd.get_context().get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
cam = UsdGeom.Camera.Define(stage, '/World/TourCamera')
cam.CreateFocalLengthAttr(12.0)
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(11.787)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200))
op = cam.AddTransformOp()
rp = rep.create.render_product('/World/TourCamera', (1280, 720))
rgb = rep.AnnotatorRegistry.get_annotator('rgb')
rgb.attach([rp])
n, t0 = 0, time.time()
per_shot = int(FPS * SECONDS)
for caption, eye, aim in SHOTS:
    eye, aim = np.array(eye, float), np.array(aim, float)
    f = aim - eye
    side = np.cross(f / np.linalg.norm(f), (0, 0, 1))
    start, end = eye - 0.10 * f - 0.15 * side, eye + 0.10 * f + 0.15 * side
    op.Set(camera_matrix(start, aim))
    for _ in range(SETTLE):     # textures of a new spot
        app.update()
    for i in range(per_shot):
        s = i / (per_shot - 1)
        s = s * s * (3 - 2 * s)
        op.Set(camera_matrix(start + (end - start) * s, aim))
        rep.orchestrator.step(rt_subframes=4)
        img = Image.fromarray(np.asarray(rgb.get_data())[..., :3].astype(np.uint8))
        d = ImageDraw.Draw(img, 'RGBA')
        w = d.textlength(caption, font=font)
        d.rectangle((24, 642, 24 + w + 32, 696), fill=(0, 0, 0, 170))
        d.text((40, 650), caption, font=font, fill=(255, 255, 255, 255))
        img.save(f'{out}/f_{n:05d}.png')
        n += 1
        if n % 12 == 0:
            print(f'[tour] {n} frames, {(time.time() - t0) / n:.2f} s/frame, {caption}', flush=True)
        if n >= max_frames:
            break
    if n >= max_frames:
        break
print(f'[tour] done {n} frames {time.time() - t0:.0f} s', flush=True)
rgb.detach()
rp.destroy()
app.close()
