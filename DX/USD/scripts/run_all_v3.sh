#!/usr/bin/env bash
# DX Hub v3 empty-building pipeline (LiDAR scan -> DX_Hub_Building_no_obj_v1). Run from this directory.
# Needs ../work/ (frame.json and caches from 01_frame.py / 02_mesh.py of the v1 pipeline, the ceiling maps from
# exploratory/, and the video frames in ../work/video/). Manual decisions live in ../config/*.json.
#   WALKTHROUGH=1 ./run_all_v3.sh   also renders the walkthrough video (~15 min)
set -euo pipefail
PY=${PY:-/home/cai/miniforge3/envs/env_isaaclab/bin/python}
PYMEDIA=${PYMEDIA:-/home/cai/miniforge3/envs/media/bin/python}
cd "$(dirname "$0")"
$PY 07_building_v3_plan.py              # measured 2D plan -> config/dx_hub_no_obj_v1_plan.json
$PY 07_building_v3_plan_figs.py         # plan evidence figures
$PY 07b_building_v3_openings.py         # doors / windows / glass -> config/dx_hub_no_obj_v1_openings.json
$PY 07b_building_v3_openings_figs.py
$PY 07c_ceiling_lights.py               # ceiling lights (+ row / column alignment) -> config/dx_hub_no_obj_v1_ceiling_lights.json
$PY 07d_ceiling_lights_fig.py
$PY 09_calibrate_colours.py ../work/video   # finish albedos -> config/dx_hub_no_obj_v1_finishes_calibrated.json
$PY 08_building_v3_usd.py               # payloads/building.usdc
$PY 08b_building_v3_assemble.py         # DX_Hub_Building_no_obj_v1.usda (variants) + _scene.usda
$PY 09_validate_v3_geometry.py          # validation/geometry_metrics.json
RENDER=${RENDER:-1} $PY 10_validate_v3_isaac.py   # physics drops / glass rays (+ v2/v3/splat renders)
$PY 12_isaaclab_load_test.py --case terrain
$PY 12_isaaclab_load_test.py --case asset
if [ "${WALKTHROUGH:-0}" = "1" ]; then
  $PY 11_walkthrough_v3.py && $PYMEDIA 11b_encode_walkthrough.py
fi
