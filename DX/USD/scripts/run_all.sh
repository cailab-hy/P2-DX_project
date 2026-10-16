#!/usr/bin/env bash
# Full DX Hub scan -> Isaac Sim USD pipeline. Run from this directory.
set -euo pipefail
PY=${PY:-/home/cai/miniforge3/envs/env_isaaclab/bin/python}
cd "$(dirname "$0")"
$PY 01_frame.py           # world frame + splat registration                 (~2 min)
$PY 02_mesh.py            # Poisson scan mesh, visual/collision LODs           (~15 min)
$PY 02b_colour.py         # scan-mesh vertex colours from the splat            (~2 min)
$PY 03_structure.py       # clean architectural model + furniture extraction  (~1 min)
$PY 04_usd.py             # USD layers                                         (~1 min)
$PY 05_validate_isaac.py  # Isaac Sim headless physics/render check            (~4 min)
$PY 06_building_only.py     # building-only export -> ../DX_Hub_Building     (~5 s)
