# Exploratory scripts (provenance only)

One-off analysis scripts from the v3 ceiling-light study. They are **not** part of the pipeline; their outputs are
cached in `../../work/` and read by `07c_ceiling_lights.py`:

| script | writes (in `work/`) |
|---|---|
| `ceilmap.py` | `ceil_intensity.npy` (mean LiDAR intensity within 5 cm of the ceiling, 2.5 cm grid), `ceil_splat_bright.npy` |
| `ceil2.py` | `ceil_fine.npy` (ceiling height map, 5 cm grid) |
| `detect_lights*.py`, `filter_lights.py` | candidate panels `light_cands*.json` -> `lights_v3b.json` (panels + HVAC) |
| `downlights_det.py` | `downlight_cands.json` (superseded by the downlight search in 07c) |
| `plotlights.py` | review figure |

Run from `USD/` (they use relative `work/` paths).
