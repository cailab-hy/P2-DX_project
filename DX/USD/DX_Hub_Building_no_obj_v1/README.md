# DX Hub Building no_obj v1 — 빈 건물 (DX허브 11층 로봇직업교육센터)

2026-06-22 LiDAR 스캔(FJD Trion SLAM, 9,276만 점)을 측정해서 만든 **가구 없는 빈 건물** USD입니다.
바닥, 벽, 기둥, 천장(단차·우물천장·조명), 창(블라인드), 문, 유리 파티션, 붙박이 카운터만 들어 있습니다.
의자, 책상, 로봇, 모니터처럼 움직일 수 있는 물체는 모두 뺐습니다.
RB-Y1 / LeRobot VLA 연구용 Isaac Sim 6.1 · Isaac Lab 환경으로 쓰는 것이 목적입니다.

- 가구를 배치한 버전은 `../DX_Hub_Building_add_obj_v1/`입니다. 그 파일은 이 에셋을 **참조**하므로, 여기를 다시 생성하면 가구 버전에도 바로 반영됩니다. 이 폴더의 파일은 가구 작업 중에 수정하지 않습니다.
- 모든 형상은 스크립트와 config JSON으로 다시 만들 수 있습니다(아래 「재생성」 참고).
- 좌표계는 이전 버전 v1(`DX_Hub`, 스캔 메시 기반)·v2(`DX_Hub_Building`, 건물만)와 같습니다. 이 두 폴더는 지금 `USD/`에 없으며, 필요하면 `../scripts/run_all.sh`로 다시 만들 수 있습니다.

> **이름 변경 (2026-10-16):** 이전 이름은 `DX_Hub_Building_v3`입니다. 폴더, USD 파일(`DX_Hub_Building_no_obj_v1.usda`, `_scene.usda`), config(`../config/dx_hub_no_obj_v1_*.json`), 순회 영상 이름을 바꿨습니다. 형상과 prim 경로(`/DX_Hub_Building`)는 그대로라 기존 코드는 파일 경로만 바꾸면 됩니다. 생성 스크립트 파일 이름(`07_building_v3_*.py`, `run_all_v3.sh` 등)은 이전 이름 그대로입니다.

## 파일 구성

| 파일 | 내용 |
|---|---|
| `DX_Hub_Building_no_obj_v1.usda` | **에셋 본체**. defaultPrim `/DX_Hub_Building`, kind `assembly`, 변형 옵션 4종, 낮 하늘 HDRI 포함. 형상은 payload로 불러옵니다 |
| `DX_Hub_Building_no_obj_v1_scene.usda` | Isaac Sim에서 바로 열 수 있는 씬. `/World/PhysicsScene`(중력 9.81)과 에셋 참조 |
| `payloads/building.usdc` | 형상, 재질, 조명, 충돌체 (1.2 MB) |
| `payloads/textures/` | 천장 타일(300×600), 탕비실 벽 타일(600×1200) 텍스처 |
| `textures/*.hdr` | 하늘 HDRI 2종 (Poly Haven, CC0) |
| `plan/`, `openings/` | 평면·개구부 측정 근거 이미지 |
| `materials/calibrated_swatches.png` | 영상으로 보정한 마감재 색 |
| `validation/` | 검증 결과(JSON)와 비교 렌더 |
| `walkthrough/DX_Hub_no_obj_v1_walkthrough.mp4` | 실내 순회 영상 (3분 54초) |

## 좌표계

- Z-up, 1 unit = 1 m, kilogramsPerUnit = 1. 바닥 z = 0, 벽은 X/Y 축에 나란합니다(사선 벽 1개만 예외).
- 범위: x −13.4 ~ 14.0, y −23.0 ~ 23.0, 천장 최고 2.76 m.
- LAS·Gaussian splat에서 이 좌표계로 가는 변환(`T_las_to_world`, `T_gsplat_to_world`, splat 배율 0.994)은 `DX_Hub_Building_no_obj_v1.usda`의 `customLayerData`와 `../config/dx_hub_no_obj_v1_plan.json`의 `frame`에 있습니다.
- **원점 (0, 0)은 남쪽 U 카운터 바로 옆**이니 로봇은 다음처럼 빈 바닥에서 시작하세요.

| 위치 | (x, y) |
|---|---|
| 홀 | (−8.0, 10.0) |
| 코어 복도 | (4.8, 1.3) |
| 강의실 | (−8.0, −10.0) |
| 로봇실 | (−9.0, −4.0) |
| 1106호 | (9.5, 13.0) |
| 케어룸 | (0.0, 19.0) |

## 프림 구조

```
/DX_Hub_Building                      (Xform, kind=assembly, variantSets: doors / blinds / fixtures / lighting)
  /Floor/<room>                        방(구역)별 바닥 메시 12개
  /Walls/wall_###, header_###, ...     벽 183개 (문 위 상인방, 창 아래·위 벽 포함, 마감재별 GeomSubset)
  /Columns/C00..C04                    기둥
  /Baseboards                          걸레받이
  /Openings/Doors/Door_00, Door_01     불투명 양여닫이문 2세트 (Frame, Leaf0, Leaf1)
  /Openings/Windows/Window_##          외부 창 14개 (Frame, Glass, Blind_Down/Half/Up)
  /Glass/A_RoomPartition .. G_CareRoomWest   유리 파티션 7라인 (프레임, 고정 유리, 유리문)
  /Ceiling/Zones/<room>_h<높이cm>      방별 천장면 (room 라벨), Soffit_h<cm> = 단차 측면
  /Ceiling/Zones/CoreCorridorPanel     코어 복도 우물천장 패널
  /Ceiling/LightPanels/Diffusers, Frames, Downlights   조명 기구 형상 (발광 재질)
  /Ceiling/HVAC/Cassettes, Grilles     천장형 에어컨 15대
  /Ceiling/Coves                       간접조명 홈의 LED 스트립
  /Fixtures/U_table_north, U_table_south, U_table_north_bar   붙박이 카운터
  /Lights/Panels/*      RectLight 83   (사각 조명, 6500 K)
  /Lights/Downlights/*  DiskLight 61   (원형 다운라이트, 6500 K)
  /Lights/Coves/*       RectLight 264  (간접조명, 3500 K. 곡선 구간은 짧은 현으로 나눠 고르게 빛남)
  /Lights/Daylight/Overcast, Sunny     DomeLight(HDRI) [+ 태양 DistantLight]   ← 에셋 루트 레이어에 있음
  /Collision/*          충돌체 708개 (purpose=guide: 박스 327 + 카운터 볼록 분할 381)
  /Looks/*              재질 (UsdPreviewSurface, 발광체만 OmniPBR MDL), PhysicsMaterial
```

충돌체는 렌더되지 않는 guide 프림으로 따로 두었습니다. 덕분에 보이는 메시는 가볍게 유지하면서도 물리는 실측 박스로 정확하게 계산됩니다.

## 변형 옵션 (variant sets)

| 세트 | 선택지 (굵게 = 기본값) | 동작 |
|---|---|---|
| `doors` | **closed** / open | open은 스캔된 두 공간을 잇는 문짝 15개(불투명 문 + 유리문)와 그 충돌체를 비활성화합니다. 미스캔 공간으로 이어지는 문은 항상 닫혀 있습니다 |
| `blinds` | **down** / half / up | 외부 창 14개의 롤 블라인드 위치 |
| `fixtures` | **counters** / none | 안내데스크·U 카운터 2개·바·싱크대(형상+충돌체) |
| `lighting` | **day** / day_sunny / night / none | day = 흐린 하늘 HDRI + 실내조명, day_sunny = 구름 낀 하늘 + 태양, night = 실내조명만, none = 조명 없음 |

문은 관절(articulation)이 아닌 정적 형상입니다. 열림은 문짝을 없애는 방식이라, 문을 여는 동작 자체를 학습해야 한다면 문짝을 별도 articulation으로 만들어야 합니다.

## 시맨틱 라벨

모든 형상에 `UsdSemantics.LabelsAPI`가 붙어 있습니다(Replicator `semantic_segmentation` 어노테이터에서 바로 쓸 수 있습니다).

- **`class`** (15종): `floor`, `wall`, `column`, `baseboard`, `ceiling`, `light_panel`, `hvac`, `door`, `door_frame`, `window`, `window_glass`, `window_sill`, `blind`, `glass_partition`, `counter`
- **`room`** (바닥·천장면): `hall`, `corridor_NS`, `corridor_core_S`, `corridor_E`, `toilet_front`, `glass_corridor_D`, `pantry_tile`, `east_room_1106`, `care_room`, `lecture_room`, `robot_room`, `room_A`

## Isaac Sim / Isaac Lab에서 쓰기

**Isaac Sim**: `DX_Hub_Building_no_obj_v1_scene.usda`를 열면 됩니다. 변형 옵션은 Property 패널의 Variants에서 바꿉니다.

**Isaac Lab, 지형으로 불러오기** (기본 변형 옵션: 문 닫힘·낮 조명·카운터 있음):

```python
from isaaclab.terrains import TerrainImporterCfg

terrain = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="usd",
    usd_path="/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_no_obj_v1/DX_Hub_Building_no_obj_v1.usda",
)
```

**Isaac Lab, 에셋으로 불러오기** (변형 옵션 지정):

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

building = AssetBaseCfg(
    prim_path="/World/DX_Hub_Building",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/cai/P2&DX_project/P2&DX/USD/DX_Hub_Building_no_obj_v1/DX_Hub_Building_no_obj_v1.usda",
        variants={"doors": "open", "fixtures": "counters", "blinds": "half", "lighting": "night"},
    ),
)
```

- 에셋 안에는 PhysicsScene이 없으므로 Isaac Lab의 `SimulationContext`가 만든 것을 씁니다.
- 두 방식 모두 Isaac Lab 17.0.2에서 검증했습니다(`../scripts/12_isaaclab_load_test.py`, 결과는 `validation/isaaclab_*.json`).

## 측정값과 추정값

| 항목 | 근거 | 구분 |
|---|---|---|
| 벽면 위치, 사선 벽, 기둥 | LiDAR 1.6–2.3 m 띠의 면 검출 | 측정 |
| 벽 두께 | 양면이 보이면 측정. 한 면만 보이면 내벽 0.12 m, 외벽 0.25 m | 측정 / 추정 |
| 문·창 개구부 폭과 높이, 창턱, 창 머리 높이 | 벽면 입면 영상 + 레이 투과 | 측정 |
| 문짝 두께, 손잡이, 경첩 방향 | 표준 치수, 사용자와 합의한 경첩 규칙 | 추정 |
| 유리 라인 위치, 프레임 간격, 필름 띠 높이 | 레이 투과, 입면, splat, 사진 | 측정(일부 사진 기반) |
| U 카운터, 바, 싱크대 높이 | LAS 단면 | 측정 |
| 천장 높이(우물 2.66–2.72 m, 단차 2.44–2.54 m), 코어 복도 패널 2.60/2.62 m | LAS 천장 높이 지도(영역별 중앙값) | 측정 |
| 단차·우물천장 경계선(물결 모양) | 높이 지도에서 추출한 뒤 층 묶기(3 cm 이내 = 같은 층), 평활(σ 0.2 m), 폭 0.4 m 미만 띠와 1.5 m² 미만 섬 제거, 선·호 근사(허용오차 4 cm) | 측정 후 정리 |
| 유리문 복도 단차 띠 끝(사선) | 높이 지도에서 잰 직선을 config `ceiling_regularization.shape_overrides`로 지정(측정점과 3–11 cm 차이) | 측정 + 지정 |
| 사각 조명 83개 위치와 크기 | LiDAR 천장 반사강도 지도 + real_03 영상 | 측정 후 열·줄 정렬(최대 0.3 m 이동, 원 위치는 `x_meas`/`y_meas`로 보존) |
| 천장형 에어컨 15대 | 천장 높이 지도 + 편집 | 측정 |
| 원형 다운라이트 | 홀은 반사강도 지도에서 검출. 코어 복도는 측정 간격 1.20 m로 균등 배치 | 측정 / 추정 |
| 간접조명 홈 폭 0.26 m | 영상 | 추정 |
| 마감재 색 | 2026-10-08 영상 보정(흰 천장 타일 = 반사율 0.80 가정) | 보정 |
| 조명 세기 | 렌더와 영상의 밝기 중앙값을 맞춤(비율 0.97) | 보정 |
| 창밖 하늘 | 일반 HDRI(실제 창밖 풍경 아님) | 대체 |

## 사용자 결정 사항 (`../config/dx_hub_no_obj_v1_overrides.json` 외)

- 미스캔 공간으로 가는 문(교수실·화장실 등)은 벽으로 채우거나 항상 닫힘으로 처리합니다. 유리 B 남쪽 방과 유리 C 너머 방은 스캔되지 않아 닫힌 문으로 두었습니다.
- 사선 벽 1개는 끝까지 이어진 불투명 벽입니다(100인치 TV 벽면).
- 창이 있는 외벽은 일직선으로 맞추고, 실제 기둥만 남겼습니다.
- 바는 `_|--` 형태입니다. 유리 쇼케이스는 가구로 보고 뺐습니다.
- 일반 문짝은 웜그레이 한 가지로 통일했습니다. 홀 천장은 한 색으로 칠했습니다. 탕비실 벽은 타일(600×1200 세로)이고, 싱크대 뒤에는 연한 민트 유리를 넣었습니다.
- 천장 조명은 직선 열·줄로 정렬했습니다(2026-10-08).
- 천장 단차·우물천장 경계는 깨진 곳 없이 매끈한 곡선입니다. 유리문 복도 우물천장은 흰 타일입니다(2026-10-09, 사진 `~/Downloads/re/`).
- 화재감지기, 스피커, 원형 디퓨저는 넣지 않았습니다.

## v2(`DX_Hub_Building`) 대비 변경점

- 스캔 메시를 다듬는 방식 대신 **측정한 벡터 평면**(벽·기둥·사선 벽)에서 형상을 새로 만들었습니다.
- 문(문틀, 문짝, 손잡이), 외부 창(유리, 창틀, 창턱, 블라인드 3단계)을 추가했습니다.
- 유리 파티션을 3라인(A–C)에서 7라인(A–G)으로 늘렸고, 유리문과 열린 출입구를 구분했습니다.
- 천장 조명을 LiDAR 반사강도로 다시 측정했습니다(사각 83, 원형 61, 에어컨 15).
- 천장 단차와 우물천장 경계를 실물 사진처럼 매끈한 곡선으로 정리했습니다. 벽 위에는 간접조명을 두지 않습니다(간접조명 광원 264개).
- 영상으로 보정한 마감재와 방별 마감 구역을 적용했습니다.
- 변형 옵션 4종, 시맨틱 class·room 라벨, 실측 박스 충돌체를 추가했습니다.

## 검증 결과

| 항목 | 결과 | 파일 |
|---|---|---|
| 벽면 ↔ LAS 거리 | 중앙값 0.89 cm, p95 4.3 cm. 벽 띠 점의 90.9%가 10 cm 이내 | `validation/geometry_metrics.json` |
| 2D 자유공간 IoU (모델 vs 측정) | 0.965. 0.15 m 미만 벽 조각 0개 | 〃 |
| 큐브 낙하 (7개 방) | 모두 z = 0.100 (바닥) | `validation/isaac_checks.json` |
| 카운터 위 낙하 | 0.978 / 1.213 / 1.218 / 0.910 m (실측 상판 + 0.1 m) | 〃 |
| 수평 레이 | 유리 A–F 모두에서 막힘. 닫힌 문에서 막히고, `doors=open`이면 통과 | 〃 |
| Isaac Lab 두 방식 로딩 | 변형 옵션 반영, 낙하 정상 | `validation/isaaclab_terrain.json`, `isaaclab_asset.json` |
| 같은 시점 v2 / v3(이 에셋) / splat 비교 (10개 시점) | 구조·개구부·유리·천장 단차가 splat과 일치 | `validation/compare_all.png`, `compare_<시점>.png` |
| 천장 조명 (영상 대비, 정렬 전후) | | `validation/ceiling_vs_video.png`, `ceiling_lights_aligned.png` |
| 순회 영상 경로 충돌 | 7,035프레임 모두 콜라이더와 0.22 m 이상 떨어짐 | `walkthrough/walkthrough_check.json`, `walkthrough_path.png` |

## 알려진 한계

- 렌더한 천장이 영상보다 약간 어둡습니다. 렌더에서 천장은 반사광으로만 밝혀지는데, 영상은 휴대폰 자동노출이 적용되어 있기 때문입니다.
- (−1.65, 5.04)의 600×1200 조명은 실제로 있는지 아직 확인하지 못했습니다(사용자가 현장 확인 예정).
- 코어 복도 다운라이트 일부는 추정 간격(1.20 m)으로 채웠습니다.
- 발광 재질(OmniPBR MDL)은 Omniverse RTX에서만 빛납니다. 다른 USD 뷰어에서는 조명 기구가 발광하지 않는 흰 면으로 보일 수 있습니다.
- LiDAR는 유리를 통과하므로 유리는 프레임과 레이 투과로 위치를 정했습니다.
- 스프링클러, 화재감지기, 비상구 표지, 콘센트, 스위치 같은 작은 설비는 없습니다.
- 스캔되지 않은 공간(유리 B 남쪽 방, 유리 C 너머 방, 교수실·화장실 쪽)은 모델에 없습니다. 그쪽 문은 항상 닫혀 있습니다.
- 문은 정적 형상입니다(열림 = 문짝 제거). 문 여닫기 동작이 필요하면 articulation을 따로 만들어야 합니다.
- 조명 열·줄 정렬로 일부 조명이 측정 위치에서 최대 0.3 m 움직였습니다. 정렬하지 않은 2개는 측정 위치 그대로입니다: 원형 (−5.37, 4.61), (−3.17, 4.09).
- splat 비교 렌더에서 좁은 복도와 탕비실은 splat 자체가 흐릿해서 비교 근거가 약합니다. 이 구간은 LiDAR 단면과 영상으로 확인했습니다.

## 재생성

```bash
cd ../scripts
./run_all_v3.sh                    # 07 → 07b → 07c → 07d → 09 보정 → 08 → 08b → 검증 → Isaac Lab 테스트 (약 6분, RENDER=0 기준)
WALKTHROUGH=1 ./run_all_v3.sh      # 순회 영상까지 (약 +15분)
python 13_package_v3.py             # 전달용 ../DX_Hub_Building_no_obj_v1.zip (NO_VIDEO=1이면 영상 제외)
```

- 입력:
  - 원본 스캔 `../../DATA/DX/...`
  - 캐시 `../work/` (v1 파이프라인 01/02가 만든 `frame.json` 등, 천장 지도, 영상 프레임 `work/video/`)
  - 천장 지도는 `../scripts/exploratory/`의 탐색 스크립트로 만들었습니다(출처 기록용).
- 형상이나 결정을 바꿀 때는 아래 config만 고친 뒤 다시 실행합니다.

| config | 내용 |
|---|---|
| `dx_hub_no_obj_v1_overrides.json` | 사용자 결정 사항(벽 제거·추가, 문 처리, 유리 배치, 외벽 정렬 등) |
| `dx_hub_no_obj_v1_zones.json` | 방별 마감 구역, 마감재, 조명 세기, 하늘 HDRI |
| `dx_hub_no_obj_v1_ceiling_lights_edits.json` | 조명 검출 수정(제거·추가·에어컨 변환)과 정렬 규칙 |
| `dx_hub_no_obj_v1_color_patches.json` | 색 보정에 쓰는 영상 패치 |
| `dx_hub_no_obj_v1_walkthrough.json` | 순회 영상의 카메라 경로와 설정 |
| `dx_hub_no_obj_v1_plan.json`, `dx_hub_no_obj_v1_openings.json`, `dx_hub_no_obj_v1_ceiling_lights.json` | 측정 결과(스크립트가 생성. 직접 고치지 마세요) |
