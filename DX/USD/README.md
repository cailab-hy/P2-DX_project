# DX허브 11층 로봇직업교육센터 — Isaac Sim 디지털 트윈 (USD)

DX허브 11층 로봇직업교육센터(CARE, Center for AI Robotics Education)를 LiDAR 스캔으로 측정해서 만든 NVIDIA Isaac Sim 6.1 / Isaac Lab용 USD 환경입니다.
RB-Y1 모바일 매니퓰레이터와 LeRobot VLA 연구에서 시뮬레이션 공간으로 쓰는 것이 목적입니다.

- **측정 기반:** 벽, 기둥, 개구부, 천장 단차, 조명 위치는 2026-06-22 LiDAR 스캔에서 쟀고, 마감재 색은 현장 영상으로 보정했습니다. 가구 위치와 크기는 2026-10 현장 사진·영상과 실측값을 썼습니다.
- **재현 가능:** 모든 USD는 `scripts/`의 스크립트와 `config/`의 JSON으로 다시 만들 수 있습니다. 생성된 USD는 직접 고치지 않고, config를 바꾼 뒤 스크립트를 다시 실행합니다.

## 에셋 한눈에 보기

| 폴더 | 내용 | 언제 쓰나 |
|---|---|---|
| [`DX_Hub_Building_no_obj_v1/`](DX_Hub_Building_no_obj_v1/README.md) | **빈 건물.** 바닥, 벽, 기둥, 천장(단차·조명·에어컨), 창과 블라인드, 문, 유리 파티션, 붙박이 카운터. 변형 옵션 4종(문 / 블라인드 / 카운터 / 조명), 시맨틱 라벨, 실측 박스 충돌체 | 가구 없이 공간만 필요할 때, 직접 물체를 배치할 때 |
| [`DX_Hub_Building_add_obj_v1/`](DX_Hub_Building_add_obj_v1/README.md) | **가구 배치 버전.** 빈 건물을 참조하고, 그 위에 홀·주방·침실(케어룸) 가구와 안내데스크 벽 로고를 얹었습니다. 문·서랍 관절, 전자레인지 버튼, 변형체(이불, 베개, 반팔티) 포함 | 실제 센터와 같은 환경에서 로봇 과제를 돌릴 때 |
| 〃 `_Hall` / `_Kitchen` / `_CareRoom.usda` | **공간별 USD.** 가구 버전 전체 파일을 참조하고 다른 공간을 끈 얇은 파일. 전체 파일이 바뀌면 같이 바뀝니다 | 한 공간만 빠르게 불러올 때 |

두 에셋의 관계:

```
DX_Hub_Building_no_obj_v1.usda          빈 건물 (스크립트 07–13이 생성)
        ▲ reference (수정하지 않음)
DX_Hub_Building_add_obj_v1.usda         + /Furniture/{Hall,Kitchen,CareRoom}, 벽 로고, 건물 덮어쓰기(카운터 타공, 냉장고 틈새)
        ▲ reference + active=false
DX_Hub_Building_add_obj_v1_{Hall,Kitchen,CareRoom}.usda   공간별
```

빈 건물을 다시 생성하면 가구 버전과 공간별 파일에 자동으로 반영되고, 가구 버전을 고치면 공간별 파일에 자동으로 반영됩니다.

## 폴더 구조

```
USD/
├── README.md                         이 문서
├── DX_Hub_Building_no_obj_v1/        빈 건물 에셋 (README 있음)
├── DX_Hub_Building_add_obj_v1/       가구 배치 에셋 + 공간별 USD + 가구 단독 USD 26종 (README 있음)
├── config/                           측정 결과와 사용자 결정 사항 (JSON)
│   ├── dx_hub_no_obj_v1_*.json         빈 건물: 평면, 개구부, 천장 조명, 마감 구역, 결정 사항, 순회 경로
│   ├── add_obj_v1_furniture.json       가구 버전: 가구 치수·물성, 배치, 건물 수정, 벽 로고, 공간 박스, 검증 시점
│   └── dx_hub_add_obj_v1_*.json        가구 버전 순회 영상 경로 5종
├── scripts/                          생성·검증 스크립트 (아래 「파이프라인」)
└── work/                             캐시와 참고 자료 사본 (약 3.5 GB, 배포에는 필요 없음)
```

- 원본 스캔은 `../DATA/DX/`(FJD Trion SLAM, LAS 9,276만 점 + Gaussian splat + 360° 영상)에 있습니다. `../DATA/P2/`는 다른 건물(산업용 로봇 실험동)입니다.
- `work/`에는 스캔에서 뽑은 지도와 메시 캐시, 현장 사진·영상 사본(`work/video/`, `work/add_obj_v1/*_ref/`), 영상 프레임이 들어 있습니다.

## 빠른 시작

**Isaac Sim:** 장면 파일을 열면 바로 시뮬레이션할 수 있습니다(PhysicsScene 포함).

| 열 파일 | 내용 |
|---|---|
| `DX_Hub_Building_no_obj_v1/DX_Hub_Building_no_obj_v1_scene.usda` | 빈 건물 |
| `DX_Hub_Building_add_obj_v1/DX_Hub_Building_add_obj_v1_scene.usda` | 가구 배치 전체 (GPU dynamics 켜짐) |
| `DX_Hub_Building_add_obj_v1/DX_Hub_Building_add_obj_v1_{Hall,Kitchen,CareRoom}_scene.usda` | 공간별 |

**Isaac Lab (17.0.2에서 확인):**

```python
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

USD = "/home/cai/P2&DX_project/P2&DX/USD"

# 빈 건물 (변형 옵션 선택 가능)
building = AssetBaseCfg(prim_path="/World/DX_Hub_Building",
                        spawn=sim_utils.UsdFileCfg(usd_path=f"{USD}/DX_Hub_Building_no_obj_v1/DX_Hub_Building_no_obj_v1.usda",
                                                   variants={"doors": "open", "lighting": "day"}))

# 가구 배치 버전 (변형체가 있으므로 SimulationCfg(device="cuda:0") 필요)
furnished = AssetBaseCfg(prim_path="/World/DX_Hub_Building_add_obj",
                         spawn=sim_utils.UsdFileCfg(usd_path=f"{USD}/DX_Hub_Building_add_obj_v1/DX_Hub_Building_add_obj_v1.usda"))
```

지형(`TerrainImporterCfg`)으로 불러오는 방법, 변형 옵션 목록, 관절 이름은 각 폴더의 README에 있습니다.

## 좌표계

- Z-up, 1 unit = 1 m, kilogramsPerUnit = 1, 바닥 z = 0. 범위는 x −13.4 ~ 14.0, y −23.0 ~ 23.0 m입니다.
- 원점 (0, 0)은 남쪽 U 카운터(안내데스크) 바로 옆입니다. 로봇 시작 위치 예: 홀 (−8.0, 10.0), 주방 통로 (1.45, 9.0), 침실 (−2.6, 17.7).
- 대략적인 위치: 홀은 서쪽 넓은 공간(x < −4.2)과 남쪽 안내데스크 앞, 주방(U 카운터)은 x −2.0 ~ 3.9 · y 2.3 ~ 16.3, 침실(케어룸)은 그 북쪽 y 16.2 ~ 21.8입니다(공간 박스는 `config/add_obj_v1_furniture.json`의 `rooms`).
- LAS·Gaussian splat → 이 좌표계 변환 행렬은 `DX_Hub_Building_no_obj_v1.usda`의 `customLayerData`와 `config/dx_hub_no_obj_v1_plan.json`의 `frame`에 있습니다.
- 가구 단독 USD는 원점이 바닥 중앙이고 정면이 +y입니다.

## 순회 영상

| 영상 | 내용 |
|---|---|
| `DX_Hub_Building_no_obj_v1/walkthrough/DX_Hub_no_obj_v1_walkthrough.mp4` | 빈 건물 전체 순회 (3분 54초) |
| `DX_Hub_Building_add_obj_v1/walkthrough/DX_Hub_add_obj_v1_walkthrough_2x.mp4` | 가구 버전 전체 순회 (2분 15초, 2배속) |
| `DX_Hub_Building_add_obj_v1/walkthrough/DX_Hub_add_obj_v1_rooms_tour_2x.mp4` | 홀·침실·주방 집중 순회 (1분 19초, 2배속) |
| `DX_Hub_Building_add_obj_v1/walkthrough/DX_Hub_add_obj_v1_{hall,bedroom,kitchen}_tour_2x.mp4` | 공간별 순회 3개: 홀 36초, 침실 21초, 주방 28초 (2배속) |

모든 영상의 카메라 경로는 렌더 전에 콜라이더에서 0.22 m 이상 떨어져 있는지 검사했습니다.

## 파이프라인 (`scripts/`)

| 번호 | 대상 | 하는 일 |
|---|---|---|
| 01–06 (`run_all.sh`) | 이전 버전 v1 / v2 | 스캔 좌표계(`work/frame.json`), 스캔 메시, 구조 추출. **01·02가 만든 캐시는 빈 건물 생성에 필요합니다.** v1(`DX_Hub/`)·v2(`DX_Hub_Building/`) 출력 폴더는 지금 `USD/`에 없습니다 |
| 07–13 (`run_all_v3.sh`) | 빈 건물 `no_obj_v1` | 07 평면 측정 → 07b 개구부 → 07c/07d 천장 조명 → 09 색 보정 → 08 USD 생성 → 08b 변형 옵션·장면 → 09/10 검증 → 12 Isaac Lab 확인 → 13 배포 zip |
| 11, 11b | 두 버전 공통 | 순회 영상 렌더(`WALK_CFG=<config>`로 경로 선택, `WALK_MODE=check`는 충돌 검사만)와 mp4 인코딩. 경로의 `look` 구간에 `pitch`(카메라 기울기), `hold_s`(멈춤)를 줄 수 있습니다 |
| 20–28 | 가구 버전 `add_obj_v1` | 20 가구 USD → 21 배치·건물 수정·로고·장면(끝에서 27 공간별 USD 자동 생성) → 22 물리·관절 검증 → 23 Isaac Lab 확인 → 24 충돌체 겹침 → 25 전자레인지 버튼 데모 → 26 사진 시점 렌더 → 28 공간별 검증 |

- 스크립트 파일 이름의 `v3`(예: `07_building_v3_plan.py`, `11_walkthrough_v3.py`)는 빈 건물의 이전 이름입니다. 출력은 모두 `no_obj_v1` 이름으로 나옵니다.
- `scripts/exploratory/`는 천장 조명 탐색에 쓴 일회성 스크립트입니다(출처 기록용, 파이프라인 아님).

## 실행 환경

| conda 환경 | 용도 |
|---|---|
| `/home/cai/miniforge3/envs/env_isaaclab` | Isaac Sim 6.1 + Isaac Lab 17.0.2. 거의 모든 스크립트 |
| `/home/cai/miniforge3/envs/media` | PyAV. 영상 디코딩(색 보정용 프레임)과 mp4 인코딩(11b) |

작업하면서 확인한 주의점:

- 변형체(이불, 베개, 반팔티)가 있는 장면은 GPU 시뮬레이션이 필요합니다. `SimulationManager.set_physics_sim_device("cuda:0")`는 **장면을 연 뒤에** 호출해야 합니다(먼저 호출하면 Kit이 비정상 종료됩니다).
- GPU 시뮬레이션 결과를 렌더할 때는 타임라인이 돌아가는 상태(`rep.orchestrator.step(..., pause_timeline=False)`)여야 합니다. 멈춘 상태로 렌더하면 이전 상태가 찍힙니다.
- Kit 프로세스 안에서 별도 설치된 `pxr`(usd-core)을 섞어 쓰지 마세요. USD 버전이 달라 충돌합니다.
- `OmniPBR.mdl`(조명 기구 발광 재질)은 Omniverse에 들어 있는 재질이라, Kit 밖의 USD 도구에서는 "찾을 수 없음" 경고가 나옵니다. 정상입니다.

## 이력

| 날짜 | 내용 |
|---|---|
| 2026-06-22 | LiDAR 스캔 (`DATA/DX/`) |
| 2026-10-07 ~ 10-09 | 빈 건물 v3 제작: 평면·개구부 측정, 천장 조명 재측정과 정렬, 마감재 색 보정, 변형 옵션, 시맨틱 라벨, 검증, 천장 물결 경계 정리 |
| 2026-10-09 ~ 10-16 | 가구 버전 add_obj_v1: 침실(케어룸), 주방(커피 로봇 부스, 싱크 실측, 전자레인지 버튼), 홀(긴 책상, 소파, 창가 책상), 반팔티 변형체, 공간별 USD, 벽 로고 보완(벽 가운데로 이동, 사진의 문구와 기관 로고, 줄마다 같은 높이 선에 정렬) |
| 2026-10-16 | 빈 건물 이름 변경: `DX_Hub_Building_v3` → `DX_Hub_Building_no_obj_v1` (폴더, USD, config) |

## 남은 확인 사항

- (−1.65, 5.04)의 600 × 1200 천장 조명이 실제로 있는지 현장 확인이 필요합니다(지금은 넣어 둠).
- 반팔티는 일반 성인 S/M 치수로 만들었습니다. 실제 옷 치수를 받으면 바꿀 예정입니다.
- 주방 동쪽 벽의 쓰레기통·소화기·화분 순서는 10-15 영상과 다를 수 있습니다(확인 요청 중).
- 홀 소파 묶음과 긴 책상은 6월 스캔 위치입니다. 그 뒤로 바뀌었다면 config의 배치만 고치면 됩니다.
