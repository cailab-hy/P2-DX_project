# DX Hub v1 배치 (브랜치 `Dx_hub_v1`)

FJD 2D 장비로 스캔(2026-10-30, 2026-11-02)해 측정한 실제 가구·로봇 배치를 DX Hub 디지털 트윈 USD에 반영한 브랜치입니다.

`main`의 `DX_Hub_Building_add_obj_v1`을 기반으로 만들었고, 같은 내용을 [kangkwonlee/Dx_hub_project](https://github.com/kangkwonlee/Dx_hub_project)에서도 볼 수 있습니다. 작업 과정에서 Claude를 참고했습니다.

![전체 모습](DX/USD/DX_Hub_Building_add_obj_v1_layout_20261102/validation/view_lamps_hall.png)

### 변경된 부분 영상

Isaac Sim에서 렌더링한 영상입니다. 추가하거나 옮긴 곳 19군데를 차례로 보여 줍니다(47.5초).

![변경된 부분 영상](DX/USD/DX_Hub_Building_add_obj_v1_layout_20261102/validation/dx_hub_tour.gif)

[고화질 MP4 영상 보기](DX/USD/DX_Hub_Building_add_obj_v1_layout_20261102/validation/dx_hub_tour.mp4)

## 추가한 파일

```
DX/USD/
├── DX_Hub_Building_add_obj_v1_layout_20261102/
│   ├── DX_Hub_Building_add_obj_v1_layout_20261102.usda   # 배치 레이어
│   ├── view.usda                                         # 조명과 확인용 카메라를 더한 스테이지
│   └── validation/                                       # 모델만 렌더링한 이미지
├── config/dx_hub_v1/    # 측정 결과와 배치 입력 JSON
└── scripts/dx_hub_v1/   # 배치, 모델 생성, 측정, 지도 생성 코드
    └── exploratory/     # 중간 시도
```

`main`에 있던 파일은 수정하지 않았습니다. 이 레이어는 `DX_Hub_Building_add_obj_v1`을 서브레이어로 불러오고, 달라진 부분만 덮어씁니다.

- **옮긴 가구:** 위치와 방향만 덮어씁니다.
- **추가한 가구:** `Furniture/Hall/Scan_*` 아래에 정의합니다.

## 사용 방법

SG2 모델은 [ROBOTIS-GIT/cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab)의 `FFW_SG2.usd`를 상대 경로로 참조합니다. 이 저장소와 같은 폴더에 받아야 합니다.

```bash
git clone -b Dx_hub_v1 https://github.com/cailab-hy/P2-DX_project.git
```

```bash
git clone https://github.com/ROBOTIS-GIT/cyclo_lab.git
```

Isaac Sim에서 `DX/USD/DX_Hub_Building_add_obj_v1_layout_20261102/view.usda`를 열면 됩니다. 확인에 쓴 환경은 Isaac Sim 6.2 / Isaac Lab 3.0(aarch64)입니다.

## 배치 다시 만들기

대상 파일이 이미 있으면 먼저 지워야 합니다. 경로는 절대 경로로 지정합니다.

```bash
cd DX/USD/scripts/dx_hub_v1
```

```bash
python place_final.py ../../config/dx_hub_v1 ../../config/dx_hub_v1 "$PWD/../../DX_Hub_Building_add_obj_v1/DX_Hub_Building_add_obj_v1.usda" "$PWD/../../DX_Hub_Building_add_obj_v1_layout_20261102/DX_Hub_Building_add_obj_v1_layout_20261102.usda"
```

`cyclo_lab`이 `~/Isaac/cyclo_lab`이 아닌 곳에 있으면 환경 변수 `CYCLO_LAB`으로 경로를 지정합니다.

`scripts/`는 아직 이전 상태입니다. 줄자 실측으로 고친 테이블, 카운터, 스툴, 시작점 앞 책상은 USD에만 반영되어 있어서, 지금 스크립트로 다시 만들면 이전 배치가 나옵니다.

## 추가한 항목

| 항목 | 수량 | 코드 |
|---|---|---|
| ROBOTIS AI Worker SG2 (관절을 끈 고정 모델) | 1 | `place_final.py` |
| BRS120M | 4 | `brs120m.py` |
| LG CLOi | 4 | `cloi.py` |
| 마음AI 에이든 | 3 | `maumai.py` |
| RBM-S100a | 2 | `rbm_s100a.py` |
| RB-Y1 | 1 | `rby1.py` |
| 서랍장 | 2 | `drawer_chest.py` |
| 둥근 테이블 / 원형 스툴 | 1 / 7 | `round_items.py` |
| 화분(몬스테라) / 꽃 화분 | 4 / 1 | `monstera.py` |
| 벽걸이 TV (스캔 기준 2.53×1.44 m) | 1 | `wall_tv.py` |
| 펜던트 조명 | 27 | `pendant_lamps.py` |
| 모니터 | 31 | `monitors.py` |
| 로봇 책상 칸막이 (스캔 기준 0.80×0.46 m, 두께 3 cm) | 1 | USD에만 반영 (코드는 Dx_hub_project의 `desk_partition.py`) |
| 작업대 박스 / RGB 큐브 (작업대 실측 USD에서 복사, 로봇 쪽 = 북쪽 책상) | 2 / 6 | USD에만 반영 (코드는 Dx_hub_project의 `desk_cube_set.py`) |
| 소화기, 사무용 책상·의자 등 기존 모델 | - | `place_final.py` |

기존 가구 26개는 스캔에 맞춰 옮겼습니다.

## 측정 방법

1. 스캔을 건물 모델에 정합합니다(`register_scan.py`). 건물과의 오차는 3.0 cm, 두 스캔 사이 오차는 2.1 cm입니다.
2. 두 스캔을 합치고 비교합니다(`compare_scans.py`).
3. 높이별 단면과 외곽선 피팅으로 위치, 방향, 크기를 측정합니다.

## Nav2 지도 비교 (2026-11-03)

실물 SG2로 주행해 비교한 후보 지도입니다. **현재 쓰는 지도는 D3입니다.**

| 지도 | 만든 방법 | 벽·물체 칸 수 | 주행 오차 평균 / 상위 10% 경계 / 최대 (cm) | 방향 오차 평균 / 최대 (°) | D3 주행 일치율 |
|---|---|---:|---:|---:|---:|
| A | USD를 단면으로 잘라 만듦 | 44,187 | 22.7 / 38.6 / 53.0 | 2.46 / 7 | 68.7% |
| B | FJD 3D 스캔 단면 | 5,456 | 주행 안 함 | - | 80.6% |
| C | B + 건물 벽 | 38,113 | 주행 안 함 | - | 82.4% |
| D | 실물 라이다 주행 기록으로 만듦 | 38,694 | 16.9 / 41.1 / 59.7 | 미측정 | 87.1% |
| D2 | D + 추가 주행 기록 | 40,983 | 9.4 / 15.6 / 23.3 | 1.35 / 4 | 96.7% |
| **D3** | **D2에서 임시 물건 자리를 비움 (최종)** | **40,393** | **10.3 / 14.1 / 16.1** | **1.47 / 5** | **91.0%** |

- **주행 오차:** AMCL 위치와, 라이다를 FJD 3D 스캔에 맞춘 위치의 차이입니다. 지도마다 주행 경로가 조금 다르고, 스캔 자체의 정확도는 따로 검증하지 않았습니다.
- **D3 주행 일치율:** 어떤 지도를 만들 때도 쓰지 않은 D3 주행(22.1 m)의 라이다 점 중, 지도의 벽·물체에서 10 cm 안에 든 비율입니다.
- **D2가 D3보다 높은 이유:** 이 주행 때 임시 물건이 그 자리에 있어서, 물건을 남겨 둔 D2의 일치율이 더 높게 나옵니다. D3가 덜 정확하다는 뜻은 아닙니다.

## 구역 명칭 (2026-11-05)

USD의 `/DX_Hub_Building_add_obj/Zones`에 구역 39개의 이름과 바닥 범위를 넣었습니다. 레이어에서는 숨겨져 있고 `view.usda`에서만 보입니다. 충돌 형상은 없고, 이름은 벽보다 높은 z 3.3 m에 있어 천장을 숨기고 위에서 봐야 읽힙니다.

| 종류 | 수 | 구역 |
|---|---:|---|
| 작업·전시 | 4 | 로봇 책상, 큐브 보관, 서빙로봇 대기, 전자칠판 앞 |
| 입구 | 11 | 출입구, 방, 서버실, 훈련실, 교육실, 행정실, 교수님 1·2·3, 커피, 주방 |
| 소파 | 6 | 소파 1~6 |
| 주방 | 1 | 싱크대 |
| 연구원 자리 | 17 | 공개 저장소라 이름 대신 자리 번호로 표기, 자리 15·16은 SkillForge 팀 1·2 |

커피 입구와 주방 입구의 바닥 범위는 추정값입니다.

남쪽 유리 줄(y -15.2)은 `Door_02`, `Door_05`만 유리문으로 두고 나머지 패널 7개를 벽으로 바꿨습니다. `wall_004`는 유리문 2짝(`Glass/H_WestDoor`)으로 바꿨습니다. 문은 모두 닫힌 상태로 고정돼 있습니다.

## 한계

- 제품 모델은 제조사 CAD가 아닙니다. 크기는 스캔이나 제품 사양에서, 모양은 제품 사진을 보고 단순화해 만들었습니다.
- 스캔으로 앞뒤를 알 수 없는 물체의 방향은 추정값입니다. 일부 모니터의 수와 방향도 여기에 해당합니다.
- 충돌 형상은 모델별로 넣었고, 짧은 물리 시뮬레이션으로 일부만 확인했습니다. GPU 물리에서는 일부 볼록 메시가 변형체(티셔츠)와 충돌하지 못한다는 경고가 18건 나옵니다.

## 포함하지 않은 것

원본 스캔(`.las`, `.fjdata`)과 합친 점 데이터는 올리지 않았습니다. `config/dx_hub_v1/`의 측정 결과로 배치는 다시 만들 수 있지만, 측정 코드(`register_scan.py`, `compare_scans.py`, 각 모델 코드의 `measure`)는 이 브랜치만으로는 실행되지 않습니다.
