# 🏎️ RPS Racing Game

**카메라 앞에서 가위·바위·보를 내며 결승선까지 달리는 2인용 레이싱 보드게임!**

YOLO TFLite 모델이 실시간으로 손 제스처를 인식하고, 매 라운드 결과에 따라 레이싱 트랙 위의 자동차가 전진·후퇴합니다. 특수 타일(부스트, 지뢰, 워프 등)이 전략적 역전 요소를 더합니다.

> **왼손 = Player 1 (🟠)** &nbsp; | &nbsp; **오른손 = Player 2 (🔵)**

---

## 📖 게임 규칙

### 라운드 진행 (5초)

| 단계 | 시간 | 설명 |
|:---:|:---:|---|
| 🪨 ROCK | 0 ~ 1초 | 화면에 "ROCK" 표시 |
| ✌️ SCISSORS | 1 ~ 2초 | 화면에 "SCISSORS" 표시 |
| 🖐️ PAPER | 2 ~ 3초 | 화면에 "PAPER" 표시 |
| 🎯 RECOGNIZE | 3 ~ 5초 | **이 구간에서만 손 제스처 인식!** 마지막 2초 동안 가장 많이 감지된 제스처가 최종 선택됩니다 |

### 승패 효과

| 상황 | 결과 |
|---|---|
| ✊ **바위**로 승리 | 진 사람 **-3칸** 후퇴 |
| ✌️ **가위**로 승리 | 이긴 사람 **+2칸** 전진 |
| 🖐️ **보**로 승리 | 이긴 사람 **+1칸**, 진 사람 **-1칸** |

### 무승부 (같은 제스처)

| 무승부 | 결과 |
|---|---|
| ✊ 바위 vs 바위 | 양쪽 모두 **-1칸** 후퇴 |
| ✌️ 가위 vs 가위 | 양쪽 위치 **SWAP!** (서로 자리 교환) |
| 🖐️ 보 vs 보 | 양쪽 모두 **+1칸** 전진 |

### 특수 상황
- 한 명만 손이 인식되면 → 상대방이 **+1칸**
- 스턴 상태의 플레이어가 있으면 → 상대방이 **+2칸**
- 양쪽 다 스턴이면 → 아무 일도 일어나지 않음

---

## 🗺️ 특수 타일 (20칸 트랙)

트랙 위 특정 칸에 도착하면 즉시 특수 효과가 발동합니다:

| 칸 | 이름 | 아이콘 | 효과 |
|:---:|---|:---:|---|
| 3 | 🚀 **BOOST** | `>>` | **+4칸** 추가 전진 |
| 6 | 🛢️ **OIL** | `##` | 다음 라운드 **스킵** (스턴) |
| 9 | ⬆️ **RAMP** | `/\` | **9→14번 칸**으로 점프 |
| 12 | 💥 **SPIKE** | `!!` | **-2칸** 후퇴 |
| 15 | 🔄 **SWITCH** | `<>` | 상대방과 **위치 교환** |
| 17 | ⬇️ **SLOW** | `\/` | **17→11번 칸**으로 미끄러짐 |

> 특수 타일은 연쇄 발동할 수 있습니다 (최대 3회).

---

## 🏁 승리 조건

- **19번 칸 (FINISH)** 에 먼저 도달한 플레이어가 승리!
- 같은 라운드에 둘 다 도달하면 **PHOTO FINISH (무승부)**
- 게임 종료 후 `[R]` 키로 재시작 가능

---

## ⚙️ Requirements

| 패키지 | 용도 |
|---|---|
| `opencv-python` | 카메라 캡처 및 UI 렌더링 |
| `numpy` | 이미지·배열 연산 |
| `tflite_runtime` | YOLO 모델 추론 (경량 런타임) |

> `tflite_runtime` 대신 `tensorflow`로도 대체 가능하나 용량이 큽니다.

---

## 📁 프로젝트 구조

```text
📦 05_Object_Detection_Based_On-Device_AI
├── 🏎️ RPS_Racing_Game.py          # 메인 게임 (multiprocessing 버전)
├── 🏎️ RPS_Racing_Game_merged.py   # 레이싱 보드 UI (threading 버전)
├── 📋 requirements.txt             # pip 패키지 목록
├── 📂 models/                      # TFLite 모델 파일들
│   ├── best_float32.tflite         #   고정밀 (10MB)
│   ├── best_float16.tflite         #   중간 (5MB)
│   └── best_full_integer_quant.tflite  #   INT8, 가장 빠름 (2.8MB)
└── 📂 examples/                    # 예시 스크립트
    ├── EX_01_Image_Capture.py      #   카메라 캡처 테스트
    └── EX_03_Board_RPS_PreTrained_YOLO.py  #   YOLO 단독 테스트
```

---

## 🚀 Quick Start

```bash
# 저장소 클론
git clone https://github.com/Due-LAMP/RPS_Racing_Game.git
cd RPS_Racing_Game

# 가상환경 생성 및 패키지 설치
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# 게임 실행
python3 RPS_Racing_Game.py
```

### 조작법

| 키 | 동작 |
|:---:|---|
| `Q` | 게임 종료 |
| `R` | 게임 리셋 |

---

## 🍓 Raspberry Pi 팁

<details>
<summary><b>성능 최적화 & 트러블슈팅</b> (클릭하여 펼치기)</summary>

### 성능 최적화
- **INT8 양자화 모델** (`best_full_integer_quant.tflite`) 사용 시 CPU 추론이 가장 빠릅니다
- 카메라 캡처: `320×240` → 디스플레이 시 `640×480`으로 업스케일 (코드에서 자동 처리)
- `MJPG` 포맷 + 버퍼 사이즈 1로 카메라 지연 최소화
- 추론은 별도 프로세스(`multiprocessing`)로 분리하여 GIL 병목 방지

### tflite_runtime 설치
```bash
# Python 버전 확인
python3 -c "import sys; print(sys.version)"

# Pi용 wheel 파일로 설치 (예시)
pip install tflite_runtime-*.whl

# 또는 tensorflow로 대체 (용량 큼)
pip install tensorflow
```

### 트러블슈팅
| 증상 | 해결 |
|---|---|
| `ModuleNotFoundError: tflite_runtime` | Pi 버전에 맞는 `.whl` 파일 설치 또는 `tensorflow`로 대체 |
| 카메라 프레임이 매우 느림 | USB 카메라의 MJPG 지원 여부 확인, `v4l2-ctl`로 포맷 점검 |
| 모델 로딩 에러 | `models/` 경로 및 파일 권한 확인 |
| 원격(XQuartz/X11) 화면 지연 | 로컬 모니터에 직접 연결하거나 VNC 사용 권장 |

</details>

---

## 🛠️ 기술 구현

| 항목 | 설명 |
|---|---|
| **모델** | YOLO TFLite (3 클래스: scissors, rock, paper) |
| **입력** | 320×320 이미지 + letterbox 전처리 |
| **추론** | `multiprocessing.Process`로 별도 프로세스에서 실행 |
| **투표** | 마지막 2초간 가장 많이 감지된 제스처를 최종 선택 |
| **보드 렌더링** | 정적 요소 캐시 + 토큰/HUD만 부분 갱신으로 최적화 |
| **UI** | OpenCV 기반 네온 레이싱 테마, 실시간 카운트다운 HUD |
