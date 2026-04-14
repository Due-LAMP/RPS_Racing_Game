**Project Overview**
- **Description:** 간단한 2인용 RPS(가위·바위·보) 레이싱 보드 게임입니다. 실시간 손 제스처(왼손=P1, 오른손=P2)를 YOLO TFLite 모델로 인식하여 턴마다 이동합니다.

**Requirements**
- **Python:** 3.8+ 권장
- **Packages:** `opencv-python`, `numpy`, `tflite_runtime` (또는 `tensorflow` + `tensorflow-lite` 대체 가능)
- **Model files:** `models/` 폴더에 TFLite 모델들을 두세요 (예: best_full_integer_quant.tflite)

**Files**
- **Main:** [RPS_Racing_Game.py](RPS_Racing_Game.py) — 게임 실행 파일
- **Examples:** [examples/EX_01_Image_Capture.py](examples/EX_01_Image_Capture.py) — 카메라 캡처 테스트
- **Models:** `models/` 폴더에 `best_float16.tflite`, `best_full_integer_quant.tflite` 등

**Quick Start**
Getting the code (example):

  git clone <your-repo-url>
  cd 05_Object_Detection_Based_On-Device_AI

Create a virtualenv and install requirements (recommended):

  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt

If you need `tflite_runtime` on Raspberry Pi, see the Pi notes below.

Run the game (models must be in `models/`):

  python3 RPS_Racing_Game.py

**Camera / Performance Tips (Raspberry Pi)**
- 캡처 해상도를 낮춰 카메라 지연을 줄였습니다: 프로그램은 기본적으로 캡처를 `320x240`으로 하고, 화면에 표시할 때 `640x480`으로 업스케일합니다.
- 가능한 경우 **INT8 양자화 모델**(`best_full_integer_quant.tflite`)을 사용하세요 — CPU 추론에서 가장 빠릅니다.
- 카메라 설정 권장: `MJPG` 포맷, 버퍼 사이즈 1. (코드에서 자동 설정)
- 메인 프로세스는 렌더링을 담당하고, 추론은 별도 **multiprocessing** 프로세스로 분리하여 GIL 병목을 피합니다.

Raspberry Pi: `tflite_runtime` 설치 팁
- TensorFlow Lite는 Pi에 맞는 휠(wheel) 파일로 설치하는 것이 일반적입니다. 공식 릴리스 페이지 또는 다음 절차를 참고하세요:

  1. Python 버전 확인: `python3 -c "import sys; print(sys.version)"`
  2. 적합한 wheel 찾기 (예: `https://github.com/google-coral/pycoral/releases` 또는 TensorFlow Lite 릴리스 아카이브)
  3. 다운로드 후 설치:

     pip install /path/to/tflite_runtime-*-cpXX-none-linux_armv7l.whl

  - wheel이 없으면 `pip install tensorflow` 로 대체할 수 있으나 용량과 의존성이 큽니다.

**UI / 원격 접속 참고**
- 로컬(모니터 직결)에서 실행하면 입력 지연이 거의 없습니다.
- XQuartz나 원격 X11 포워딩을 통해 화면을 전송하면 네트워크/창 관리자 지연이 발생할 수 있습니다. 이 경우 원격 GUI가 아닌 로컬에서 실행하거나 VNC/스크린캐스트 성능을 확인하세요.

**Troubleshooting**
- `ModuleNotFoundError: tflite_runtime`: `tflite_runtime`이 설치되지 않았습니다. 라즈베리파이용 바인딩을 설치하거나 `tensorflow`로 대체하세요.
- 카메라 프레임이 매우 느릴 경우: USB 카메라가 MJPG를 지원하는지 확인하고, 다른 캡포맷(예: YUYV)으로 시도해보세요.
- 모델 로딩이나 인터프리터 에러 발생 시 모델 파일 경로와 파일 권한을 확인하세요.

Remote display note
- X11 forwarding (XQuartz) 또는 other remote GUIs can add latency. For remote runs prefer:
  - SSH into the Pi and run locally with an attached display, or
  - Use VNC with good network bandwidth, or
  - Capture frames and stream a compressed video rather than forwarding raw X.

**Development Notes**
- 보드 그래픽은 정적 요소를 캐시하고, 토큰·HUD만 부분 갱신하여 렌더 비용을 최소화했습니다. 관련 함수: `get_static_board/_build_board_cache`, `draw_board`.
- 추론 파이프라인은 `inference_worker()` (multiprocessing) 내부에 있으며, `frame_q`/`result_q`로 프레임과 결과를 주고받습니다.

**Next Steps / Optional Improvements**
- 카메라 드라이버(MJPEG)에 문제가 있으면 `v4l2-ctl`로 포맷을 점검하세요.
- 성능 더 개선하려면 모델을 EdgeTPU용으로 변환하거나, OpenVINO/Coral 같은 하드웨어 가속 플랫폼을 사용하세요.

Files and structure
- `RPS_Racing_Game.py` : 메인 실행 파일
- `models/` : tflite 모델들을 넣으세요 (기본 코드에서 `./models/best_full_integer_quant.tflite` 사용 가능)
- `examples/EX_01_Image_Capture.py` : 카메라 테스트 스크립트
- `requirements.txt` : GitHub에서 받은 사용자가 `pip install -r requirements.txt` 로 설치할 수 있도록 추가했습니다.

---
작성자: 프로젝트 스캐폴드 자동 문서 (요약)