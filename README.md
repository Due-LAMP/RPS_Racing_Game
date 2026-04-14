# ✌️ RPS Racing Game
간단한 2인용 가위·바위·보 레이싱 보드게임입니다. 실시간 손 제스처(왼손=P1, 오른손=P2)를 YOLO TFLite 모델로 인식하여 턴마다 이동합니다.

### ⚙️ Requirements
* **Python**: 3.8+ 권장
* **Packages**: `opencv-python`, `numpy`, `tflite_runtime` (또는 `tensorflow` + `lite`)

### 📁 Structure

```text
📦 Project
├── 📜 RPS_Racing_Game.py              # 메인 게임 실행 파일
├── 📜 requirements.txt                # 패키지 설치 파일
├── 📂 models/                         # TFLite 모델 보관
└── 📂 examples/EX_01_Image_Capture.py # 캡처 테스트 스크립트
```

🚀 Quick Start

Bash
```
git clone <your-repo-url>
cd 05_Object_Detection_Based_On-Device_AI
python3 -m venv env && source env/bin/activate
pip install -r requirements.txt
python3 RPS_Racing_Game.py
```


🍓 Raspberry Pi 성능 팁 & ⚠️ 트러블슈팅 (클릭)

성능 최적화: 가장 빠른 CPU 추론을 위해 INT8 양자화 모델을 사용하세요.
에러 해결: ModuleNotFoundError: tflite_runtime 발생 시, Pi 버전에 맞는 .whl 파일로 설치하거나 tensorflow로 대체해야 합니다.


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
