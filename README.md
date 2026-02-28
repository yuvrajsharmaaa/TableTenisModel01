# Table Tennis Analytics — Streamlit App

Real-time table-tennis video analytics built by combining three research repositories:

| Repo | Contribution |
|------|-------------|
| [wutonytt/Camera-Based-Table-Tennis-Posture-Analysis](https://github.com/wutonytt/Camera-Based-Table-Tennis-Posture-Analysis) | Pose analysis · stroke classification (forehand/backhand) |
| [ckjellson/tt_tracker](https://github.com/ckjellson/tt_tracker) | Ball tracking · Kalman filter · bounce & net-cross detection |
| [centralelyon/table-tennis-analytics](https://github.com/centralelyon/table-tennis-analytics) | Score probability · domination · expected score · stress model |

---

## Architecture

```
app.py  (Streamlit UI)
│
└── modules/tt_analyzer.py   ← tt_analyzer(frame) entry point
        ├── modules/ball_tracker.py    YOLOv8 + ByteTrack + Kalman
        ├── modules/pose_analyzer.py   MediaPipe Pose Lite per player
        ├── modules/score_engine.py    Score probability + domination
        └── modules/annotator.py       OpenCV HUD rendering
                └── utils/geometry.py  Court homography + minimap
```

### Data flow (per frame)

```
video frame (BGR ndarray)
    │
    ▼
BallTracker.update()
    → YOLO detects ball + players
    → ByteTrack assigns player IDs
    → Kalman filter predicts ball when occluded
    → bounce / net-cross events fired
    │
    ▼
PoseAnalyzer.update()
    → MediaPipe Pose Lite on each player ROI crop
    → Rule-based forehand / backhand classification
    │
    ▼
ScoreEngine.tick()
    → on_net_cross() → candidate point
    → on_point_scored() → recursive win-prob update
    → Domination = 0.40·score + 0.30·(-stress) + 0.30·physical
    │
    ▼
Annotator.annotate()
    → player boxes + IDs
    → skeleton overlay
    → ball trail (last 30 frames)
    → HUD: score, speed, domination bar, xScore
    → minimap embed (court top-down)
    │
    ▼
(annotated_frame, analytics_dict)
```

---

## Project structure

```
tt/
├── app.py                      Streamlit entry point
├── requirements.txt
├── models/
│   └── yolov8_tt.pt            ← place custom YOLOv8 weights here
├── modules/
│   ├── __init__.py
│   ├── ball_tracker.py
│   ├── pose_analyzer.py
│   ├── score_engine.py
│   ├── annotator.py
│   └── tt_analyzer.py
└── utils/
    ├── __init__.py
    └── geometry.py
```

---

## Installation

### 1 — Clone

```bash
git clone <this-repo>
cd tt
```

### 2 — Python environment (Python ≥ 3.10 recommended)

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3 — Install CUDA PyTorch first (RTX 3070 / CUDA 12.x)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## YOLOv8 weights

The tracker expects a YOLOv8 model with **two classes**:

| Class ID | Name   |
|----------|--------|
| 0        | ball   |
| 1        | player |

### Option A — use a pre-trained generic model (quick start)

```python
# in app.py sidebar, set model path to:
yolov8n.pt   # nano — downloads automatically from ultralytics hub
```

Player detection will work; ball detection accuracy depends on the generic model.

### Option B — fine-tune on TT data (recommended)

1. Label table-tennis footage with [Roboflow](https://roboflow.com) (classes: `ball`, `player`).
2. Export in YOLOv8 format.
3. Fine-tune:

```bash
yolo detect train data=tt.yaml model=yolov8n.pt epochs=100 imgsz=1280 device=0
```

4. Copy `runs/detect/train/weights/best.pt` → `models/yolov8_tt.pt`.

---

## Running the app

```bash
source .venv/bin/activate
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### Sidebar controls

| Control | Description |
|---------|-------------|
| Model path | Path or URL to YOLOv8 `.pt` file |
| Device | `cuda:0` (RTX 3070) or `cpu` |
| Confidence threshold | YOLO detection threshold (default 0.35) |
| Input source | Upload video file or live webcam |
| Manual scoring | Override automatic score detection |
| Reset | Clear all state and restart |

---

## Programmatic API

```python
from modules.tt_analyzer import tt_analyzer
import cv2

cap = cv2.VideoCapture("match.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    annotated, info = tt_analyzer(
        frame,
        model_path="models/yolov8_tt.pt",
        fps=30,
        device="cuda:0",
    )
    # info keys: ball_pos, speed, player_keypoints, score,
    #             domination, x_score_a, x_score_b, stress_diff, ...
    cv2.imshow("TT Analytics", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

---

## Analytics dictionary

```python
{
    "ball_pos":        (x, y),          # pixel coords, None if not detected
    "speed":           float,           # km/h
    "player_keypoints": [               # one dict per tracked player
        {
            "player_id": int,
            "keypoints": [(x,y,visibility), ...],  # 33 MediaPipe landmarks
            "stroke":    "forehand" | "backhand" | "neutral",
            "fore_ratio": float,        # forehand / (forehand + backhand)
        }
    ],
    "score":           int,             # current total points scored (player A)
    "score_a":         int,
    "score_b":         int,
    "set_score_a":     int,
    "set_score_b":     int,
    "domination":      float,           # ∈ [-1, 1]
    "x_score_a":       float,           # expected-score probability A
    "x_score_b":       float,
    "stress_diff":     float,
    "net_crosses":     int,             # rally length
}
```

---

## Performance notes

| Component | Device | Typical latency (1080p) |
|-----------|--------|------------------------|
| YOLOv8-nano detection | RTX 3070 (CUDA) | ~5 ms |
| ByteTrack | CPU | ~1 ms |
| Kalman filter | CPU | < 0.1 ms |
| MediaPipe Pose (per player, lite) | CPU | ~4 ms |
| Score engine | CPU | < 0.1 ms |
| Annotation | CPU | ~2 ms |
| **Total** | | **~12 ms → ~80 FPS** |

Pose is computed every 3 frames by default (`pose_every=3`) to stay comfortably above 30 FPS even with two players on CPU pose.

---

## Credits

- **Posture analysis pipeline** — [wutonytt/Camera-Based-Table-Tennis-Posture-Analysis](https://github.com/wutonytt/Camera-Based-Table-Tennis-Posture-Analysis)
- **Ball tracking & Kalman filter** — [ckjellson/tt_tracker](https://github.com/ckjellson/tt_tracker)
- **Score probability & domination models** — [centralelyon/table-tennis-analytics](https://github.com/centralelyon/table-tennis-analytics)
