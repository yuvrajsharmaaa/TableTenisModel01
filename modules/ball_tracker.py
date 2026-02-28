"""
ball_tracker.py
──────────────────────────────────────────────────────────────────────────────
YOLOv8  +  ByteTrack (via supervision)  +  Kalman-filter for ball trajectory.

Combines:
  • ckjellson/tt_tracker  – Kalman prediction, bounce detection, turn detection,
                            velocity computation logic (ported from 3-D back to 2-D
                            with an optional Z-lift from stereo later)
  • wutonytt/…            – background-aware mask usage (adapted to YOLO detections)

GPU path: all inference runs on CUDA via ultralytics. Kalman is numpy-only so
it stays on CPU (negligible cost).
"""

from __future__ import annotations

import collections
import math
from typing import Deque, List, Optional, Tuple

import cv2
import numpy as np

try:
    from ultralytics import YOLO
    import supervision as sv
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
BALL_CLASS_ID   = 0   # YOLO class: ball
PLAYER_CLASS_ID = 1   # YOLO class: player

TABLE_WIDTH_M  = 2.74
TABLE_HEIGHT_M = 1.525
NET_HEIGHT_M   = 0.1525

# Pixel-to-metre scale:  resolved at runtime from detected table corners.
DEFAULT_PX_PER_M = 200.0          # fallback when table not yet calibrated

# Maximum inter-frame gap to still run Kalman forward prediction (frames)
MAX_PREDICTION_GAP = 8

# Rolling history for trajectory trail visual (frames)
TRAIL_LENGTH = 30

# Speed threshold to flag "fast ball" (m/s → ~200 km/h ≈ 55.6 m/s)
FAST_BALL_THRESHOLD_MS = 20.0     # 72 km/h in the 2-D projection; realistic limit


# ──────────────────────────────────────────────────────────────────────────────
# Tiny Kalman filter – 2D constant-acceleration model
# State  : [x, y, vx, vy, ax, ay]
# Measure: [x, y]
# ──────────────────────────────────────────────────────────────────────────────
class BallKalman:
    """Lightweight 2-D Kalman filter optimised for small fast objects."""

    def __init__(self, fps: float = 30.0):
        dt = 1.0 / fps
        # State-transition matrix  (constant-acceleration model)
        self.F = np.array([
            [1, 0, dt, 0, 0.5*dt**2, 0],
            [0, 1, 0, dt, 0, 0.5*dt**2],
            [0, 0, 1,  0, dt,         0],
            [0, 0, 0,  1,  0,        dt],
            [0, 0, 0,  0,  1,         0],
            [0, 0, 0,  0,  0,         1],
        ], dtype=np.float64)

        # Measurement matrix  (observe x, y only)
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Process noise covariance
        self.Q = np.eye(6, dtype=np.float64) * 0.5
        self.Q[4, 4] = 5.0   # higher uncertainty in acceleration
        self.Q[5, 5] = 5.0

        # Measurement noise covariance  (pixels)
        self.R = np.eye(2, dtype=np.float64) * 4.0

        # Posterior state & covariance
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 500.0
        self.initialized = False

    def init(self, cx: float, cy: float) -> None:
        self.x[:] = 0.0
        self.x[0, 0] = cx
        self.x[1, 0] = cy
        self.P = np.eye(6, dtype=np.float64) * 500.0
        self.initialized = True

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        z = np.array([[cx], [cy]], dtype=np.float64)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])

    @property
    def velocity_px(self) -> Tuple[float, float]:
        return float(self.x[2, 0]), float(self.x[3, 0])

    @property
    def accel_px(self) -> Tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])


# ──────────────────────────────────────────────────────────────────────────────
# BallTracker
# ──────────────────────────────────────────────────────────────────────────────
class BallTracker:
    """
    Main ball tracking class.

    Responsibilities
    ─────────────────
    1. Run YOLOv8 inference per frame (GPU).
    2. Feed detections into ByteTrack via supervision.
    3. Maintain Kalman filter for ball (single object – not multi).
    4. Estimate ball speed in km/h from pixel displacement + calibration.
    5. Detect bounces (local Z-minima ported to 2-D Y-reversal heuristic).
    6. Detect net-crossings (ball X midpoint).
    7. Keep rolling trajectory for visual trail.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",  # override with your custom weights
        fps: float = 30.0,
        px_per_metre: float = DEFAULT_PX_PER_M,
        conf_threshold: float = 0.35,
        device: str = "cuda",
        trail_length: int = TRAIL_LENGTH,
    ):
        self.fps = fps
        self.px_per_metre = px_per_metre
        self.conf_threshold = conf_threshold
        self.trail_length = trail_length
        self.device = device

        # YOLO model
        self._model: Optional[object] = None
        self._model_path = model_path

        # ByteTrack player tracker
        self._byte_tracker: Optional[object] = None

        # Ball Kalman filter
        self._kalman = BallKalman(fps=fps)
        self._gap_counter = 0          # frames since last ball detection

        # State
        self.ball_pos: Optional[Tuple[int, int]] = None            # smoothed (x,y)
        self.ball_raw: Optional[Tuple[int, int]] = None            # raw YOLO detection
        self.ball_conf: float = 0.0
        self.ball_speed_kmh: float = 0.0
        self.ball_trail: Deque[Tuple[int, int]] = collections.deque(maxlen=trail_length)
        self.player_boxes: List[np.ndarray] = []                   # [[x1,y1,x2,y2], …]
        self.player_ids: List[int] = []

        # Net X coordinate (pixel) – set externally or auto-detected
        self.net_x_px: Optional[int] = None

        # Bounce detection (ported from ckjellson – 2D Y reversal)
        self._prev_vy: float = 0.0
        self.bounce_detected: bool = False
        self.bounce_pos: Optional[Tuple[int, int]] = None
        self.bounce_count: int = 0

        # Net-crossing counters (centralelyon logic)
        self.net_cross_count: int = 0
        self._prev_side: Optional[int] = None   # -1=left, +1=right

        # Occlusion / prediction flag
        self.is_predicted: bool = False

        # Internal prev position for speed estimation
        self._prev_pos: Optional[Tuple[float, float]] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy model loading
    # ──────────────────────────────────────────────────────────────────────────
    def _ensure_model(self) -> None:
        if self._model is None:
            if not YOLO_AVAILABLE:
                raise RuntimeError("ultralytics / supervision not installed.")
            self._model = YOLO(self._model_path)
            self._model.to(self.device)
            self._byte_tracker = sv.ByteTrack()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────
    def update(self, frame: np.ndarray) -> dict:
        """
        Process one BGR frame.

        Returns
        ───────
        {
          'ball_pos'    : (x, y) or None,
          'ball_conf'   : float,
          'is_predicted': bool,          # True when Kalman-predicted (no detection)
          'speed_kmh'   : float,
          'is_fast_ball': bool,
          'bounce'      : bool,
          'bounce_pos'  : (x,y) or None,
          'net_cross'   : bool,
          'player_boxes': [[x1,y1,x2,y2], …],
          'player_ids'  : [int, …],
          'trail'       : [(x,y), …],
        }
        """
        self._ensure_model()
        self._reset_frame_flags()

        h, w = frame.shape[:2]

        # ── 1. YOLO inference ─────────────────────────────────────────────────
        results = self._model(
            frame,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )[0]

        ball_det: Optional[np.ndarray] = None
        player_dets: List[np.ndarray] = []

        if results.boxes is not None and len(results.boxes):
            boxes = results.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf   = float(boxes.conf[i].item())
                xyxy   = boxes.xyxy[i].cpu().numpy().astype(int)

                if cls_id == BALL_CLASS_ID:
                    if ball_det is None or conf > self.ball_conf:
                        ball_det = xyxy
                        self.ball_conf = conf
                elif cls_id == PLAYER_CLASS_ID:
                    player_dets.append(np.append(xyxy, conf))

        # ── 2. Player tracking via ByteTrack ──────────────────────────────────
        if player_dets:
            sv_dets = sv.Detections(
                xyxy=np.array([d[:4] for d in player_dets]),
                confidence=np.array([d[4] for d in player_dets]),
                class_id=np.full(len(player_dets), PLAYER_CLASS_ID, dtype=int),
            )
            tracked = self._byte_tracker.update_with_detections(sv_dets)
            self.player_boxes = tracked.xyxy.astype(int).tolist() if len(tracked) else []
            self.player_ids   = tracked.tracker_id.tolist() if len(tracked) and tracked.tracker_id is not None else []
        else:
            self.player_boxes = []
            self.player_ids   = []

        # ── 3. Ball Kalman tracking ───────────────────────────────────────────
        if ball_det is not None:
            cx = int((ball_det[0] + ball_det[2]) / 2)
            cy = int((ball_det[1] + ball_det[3]) / 2)
            self.ball_raw = (cx, cy)

            if not self._kalman.initialized:
                self._kalman.init(cx, cy)
                sx, sy = float(cx), float(cy)
            else:
                self._kalman.predict()
                sx, sy = self._kalman.update(cx, cy)

            self._gap_counter = 0
            self.is_predicted = False
        else:
            # Occluded / not detected – predict if within gap allowance
            if self._kalman.initialized and self._gap_counter < MAX_PREDICTION_GAP:
                sx, sy = self._kalman.predict()
                self._gap_counter += 1
                self.is_predicted = True
                self.ball_conf = max(0.0, self.ball_conf - 0.1)
            else:
                self.ball_pos = None
                self._gap_counter += 1
                return self._build_result()

        self.ball_pos = (int(sx), int(sy))
        self.ball_trail.append(self.ball_pos)

        # ── 4. Speed estimation ───────────────────────────────────────────────
        vx_px, vy_px = self._kalman.velocity_px
        speed_px_per_frame = math.hypot(vx_px, vy_px)
        speed_ms = speed_px_per_frame * self.fps / self.px_per_metre
        self.ball_speed_kmh = speed_ms * 3.6

        # ── 5. Bounce detection (ported from ckjellson find_bounces logic) ────
        # In 2-D, a bounce = Y-velocity sign reversal (upward → downward flip)
        cur_vy = vy_px
        if (self._prev_vy < -2.0 and cur_vy > 2.0):   # was going up, now down
            self.bounce_detected = True
            self.bounce_pos = self.ball_pos
            self.bounce_count += 1
        self._prev_vy = cur_vy

        # ── 6. Net-crossing detection (centralelyon score logic) ──────────────
        if self.net_x_px is not None:
            cur_side = 1 if self.ball_pos[0] > self.net_x_px else -1
            if self._prev_side is not None and cur_side != self._prev_side:
                self.net_cross_count += 1
                self._net_cross_this_frame = True
            self._prev_side = cur_side

        self._prev_pos = (sx, sy)
        return self._build_result()

    # ──────────────────────────────────────────────────────────────────────────
    def calibrate_from_table_corners(
        self, corners: List[Tuple[int, int]], frame_shape: Tuple[int, int]
    ) -> None:
        """
        Derive px_per_metre and net_x_px from detected table corners.
        corners: [TL, TR, BR, BL] in pixel space.
        """
        if len(corners) < 4:
            return
        tl, tr, br, bl = corners[:4]
        table_px_w = math.hypot(tr[0]-tl[0], tr[1]-tl[1])
        self.px_per_metre = table_px_w / TABLE_WIDTH_M
        self.net_x_px = int((tl[0] + bl[0]) / 2 + (tr[0] + br[0]) / 2) // 2

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _reset_frame_flags(self) -> None:
        self.bounce_detected        = False
        self.bounce_pos             = None
        self._net_cross_this_frame  = False

    def _build_result(self) -> dict:
        return {
            "ball_pos"    : self.ball_pos,
            "ball_conf"   : round(self.ball_conf, 3),
            "is_predicted": self.is_predicted,
            "speed_kmh"   : round(self.ball_speed_kmh, 1),
            "is_fast_ball": self.ball_speed_kmh > FAST_BALL_THRESHOLD_MS * 3.6,
            "bounce"      : self.bounce_detected,
            "bounce_pos"  : self.bounce_pos,
            "net_cross"   : getattr(self, "_net_cross_this_frame", False),
            "player_boxes": self.player_boxes,
            "player_ids"  : self.player_ids,
            "trail"       : list(self.ball_trail),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Reset for new video / match."""
        self._kalman = BallKalman(fps=self.fps)
        self._gap_counter = 0
        self.ball_pos     = None
        self.ball_raw     = None
        self.ball_conf    = 0.0
        self.ball_speed_kmh = 0.0
        self.ball_trail.clear()
        self.player_boxes = []
        self.player_ids   = []
        self._prev_vy     = 0.0
        self.bounce_count = 0
        self.net_cross_count = 0
        self._prev_side   = None
        self._prev_pos    = None
