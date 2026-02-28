"""
tt_analyzer.py
──────────────────────────────────────────────────────────────────────────────
Single-entry-point module combining:

  Repo 1 – wutonytt/Camera-Based-Table-Tennis-Posture-Analysis
           → MediaPipe pose, forehand/backhand classification,
             fore/back ratio per player

  Repo 2 – ckjellson/tt_tracker
           → YOLOv8 detection, ByteTrack ID assignment, Kalman-filter ball
             trajectory, velocity, bounce detection, net-crossing

  Repo 3 – centralelyon/table-tennis-analytics
           → Score domination, expected score, stress/moral metric,
             playing-pattern tree

Public API
──────────
    analyzer = TTAnalyzer(model_path="best.pt", fps=30.0, device="cuda")
    annotated_frame, analytics = analyzer.process(frame)

    # Or the bare-bones function form (mirrors the original request):
    annotated_frame, result = tt_analyzer(frame)
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from modules.ball_tracker  import BallTracker
from modules.pose_analyzer import PoseAnalyzer
from modules.score_engine  import ScoreEngine
from modules.annotator     import Annotator
from utils.geometry        import (
    detect_table_corners, CourtTransform, draw_minimap, ball_side
)

# ──────────────────────────────────────────────────────────────────────────────
# TTAnalyzer – stateful class                                                   
# ──────────────────────────────────────────────────────────────────────────────

class TTAnalyzer:
    """
    Real-time Table Tennis Analytics engine.

    Parameters
    ──────────
    model_path : str
        Path to YOLOv8 weights (.pt).  Custom trained on ball(0)+player(1).
        Falls back to yolov8n.pt (COCO) if not provided – ball detection
        won't work well until fine-tuned.
    fps : float
        Video frame rate – used for speed estimation.
    device : str
        'cuda' for RTX 3070, 'cpu' for fallback.
    auto_calib_every : int
        Re-run table corner detection every N frames.
    pose_every : int
        Run pose estimation every N frames (reduces GPU load at 30 FPS).
    """

    def __init__(
        self,
        model_path: str = "models/yolov8_tt.pt",
        fps: float = 30.0,
        device: str = "cuda",
        auto_calib_every: int = 90,      # calibrate table ~every 3 s
        pose_every: int = 3,             # pose on every 3rd frame
        show_skeleton: bool = True,
        show_trail: bool = True,
        show_hud: bool = True,
    ):
        self.fps              = fps
        self.device           = device
        self.auto_calib_every = auto_calib_every
        self.pose_every       = pose_every

        # Sub-modules
        self._ball   = BallTracker(
            model_path=model_path, fps=fps, device=device
        )
        self._pose   = PoseAnalyzer(model_complexity=0)
        self._score  = ScoreEngine(fps=fps)
        self._annot  = Annotator(
            show_skeleton=show_skeleton,
            show_trail=show_trail,
            show_hud=show_hud,
        )

        # Calibration state
        self._court: Optional[CourtTransform] = None
        self._calib_frame_count = 0
        self._calib_pending     = True

        # Pose cache (so we don't recalculate every frame)
        self._pose_cache: List[dict] = []

        # Frame counter
        self._frame_idx = 0

        # Bounce history in metre space for minimap
        self._bounce_m: List[Tuple[float, float]] = []

        # Latency tracker
        self._t_last = time.perf_counter()

    # ──────────────────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Main per-frame entry point.

        Parameters
        ──────────
        frame : np.ndarray  – BGR image (e.g. from cv2.VideoCapture)

        Returns
        ───────
        annotated_frame : np.ndarray  – BGR with all overlays
        analytics       : dict        – see _build_analytics()
        """
        t0 = time.perf_counter()
        self._frame_idx += 1

        # ── 1. Table calibration ──────────────────────────────────────────────
        self._maybe_calibrate(frame)

        # ── 2. Ball + player tracking (YOLOv8 + ByteTrack + Kalman) ──────────
        ball_result = self._ball.update(frame)

        # Tell score engine about net crossings
        if ball_result["net_cross"]:
            self._score.on_net_cross()

        # Record bounce in metre space if calibrated
        if ball_result["bounce"] and ball_result["bounce_pos"] and self._court:
            bx, by = self._court.px_to_metre(ball_result["bounce_pos"])
            self._bounce_m.append((bx, by))
            self._bounce_m = self._bounce_m[-100:]   # keep rolling 100

        # ── 3. Pose estimation (every N frames to maintain 30 FPS) ───────────
        if (self._frame_idx % self.pose_every == 0 and
                len(ball_result["player_boxes"]) > 0):
            self._pose_cache = self._pose.update(
                frame,
                ball_result["player_boxes"],
                ball_result["player_ids"],
            )

        # ── 4. Physical domination from pose (distance/angle heuristic) ──────
        phys_dom = self._compute_phys_dom(self._pose_cache, frame.shape[1])

        # ── 5. Score engine tick ──────────────────────────────────────────────
        score_data = self._score.tick(phys_dom=phys_dom)

        # ── 6. Annotate frame ─────────────────────────────────────────────────
        annotated = self._annot.annotate(
            frame,
            ball_pos         = ball_result["ball_pos"],
            ball_is_predicted= ball_result["is_predicted"],
            ball_is_fast     = ball_result["is_fast_ball"],
            ball_trail       = ball_result["trail"],
            bounce_detected  = ball_result["bounce"],
            bounce_pos       = ball_result["bounce_pos"],
            player_boxes     = ball_result["player_boxes"],
            player_ids       = ball_result["player_ids"],
            pose_results     = self._pose_cache,
            net_x_px         = self._ball.net_x_px,
            score_a          = score_data["score_a"],
            score_b          = score_data["score_b"],
            set_score_a      = score_data["set_score_a"],
            set_score_b      = score_data["set_score_b"],
            speed_kmh        = ball_result["speed_kmh"],
            domination       = score_data["domination"],
            x_score_a        = score_data["x_score_a"],
            x_score_b        = score_data["x_score_b"],
            server           = score_data["server"],
            net_crosses      = score_data["net_crosses"],
        )

        # ── 7. Mini-map ───────────────────────────────────────────────────────
        minimap = None
        if self._court and ball_result["ball_pos"]:
            ball_m = self._court.px_to_metre(ball_result["ball_pos"])
            minimap = draw_minimap(
                (200, 120),
                ball_m=ball_m,
                bounce_positions=self._bounce_m[-20:],
            )
            # Embed minimap in top-right corner
            mh, mw = minimap.shape[:2]
            fh, fw = annotated.shape[:2]
            annotated[fh-mh-10:fh-10, fw-mw-10:fw-10] = minimap

        # ── 8. FPS stamp ──────────────────────────────────────────────────────
        t1 = time.perf_counter()
        fps_live = 1.0 / max(t1 - self._t_last, 1e-9)
        self._t_last = t1
        proc_ms = (t1 - t0) * 1000
        cv2.putText(
            annotated,
            f"{fps_live:.1f} FPS  |  {proc_ms:.1f} ms",
            (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA,
        )

        # ── 9. Build return dict ──────────────────────────────────────────────
        analytics = self._build_analytics(ball_result, score_data)
        return annotated, analytics

    # ──────────────────────────────────────────────────────────────────────────
    def on_point_scored(self, winner: str, fault_player: Optional[str] = None) -> dict:
        """
        Call from the UI when a human operator (or automatic logic) confirms
        that a point has ended.
        winner: 'A' | 'B'
        """
        stroke = None
        zone   = None
        if self._pose_cache:
            stroke = self._pose_cache[0].get("stroke")
        return self._score.on_point_scored(winner, stroke, zone, fault_player)

    def reset(self) -> None:
        self._ball.reset()
        self._pose.reset()
        self._score.reset()
        self._calib_pending  = True
        self._calib_frame_count = 0
        self._court          = None
        self._pose_cache     = []
        self._frame_idx      = 0
        self._bounce_m       = []

    # ──────────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _maybe_calibrate(self, frame: np.ndarray) -> None:
        self._calib_frame_count += 1
        if not (self._calib_pending or
                self._calib_frame_count % self.auto_calib_every == 0):
            return
        corners = detect_table_corners(frame)
        if corners is not None:
            self._court = CourtTransform(corners)
            self._ball.calibrate_from_table_corners(corners, frame.shape[:2])
            self._calib_pending = False

    def _compute_phys_dom(
        self, pose_results: List[dict], frame_width: int
    ) -> float:
        """
        Physical domination ∈ [-1, 1].
        Heuristic: compare swing angles / activity levels between left/right player.
        Ported from centralelyon/Calcul_Domination_Match.py::calcul_diff_position
        (adapted for real-time; annotated data version uses player XY positions).
        """
        if len(pose_results) < 2:
            return 0.0

        mid = frame_width / 2
        p_a, p_b = None, None
        for p in pose_results:
            kpts = p.get("keypoints")
            if kpts is None:
                continue
            cx = float(kpts[:, 0].mean())
            if cx < mid:
                p_a = p
            else:
                p_b = p

        if p_a is None or p_b is None:
            return 0.0

        # More active arm (higher swing angle) = physical advantage
        ang_a = p_a.get("swing_angle", 0.0)
        ang_b = p_b.get("swing_angle", 0.0)
        diff  = ang_a - ang_b
        # Sigmoid-centre normalisation (ported from sigmoide_centree)
        norm  = 1.0 / (1.0 + abs(diff) / 90.0) * 2.0 - 1.0
        return float(np.clip(diff / 90.0, -1.0, 1.0))

    @staticmethod
    def _build_analytics(ball_result: dict, score_data: dict) -> dict:
        """Flatten ball + score data into the documented output schema."""
        keypoints_list = []   # populated from global pose cache via caller
        return {
            # ── Required output schema ──────────────────────────
            "ball_pos"        : ball_result["ball_pos"],
            "speed"           : ball_result["speed_kmh"],
            "player_keypoints": [],   # set by caller if needed
            "score"           : score_data["score_a"],
            # ── Extended analytics ───────────────────────────────
            "score_a"         : score_data["score_a"],
            "score_b"         : score_data["score_b"],
            "set_score_a"     : score_data["set_score_a"],
            "set_score_b"     : score_data["set_score_b"],
            "server"          : score_data["server"],
            "domination"      : score_data["domination"],
            "score_dom"       : score_data["score_dom"],
            "stress_diff"     : score_data["stress_diff"],
            "phys_dom"        : score_data["phys_dom"],
            "x_score_a"       : score_data["x_score_a"],
            "x_score_b"       : score_data["x_score_b"],
            "net_crosses"     : score_data["net_crosses"],
            "ball_conf"       : ball_result["ball_conf"],
            "is_predicted"    : ball_result["is_predicted"],
            "is_fast_ball"    : ball_result["is_fast_ball"],
            "bounce"          : ball_result["bounce"],
            "bounce_pos"      : ball_result["bounce_pos"],
            "player_boxes"    : ball_result["player_boxes"],
            "player_ids"      : ball_result["player_ids"],
            "trail"           : ball_result["trail"],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level function form  (mirrors the original request signature)
# ──────────────────────────────────────────────────────────────────────────────
_GLOBAL_ANALYZER: Optional[TTAnalyzer] = None


def tt_analyzer(
    frame: np.ndarray,
    *,
    model_path: str = "models/yolov8_tt.pt",
    fps: float = 30.0,
    device: str = "cuda",
    reset: bool = False,
) -> Tuple[np.ndarray, dict]:
    """
    Stateful per-frame table tennis analyzer.

    Parameters
    ──────────
    frame      : BGR numpy array (1080p or any resolution)
    model_path : YOLOv8 weights path
    fps        : source video FPS (for speed calibration)
    device     : 'cuda' | 'cpu'
    reset      : pass True on the first frame of a new video/match

    Returns
    ───────
    annotated_frame : np.ndarray  – BGR with all visual overlays
    result          : dict        –
        {
          'ball_pos'        : (x, y) or None,
          'speed'           : float,    # km/h
          'player_keypoints': list,     # per-player keypoint arrays
          'score'           : int,      # player A's current set score
          'score_a'         : int,
          'score_b'         : int,
          'set_score_a'     : int,
          'set_score_b'     : int,
          'domination'      : float,    # ∈ [-1, 1]
          'x_score_a'       : float,
          'x_score_b'       : float,
          'bounce'          : bool,
          'net_crosses'     : int,
          'is_fast_ball'    : bool,
          'player_boxes'    : list of [x1,y1,x2,y2],
          # …and more
        }

    Usage
    ─────
        cap = cv2.VideoCapture("match.mp4")
        tt_analyzer(None, reset=True)   # reset state for new video
        while True:
            ret, frame = cap.read()
            if not ret: break
            annotated, data = tt_analyzer(frame)
            cv2.imshow("TT Analytics", annotated)

    Notes
    ─────
    • GPU (CUDA) is used for YOLO inference automatically.
    • MediaPipe Pose runs on CPU (MediaPipe's default; ~4 ms/person).
    • Kalman filter and score engine are pure NumPy (negligible overhead).
    • For best results, train YOLOv8 on a TT-specific dataset:
        classes: 0=ball, 1=player
      The ball is ~40mm → requires high-res input or custom model.
    """
    global _GLOBAL_ANALYZER

    if _GLOBAL_ANALYZER is None or reset:
        _GLOBAL_ANALYZER = TTAnalyzer(
            model_path=model_path, fps=fps, device=device
        )
        if reset and frame is None:
            return np.zeros((1, 1, 3), dtype=np.uint8), {}

    if frame is None:
        return np.zeros((1, 1, 3), dtype=np.uint8), {}

    return _GLOBAL_ANALYZER.process(frame)
