"""
annotator.py
──────────────────────────────────────────────────────────────────────────────
Pure-OpenCV frame annotation utilities.

Draws:
  • Player bounding boxes  (colour-coded by stroke label + player ID)
  • Pose skeleton overlay  (MediaPipe connections)
  • Ball position circle   (filled/outlined, yellow predicted)
  • Ball trajectory trail  (fading)
  • Bounce flash indicator
  • Net line (if calibrated)
  • HUD: score, domination bar, speed, stroke label, fore/back ratio
"""

from __future__ import annotations

import colorsys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Colour palette
# ──────────────────────────────────────────────────────────────────────────────
_COLOUR = {
    "ball_detect"  : (0, 255, 255),     # cyan
    "ball_predict" : (0, 165, 255),     # orange
    "ball_fast"    : (0, 0, 255),       # red
    "bounce"       : (255, 255,   0),   # yellow
    "net"          : (255, 255, 255),   # white
    "player_a"     : (100, 220, 100),   # green-ish
    "player_b"     : (100, 100, 220),   # blue-ish
    "forehand"     : (0, 220, 80),      # bright green
    "backhand"     : (220, 80,   0),    # bright orange
    "neutral"      : (180, 180, 180),   # grey
    "hud_bg"       : (20, 20, 20),      # near-black
    "dom_pos"      : (80, 200, 80),     # positive domination
    "dom_neg"      : (80, 80, 200),     # negative domination
    "text"         : (240, 240, 240),
    "text_hi"      : (0, 255, 150),
}

# MediaPipe Pose connections subset (arms, torso, legs)
_MP_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]


def _player_colour(pid: int | None) -> Tuple[int, int, int]:
    if pid is None or pid % 2 == 0:
        return _COLOUR["player_a"]
    return _COLOUR["player_b"]


def _stroke_colour(stroke: str) -> Tuple[int, int, int]:
    return _COLOUR.get(stroke, _COLOUR["neutral"])


def _fade_colour(base: Tuple[int, int, int], alpha: float) -> Tuple[int, int, int]:
    return tuple(int(c * alpha) for c in base)


# ──────────────────────────────────────────────────────────────────────────────
# Annotator class
# ──────────────────────────────────────────────────────────────────────────────
class Annotator:
    """
    Stateless-ish frame annotator.  All draw calls operate on an in-place copy
    or on a provided frame buffer.  Call ``annotate()`` to get the fully
    annotated frame back.
    """

    def __init__(
        self,
        line_width: int = 2,
        font_scale: float = 0.55,
        hud_alpha: float = 0.65,
        show_skeleton: bool = True,
        show_trail: bool = True,
        show_hud: bool = True,
    ):
        self.line_width    = line_width
        self.font_scale    = font_scale
        self.hud_alpha     = hud_alpha
        self.show_skeleton = show_skeleton
        self.show_trail    = show_trail
        self.show_hud      = show_hud

        self._font = cv2.FONT_HERSHEY_SIMPLEX

    # ──────────────────────────────────────────────────────────────────────────
    def annotate(
        self,
        frame: np.ndarray,
        *,
        # Ball
        ball_pos: Optional[Tuple[int, int]] = None,
        ball_is_predicted: bool = False,
        ball_is_fast: bool = False,
        ball_trail: Optional[List[Tuple[int, int]]] = None,
        bounce_detected: bool = False,
        bounce_pos: Optional[Tuple[int, int]] = None,
        # Players
        player_boxes: Optional[List[List[int]]] = None,
        player_ids: Optional[List[int]] = None,
        pose_results: Optional[List[dict]] = None,  # from PoseAnalyzer
        # Court
        net_x_px: Optional[int] = None,
        # Score / analytics
        score_a: int = 0,
        score_b: int = 0,
        set_score_a: int = 0,
        set_score_b: int = 0,
        speed_kmh: float = 0.0,
        domination: float = 0.0,
        x_score_a: float = 0.0,
        x_score_b: float = 0.0,
        server: str = "A",
        net_crosses: int = 0,
    ) -> np.ndarray:
        """
        Master annotation call.  Returns the annotated frame (writes in-place).
        """
        out = frame.copy()
        h, w = out.shape[:2]

        # ── Net line ─────────────────────────────────────────────────────────
        if net_x_px is not None:
            cv2.line(out, (net_x_px, 0), (net_x_px, h),
                     _COLOUR["net"], 1, cv2.LINE_AA)

        # ── Ball trail ───────────────────────────────────────────────────────
        if self.show_trail and ball_trail:
            self._draw_trail(out, ball_trail, ball_is_fast)

        # ── Player boxes + labels ────────────────────────────────────────────
        if player_boxes:
            for i, box in enumerate(player_boxes):
                pid   = player_ids[i] if player_ids and i < len(player_ids) else i
                pose  = pose_results[i] if pose_results and i < len(pose_results) else None
                self._draw_player(out, box, pid, pose)

        # ── Ball ─────────────────────────────────────────────────────────────
        if ball_pos is not None:
            self._draw_ball(out, ball_pos, ball_is_predicted, ball_is_fast, bounce_detected)

        # ── Bounce flash ─────────────────────────────────────────────────────
        if bounce_detected and bounce_pos is not None:
            cv2.circle(out, bounce_pos, 18, _COLOUR["bounce"], 2, cv2.LINE_AA)
            cv2.putText(out, "BOUNCE", (bounce_pos[0]+20, bounce_pos[1]),
                        self._font, 0.5, _COLOUR["bounce"], 1, cv2.LINE_AA)

        # ── HUD overlay ──────────────────────────────────────────────────────
        if self.show_hud:
            self._draw_hud(
                out, score_a, score_b, set_score_a, set_score_b,
                speed_kmh, domination, x_score_a, x_score_b,
                server, net_crosses
            )

        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _draw_trail(
        self,
        out: np.ndarray,
        trail: List[Tuple[int, int]],
        is_fast: bool,
    ) -> None:
        n = len(trail)
        base_col = _COLOUR["ball_fast"] if is_fast else _COLOUR["ball_detect"]
        for i in range(1, n):
            alpha  = i / n
            colour = _fade_colour(base_col, alpha)
            r      = max(1, int(4 * alpha))
            cv2.line(out, trail[i-1], trail[i], colour, r, cv2.LINE_AA)

    def _draw_ball(
        self,
        out: np.ndarray,
        pos: Tuple[int, int],
        predicted: bool,
        fast: bool,
        bounce: bool,
    ) -> None:
        col = _COLOUR["ball_fast"] if fast else (
              _COLOUR["ball_predict"] if predicted else _COLOUR["ball_detect"])
        cv2.circle(out, pos, 8, col, -1, cv2.LINE_AA)
        cv2.circle(out, pos, 8, (255, 255, 255), 1, cv2.LINE_AA)   # white ring
        if predicted:
            cv2.putText(out, "KF", (pos[0]+10, pos[1]-8),
                        self._font, 0.4, col, 1, cv2.LINE_AA)

    def _draw_player(
        self,
        out: np.ndarray,
        box: List[int],
        pid: int,
        pose: Optional[dict],
    ) -> None:
        x1, y1, x2, y2 = [int(v) for v in box]
        pcol   = _player_colour(pid)
        stroke = pose["stroke"] if pose else "neutral"
        scol   = _stroke_colour(stroke)

        # Box
        cv2.rectangle(out, (x1, y1), (x2, y2), pcol, self.line_width, cv2.LINE_AA)

        # Label bar at top of box
        label = f"P{pid} {stroke.upper()[:2]}"
        (tw, th), _ = cv2.getTextSize(label, self._font, self.font_scale, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+6, y1), scol, -1)
        cv2.putText(out, label, (x1+3, y1-4), self._font, self.font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA)

        if pose is None:
            return

        # Swing angle indicator
        ang_text = f"{pose['swing_angle']:.0f}°"
        cv2.putText(out, ang_text, (x1+3, y2-8),
                    self._font, 0.45, _COLOUR["text"], 1, cv2.LINE_AA)

        # Fore/back ratio bar (mini-bar at bottom of box)
        bw = x2 - x1
        fr = pose.get("fore_ratio", 0.0)
        cv2.rectangle(out, (x1, y2+2), (x1+bw, y2+8), (60,60,60), -1)
        cv2.rectangle(out, (x1, y2+2), (x1+int(bw*fr), y2+8),
                      _COLOUR["forehand"], -1)

        # Skeleton
        if self.show_skeleton and pose.get("keypoints") is not None:
            self._draw_skeleton(out, pose["keypoints"], pose.get("visibility"))

    def _draw_skeleton(
        self,
        out: np.ndarray,
        kpts: np.ndarray,
        vis: Optional[np.ndarray],
    ) -> None:
        for (i, j) in _MP_CONNECTIONS:
            if vis is not None and (vis[i] < 0.4 or vis[j] < 0.4):
                continue
            p1 = (int(kpts[i, 0]), int(kpts[i, 1]))
            p2 = (int(kpts[j, 0]), int(kpts[j, 1]))
            cv2.line(out, p1, p2, (0, 200, 200), 1, cv2.LINE_AA)
        # Keypoint dots
        for i, (x, y) in enumerate(kpts):
            if vis is not None and vis[i] < 0.4:
                continue
            cv2.circle(out, (int(x), int(y)), 3, (255, 255, 0), -1, cv2.LINE_AA)

    def _draw_hud(
        self,
        out: np.ndarray,
        score_a: int,
        score_b: int,
        set_a: int,
        set_b: int,
        speed: float,
        dom: float,
        xs_a: float,
        xs_b: float,
        server: str,
        net_x: int,
    ) -> None:
        h, w = out.shape[:2]
        hud_h = 80
        overlay = out[:hud_h, :w].copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), _COLOUR["hud_bg"], -1)
        cv2.addWeighted(overlay, self.hud_alpha, out[:hud_h, :w],
                        1 - self.hud_alpha, 0, out[:hud_h, :w])

        fn, fs, fc = self._font, self.font_scale * 1.1, _COLOUR["text"]

        # ── Score ──
        score_str = f"A  {score_a} : {score_b}  B"
        cv2.putText(out, score_str, (w//2 - 70, 28), fn, 1.0, _COLOUR["text_hi"], 2, cv2.LINE_AA)
        cv2.putText(out, f"Sets  {set_a}-{set_b}", (w//2 - 35, 52), fn, 0.55, fc, 1, cv2.LINE_AA)

        # Server indicator
        sv_x = 30 if server == "A" else w-60
        cv2.putText(out, "SRV", (sv_x, 28), fn, 0.5, (255,200,0), 1, cv2.LINE_AA)

        # ── Speed ──
        spd_col = _COLOUR["ball_fast"] if speed > 72 else fc
        cv2.putText(out, f"{speed:.0f} km/h", (w-130, 28), fn, 0.7, spd_col, 2, cv2.LINE_AA)
        cv2.putText(out, f"xGA:{xs_a:.1f}  xGB:{xs_b:.1f}", (w-145, 55), fn, 0.45, fc, 1)

        # ── Domination bar ────────────────────────────────────────────────────
        bar_x, bar_y, bar_w, bar_h2 = 20, 60, 240, 12
        cv2.rectangle(out, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h2), (60,60,60), -1)
        mid = bar_x + bar_w // 2
        dom_px = int(dom * (bar_w / 2))
        if dom >= 0:
            cv2.rectangle(out, (mid, bar_y), (mid+dom_px, bar_y+bar_h2),
                          _COLOUR["dom_pos"], -1)
        else:
            cv2.rectangle(out, (mid+dom_px, bar_y), (mid, bar_y+bar_h2),
                          _COLOUR["dom_neg"], -1)
        cv2.line(out, (mid, bar_y-2), (mid, bar_y+bar_h2+2), (200,200,200), 1)
        cv2.putText(out, f"DOM {dom:+.2f}", (bar_x, bar_y-3), fn, 0.45, fc, 1)
        cv2.putText(out, "A", (bar_x-14, bar_y+10), fn, 0.45, _COLOUR["dom_pos"], 1)
        cv2.putText(out, "B", (bar_x+bar_w+4, bar_y+10), fn, 0.45, _COLOUR["dom_neg"], 1)
