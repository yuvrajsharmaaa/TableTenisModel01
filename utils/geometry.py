"""
geometry.py
──────────────────────────────────────────────────────────────────────────────
Court / table geometry helpers.

Features
────────
• Auto table-corner detection using edge / contour / colour heuristics.
• Net line estimation from table corners.
• Coordinate normalisation (pixel ↔ table-metre space).
• Point-side discrimination (left half vs right half).

Inspired by:
  ckjellson/tt_tracker  – manual corner selection + camera-matrix approach
  Here we automate corner detection so no manual input is needed.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Physical table dimensions (ITTF)
# ──────────────────────────────────────────────────────────────────────────────
TABLE_W_M = 2.74
TABLE_H_M = 1.525
NET_H_M   = 0.1525
ASPECT_RATIO = TABLE_W_M / TABLE_H_M  # ≈ 1.796


# ──────────────────────────────────────────────────────────────────────────────
# Colour-based table segmentation
# ──────────────────────────────────────────────────────────────────────────────

# Common table surface colours in HSV ranges  (blue / green / black)
_TABLE_COLOUR_RANGES = [
    # Blue tables (most common broadcast tables)
    (np.array([100, 60, 40]), np.array([130, 255, 220])),
    # Green tables
    (np.array([40, 50, 30]),  np.array([80, 255, 200])),
    # Dark blue / grey
    (np.array([90, 30, 20]),  np.array([130, 200, 150])),
]


def detect_table_mask(frame: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask (uint8) where the table surface is ~255.
    Uses HSV colour segmentation + morphological cleanup.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for lo, hi in _TABLE_COLOUR_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        mask = cv2.bitwise_or(mask, m)

    # Morphology: close small holes, remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    return mask


def detect_table_corners(
    frame: np.ndarray,
) -> Optional[List[Tuple[int, int]]]:
    """
    Returns [TL, TR, BR, BL] pixel coordinates of table corners, or None.

    Strategy:
    1. Colour-segment the table.
    2. Find the largest quadrilateral contour.
    3. Approximate to 4 corners.
    """
    mask = detect_table_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Largest contour by area
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.01 * frame.shape[0] * frame.shape[1]:
        return None   # too small – probably noise

    # Approximate polygon
    eps = 0.02 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    if len(approx) != 4:
        # Fall back: use bounding rectangle
        rect = cv2.minAreaRect(c)
        box  = cv2.boxPoints(rect)
        approx = box.reshape(4, 1, 2).astype(int)

    pts = approx.reshape(-1, 2).tolist()
    pts = [tuple(p) for p in pts]
    pts = _sort_corners(pts)
    return pts


def _sort_corners(pts: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort corners: [TL, TR, BR, BL]."""
    pts = sorted(pts, key=lambda p: p[1])   # sort by y (top first)
    top = sorted(pts[:2], key=lambda p: p[0])
    bot = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bot[1], bot[0]]   # TL, TR, BR, BL


# ──────────────────────────────────────────────────────────────────────────────
# Coordinate transform helpers
# ──────────────────────────────────────────────────────────────────────────────

class CourtTransform:
    """
    Computes a homography from detected table corners to a normalised
    top-view court rectangle.

    Normalised space: x ∈ [0, TABLE_W_M], y ∈ [0, TABLE_H_M] (in metres).
    """

    def __init__(self, corners: List[Tuple[int, int]]):
        """
        corners: [TL, TR, BR, BL] pixel coords.
        """
        src = np.array(corners, dtype=np.float32)
        dst = np.array([
            [0,         0        ],
            [TABLE_W_M, 0        ],
            [TABLE_W_M, TABLE_H_M],
            [0,         TABLE_H_M],
        ], dtype=np.float32)
        self.H, _  = cv2.findHomography(src, dst)
        self.H_inv, _ = cv2.findHomography(dst, src)

    def px_to_metre(self, px: Tuple[float, float]) -> Tuple[float, float]:
        p = np.array([[[px[0], px[1]]]], dtype=np.float32)
        r = cv2.perspectiveTransform(p, self.H)[0][0]
        return float(r[0]), float(r[1])

    def metre_to_px(self, m: Tuple[float, float]) -> Tuple[int, int]:
        p = np.array([[[m[0], m[1]]]], dtype=np.float32)
        r = cv2.perspectiveTransform(p, self.H_inv)[0][0]
        return int(r[0]), int(r[1])

    @property
    def net_px_x(self) -> int:
        """X pixel coordinate of net centre line."""
        net_m = (TABLE_W_M / 2, TABLE_H_M / 2)
        px = self.metre_to_px(net_m)
        return px[0]

    @property
    def px_per_metre(self) -> float:
        """Approximate pixels-per-metre at table centre."""
        # Estimate from left-edge to right-edge projected
        left_px  = self.metre_to_px((0.0, TABLE_H_M / 2))
        right_px = self.metre_to_px((TABLE_W_M, TABLE_H_M / 2))
        return math.hypot(
            right_px[0]-left_px[0], right_px[1]-left_px[1]
        ) / TABLE_W_M


# ──────────────────────────────────────────────────────────────────────────────
# Utility: draw top-view court mini-map
# ──────────────────────────────────────────────────────────────────────────────

def draw_minimap(
    size: Tuple[int, int],                         # (width, height) pixels
    ball_m: Optional[Tuple[float, float]] = None,  # ball position in metres
    bounce_positions: Optional[List[Tuple[float, float]]] = None,
    player_positions: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Returns a BGR top-view minimap image of the table.

    ball_m:            (x, y) in table-metre coordinates
    bounce_positions:  list of (x, y) in metres
    player_positions:  list of (x, y) in metres (approximate)
    """
    W, H = size
    img  = np.zeros((H, W, 3), dtype=np.uint8)

    # Background - table green/blue
    cv2.rectangle(img, (0, 0), (W, H), (40, 100, 40), -1)

    # Table outline
    cv2.rectangle(img, (2, 2), (W-3, H-3), (200, 200, 200), 2)

    # Centre line (net)
    cv2.line(img, (W//2, 0), (W//2, H), (255, 255, 255), 2)

    # Half-time lines
    cv2.line(img, (0, H//2), (W, H//2), (150, 150, 150), 1)

    def m2px(mx: float, my: float) -> Tuple[int, int]:
        px = int(mx / TABLE_W_M * W)
        py = int(my / TABLE_H_M * H)
        return px, py

    # Bounce heatmap dots  (ported from ckjellson bounce_heatmap)
    if bounce_positions:
        for bx, by in bounce_positions:
            if 0 <= bx <= TABLE_W_M and 0 <= by <= TABLE_H_M:
                pp = m2px(bx, by)
                cv2.circle(img, pp, 4, (0, 255, 255), -1, cv2.LINE_AA)

    # Player markers
    if player_positions:
        for i, (px, py) in enumerate(player_positions):
            col = (100, 220, 100) if i == 0 else (100, 100, 220)
            pp  = m2px(
                max(0, min(TABLE_W_M, px)),
                max(0, min(TABLE_H_M, py)),
            )
            cv2.drawMarker(img, pp, col, cv2.MARKER_TRIANGLE_UP, 12, 2)

    # Ball
    if ball_m is not None:
        bx_px, by_px = m2px(
            max(0, min(TABLE_W_M, ball_m[0])),
            max(0, min(TABLE_H_M, ball_m[1])),
        )
        cv2.circle(img, (bx_px, by_px), 5, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (bx_px, by_px), 5, (255,255,255), 1, cv2.LINE_AA)

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Side detector
# ──────────────────────────────────────────────────────────────────────────────

def ball_side(ball_px_x: float, net_px_x: int) -> str:
    """Returns 'A' (left of net) or 'B' (right of net)."""
    return "A" if ball_px_x <= net_px_x else "B"
