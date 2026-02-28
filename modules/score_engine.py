"""
score_engine.py
──────────────────────────────────────────────────────────────────────────────
Score, domination, expected-score, and stress logic.

Directly ported / adapted from:
  centralelyon/table-tennis-analytics
  – Calcul_Domination_Match.py  (score → win-probability, physical domination)
  – Calcul_Domination_Set.py    (stress / moral metric)
  – Expected_Points.py          (expected-score via playing-pattern tree)
  – Creativity.py               (shot diversity distance measure)

All functions are pure-Python / NumPy, stateless helpers + a stateful
ScoreEngine class used frame-by-frame by the main tt_analyzer.
"""

from __future__ import annotations

import math
import collections
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# ── Score probability helpers (centralelyon/Calcul_Domination_Match.py) ──────
# ──────────────────────────────────────────────────────────────────────────────

def proba_a_wins_set(score_a: int, score_b: int) -> float:
    """
    Recursive probability that player A wins the current set given current scores.
    Table tennis: first to 11, lead by 2 (deuce above 10-10).
    Ported from: centralelyon/Calcul_Domination_Match.py::proba_A_gagne
    """
    if score_a == score_b == 0:
        return 0.5

    total = score_a + score_b
    if total == 0:
        p = 0.5
    else:
        p = score_a / total     # empirical win-per-point probability
    q = 1.0 - p

    # Already past 11 → check lead
    if score_a >= 11 or score_b >= 11:
        diff = score_a - score_b
        if diff > 1:
            return 1.0
        if diff < -1:
            return 0.0
        if diff == 1:
            return p + q * 0.5
        if diff == -1:
            return p * 0.5

    # Early in the set
    if total < 5:
        return (proba_a_wins_set(score_a+1, score_b) +
                proba_a_wins_set(score_a, score_b+1)) / 2

    # General case: from each state p(win) = p*p(win|A scores) + q*p(win|B scores)
    # Memoized via helper below
    return _proba_set_memo(score_a, score_b, p, q)


_memo_cache: Dict[tuple, float] = {}

def _proba_set_memo(a: int, b: int, p: float, q: float) -> float:
    key = (a, b)
    if key in _memo_cache:
        return _memo_cache[key]
    if a >= 11 or b >= 11:
        diff = a - b
        if diff > 1:   result = 1.0
        elif diff < -1: result = 0.0
        elif diff == 1: result = p + q * 0.5
        else:           result = p * 0.5
    else:
        result = p * _proba_set_memo(a+1, b, p, q) + q * _proba_set_memo(a, b+1, p, q)
    _memo_cache[key] = result
    return result


def score_domination(score_a: int, score_b: int) -> float:
    """
    Score-based domination ∈ [-1, 1].
    1.0 = A dominates completely, -1.0 = B dominates.
    Ported from: centralelyon/Calcul_Domination_Set.py::fonction_domination_score
    """
    _memo_cache.clear()
    return 2.0 * proba_a_wins_set(score_a, score_b) - 1.0


# ──────────────────────────────────────────────────────────────────────────────
# ── Playing-Pattern Tree (centralelyon/ExpectedScore) ─────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class _TreeNode:
    """Node in the playing-pattern tree."""
    __slots__ = ("name", "children", "hits", "wins", "win_prob")

    def __init__(self, name: str):
        self.name     = name
        self.children : Dict[str, "_TreeNode"] = {}
        self.hits     = 0
        self.wins     = 0
        self.win_prob = 0.0

    def child(self, key: str) -> "_TreeNode":
        return self.children.setdefault(key, _TreeNode(key))

    def update_probs(self):
        self.win_prob = (self.wins / self.hits) if self.hits > 0 else 0.0
        for c in self.children.values():
            c.update_probs()


class PatternTree:
    """
    Tree of playing patterns for expected-score computation.

    Each path encodes a rally as:
    [root, lat, stroke_type, zone, lat, stroke_type, zone, …, outcome]

    • ''outcome'' is either '<zone> pt_gagne'  or  'faute'
    • Depth-3 lookahead (service + 2 shots) is enough for real-time.

    Ported from: centralelyon/ExpectedScore/Analyse_Simu.py + Expected_Points.py
    """

    MAX_DEPTH = 9   # 3 strokes × 3 tokens each

    def __init__(self):
        self.root = _TreeNode("root")

    def add_rally(self, path: List[str], winner_is_server: bool) -> None:
        """Feed one completed rally into the tree."""
        node = self.root
        node.hits += 1
        if winner_is_server:
            node.wins += 1
        for token in path[:self.MAX_DEPTH]:
            node = node.child(token)
            node.hits += 1
            if winner_is_server:
                node.wins += 1
        self.root.update_probs()

    def expected_score(
        self, partial_path: List[str], server_is_a: bool
    ) -> Tuple[float, float]:
        """
        Look up expected points increment for players A and B given the
        partial rally path and who is serving.
        Returns (x_score_a_increment, x_score_b_increment).
        """
        node = self.root
        for token in partial_path[:self.MAX_DEPTH]:
            if token in node.children:
                node = node.children[token]
            else:
                break   # unknown branch – use current node's probability

        p = node.win_prob if node.win_prob > 0 else 0.5
        if server_is_a:
            return p, 1.0 - p
        else:
            return 1.0 - p, p


# ──────────────────────────────────────────────────────────────────────────────
# ── Stress / Moral metric (centralelyon/Calcul_Domination_Set.py::calcul_stress)
# ──────────────────────────────────────────────────────────────────────────────

class _StressState:
    def __init__(self):
        self.stress_a = 1.0
        self.stress_b = 1.0
        self.rally_duration = 0          # strokes in current rally
        self.consec_fault_a = 0
        self.consec_fault_b = 0

    def update_rally_shot(self) -> None:
        self.rally_duration += 1

    def update_point_end(
        self, winner: str, fault_player: Optional[str], score_a: int, score_b: int
    ) -> None:
        """
        Update stress after a point ends.
        winner: 'A' | 'B'
        fault_player: 'A' | 'B' | None  (None = winner scored, not opponent fault)
        """
        # Long rallies increase loser stress
        if self.rally_duration > 6:
            stress_inc = 0.1 * (self.rally_duration - 6) / 10.0
            if winner == "A":
                self.stress_b = min(3.0, self.stress_b + stress_inc)
            else:
                self.stress_a = min(3.0, self.stress_a + stress_inc)

        # Faults increase that player's stress
        if fault_player == "A":
            self.consec_fault_a += 1
            self.consec_fault_b  = 0
            self.stress_a = min(3.0, self.stress_a + 0.05 * self.consec_fault_a)
        elif fault_player == "B":
            self.consec_fault_b += 1
            self.consec_fault_a  = 0
            self.stress_b = min(3.0, self.stress_b + 0.05 * self.consec_fault_b)
        else:
            self.consec_fault_a = 0
            self.consec_fault_b = 0

        # Ball-point (near-win) stress
        if score_a >= 10 and score_b >= 10:
            self.stress_a = min(3.0, self.stress_a + 0.05)
            self.stress_b = min(3.0, self.stress_b + 0.05)

        # Reset rally duration
        self.rally_duration = 0

        # Gradual decay toward baseline
        self.stress_a = max(1.0, self.stress_a * 0.98)
        self.stress_b = max(1.0, self.stress_b * 0.98)

    def normalised(self) -> Tuple[float, float]:
        """Stress normalised to [-1, 1] (positive = A more stressed than B)."""
        diff = self.stress_a - self.stress_b
        norm = max(1.0, abs(diff))
        return diff / norm, -diff / norm


# ──────────────────────────────────────────────────────────────────────────────
# ── Main stateful ScoreEngine ─────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────────

class ScoreEngine:
    """
    Maintains match state and computes analytics metrics per point/rally:

    Metrics exposed
    ───────────────
    • score_a, score_b            – raw point scores
    • net_cross_count             – total net crossings (proxy for rally length)
    • score_dom                   – score-based domination  ∈ [-1, 1]
    • stress_diff_a, stress_diff_b– mental stress normalised  ∈ [-1, 1]
    • x_score_a, x_score_b        – expected score (running total)
    • domination                  – combined domination (score 40% + stress 30% + phys 30%)
    Physical domination is fed in from outside (pose/distance data).

    Net-crossing rule → score
    ──────────────────────────
    • Ball crosses net           → rally in progress
    • Bounce detected on server's side again without crossing → point for opponent
    • Simplified: after every 2 crosses (serve + return) → check bounce side
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps

        # Scores
        self.score_a   = 0
        self.score_b   = 0
        self.set_score_a = 0  # sets won
        self.set_score_b = 0

        # Rally state
        self._in_rally       = False
        self._net_cross_rally= 0
        self._server         = "A"   # alternate each point
        self._server_locked  = False

        # Net-crossing total
        self.net_cross_count = 0

        # Domination components
        self.score_dom    = 0.0
        self.stress_a_val = 0.0
        self.stress_b_val = 0.0
        self.phys_dom     = 0.0       # set externally from pose analyzer data
        self.domination   = 0.0

        # Expected score
        self._tree = PatternTree()
        self.x_score_a = 0.0
        self.x_score_b = 0.0
        self._rally_path: List[str] = ["root"]

        # Stress
        self._stress = _StressState()

        # History for charts (kept as rolling deques)
        self.history_score_a  : collections.deque = collections.deque(maxlen=200)
        self.history_score_b  : collections.deque = collections.deque(maxlen=200)
        self.history_dom      : collections.deque = collections.deque(maxlen=200)
        self.history_x_score_a: collections.deque = collections.deque(maxlen=200)
        self.history_x_score_b: collections.deque = collections.deque(maxlen=200)

    # ──────────────────────────────────────────────────────────────────────────
    def on_net_cross(self) -> None:
        """Call each time the ball crosses the net."""
        self.net_cross_count     += 1
        self._net_cross_rally    += 1
        self._in_rally            = True
        self._stress.update_rally_shot()

        # Update expected score using partial rally path
        xa, xb = self._tree.expected_score(
            self._rally_path, server_is_a=(self._server == "A")
        )
        self.x_score_a += xa * 0.01   # incremental update
        self.x_score_b += xb * 0.01

    def on_bounce(self, on_a_side: bool) -> None:
        """
        Call when a bounce is detected.
        on_a_side: True if bounce is on player A's half of the table.
        """
        # A bounce on your own side after you've already returned = potential fault
        # If server A has served, ball bounces their side twice → point to B
        pass   # Full bounce-based scoring requires table corner calibration;
               # deferred to the external scoring panel in the Streamlit app.

    def on_point_scored(
        self,
        winner: str,
        stroke_type: Optional[str] = None,
        zone: Optional[str] = None,
        fault_player: Optional[str] = None,
    ) -> dict:
        """
        Call when a point ends (winner = 'A' or 'B').
        Updates scores, set detection, domination, stress, expected score.
        Returns snapshot of current metrics.
        """
        if winner == "A":
            self.score_a += 1
        else:
            self.score_b += 1

        # Build rally path token for pattern tree
        if stroke_type:
            self._rally_path.append(stroke_type)
        if zone:
            self._rally_path.append(zone if winner == self._server else "faute")

        # Feed rally into pattern tree
        self._tree.add_rally(
            self._rally_path[1:],
            winner_is_server=(winner == self._server)
        )
        self._rally_path = ["root"]

        # Stress update
        self._stress.update_point_end(winner, fault_player, self.score_a, self.score_b)

        # Alternate server
        self._server = "B" if self._server == "A" else "A"
        self._net_cross_rally = 0
        self._in_rally = False

        # Set detection (ITTF: first to 11, lead by 2)
        if (self.score_a >= 11 and self.score_a - self.score_b >= 2) or \
           (self.score_b >= 11 and self.score_b - self.score_a >= 2):
            if self.score_a > self.score_b:
                self.set_score_a += 1
            else:
                self.set_score_b += 1
            self.score_a = 0
            self.score_b = 0

        return self._compute_metrics()

    def tick(self, phys_dom: float = 0.0) -> dict:
        """
        Called every frame – updates derived metrics without changing scores.
        phys_dom: physical domination ∈ [-1, 1] from pose data.
        """
        self.phys_dom = phys_dom
        return self._compute_metrics()

    # ──────────────────────────────────────────────────────────────────────────
    def _compute_metrics(self) -> dict:
        self.score_dom = score_domination(self.score_a, self.score_b)
        sa_norm, _ = self._stress.normalised()
        self.stress_a_val = sa_norm

        # Combined domination: 40% score, 30% stress, 30% physical
        self.domination = (
            0.40 * self.score_dom
            + 0.30 * (-sa_norm)   # stress higher for A → A under pressure → lower dom
            + 0.30 * self.phys_dom
        )
        self.domination = max(-1.0, min(1.0, self.domination))

        snapshot = {
            "score_a"      : self.score_a,
            "score_b"      : self.score_b,
            "set_score_a"  : self.set_score_a,
            "set_score_b"  : self.set_score_b,
            "server"       : self._server,
            "score_dom"    : round(self.score_dom, 3),
            "stress_diff"  : round(sa_norm, 3),
            "phys_dom"     : round(self.phys_dom, 3),
            "domination"   : round(self.domination, 3),
            "x_score_a"    : round(self.x_score_a, 2),
            "x_score_b"    : round(self.x_score_b, 2),
            "net_crosses"  : self.net_cross_count,
        }

        self.history_score_a.append(self.score_a)
        self.history_score_b.append(self.score_b)
        self.history_dom.append(self.domination)
        self.history_x_score_a.append(self.x_score_a)
        self.history_x_score_b.append(self.x_score_b)

        return snapshot

    def reset(self) -> None:
        """Full match reset."""
        self.__init__(self.fps)

    def reset_set(self) -> None:
        """Reset point scores only (keep set scores)."""
        self.score_a = 0
        self.score_b = 0
        self._server = "A"
        self._net_cross_rally = 0
        self._in_rally = False
        self._rally_path = ["root"]
        self._stress = _StressState()
