"""
Table Tennis Analytics Modules
Combines ideas from:
  - wutonytt/Camera-Based-Table-Tennis-Posture-Analysis  (pose/stroke classification)
  - ckjellson/tt_tracker                                  (3D ball tracking, bounce detection)
  - centralelyon/table-tennis-analytics                  (score domination, expected score, creativity)
"""
from .ball_tracker import BallTracker
from .pose_analyzer import PoseAnalyzer
from .score_engine import ScoreEngine
from .annotator import Annotator

__all__ = ["BallTracker", "PoseAnalyzer", "ScoreEngine", "Annotator"]
