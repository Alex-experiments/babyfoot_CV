from dataclasses import dataclass

from sklearn.cluster import DBSCAN

from src.stats_computation.interface.classes import *
from src.stats_computation.utils import Perspective

FIELD_MARGIN = 0.1

DBSCAN_EPS = 0.1
DBSCAN_MIN_SAMPLES = 1

# Field measures

FIELD_LENGTH = 1.0
FIELD_WIDTH = 0.57
BALL_RADIUS = 0.02

GOAL_REL_X = 0.003
DEFENSE_REL_X = 0.147
MIDDLE_REL_X = 0.432
ATTACK_REL_X = 0.717

DEFENSE_REL_W = 0.241
MIDDLE_REL_W = 0.422
ATTACK_REL_W = 0.341


@dataclass
class FieldState:
    image: Image
    detection: Detection

    def __post_init__(self):
        self.margin_field = self.compute_field_margin()
        self.init_perspective()
        self.compute_relative_position()

    def compute_field_margin(self) -> DetectedField:
        p1, p2, p3, p4 = self.detection.field.corners
        res1 = p1 + (p1 - p3) * FIELD_MARGIN
        res2 = p2 + (p2 - p4) * FIELD_MARGIN
        res3 = p3 + (p3 - p1) * FIELD_MARGIN
        res4 = p4 + (p4 - p2) * FIELD_MARGIN
        return DetectedField([res1, res2, res3, res4])

    def init_perspective(self):
        self.perspective = Perspective(self.detection.field)

    def compute_relative_position(self):
        if self.detection.ball is not None:
            self.ball_pos = self.perspective([self.detection.ball.pos])[0]
