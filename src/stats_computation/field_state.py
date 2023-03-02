from dataclasses import dataclass

from sklearn.cluster import DBSCAN

from src.stats_computation.interface.classes import *
from src.stats_computation.utils import compute_relative_coord

FIELD_MARGIN = 0.1


@dataclass
class FieldState:
    image: Image
    detection: Detection

    def __post_init__(self):
        self.margin_field = self.compute_field_margin()

    def compute_field_margin(self) -> DetectedField:
        p1, p2, p3, p4 = self.detection.field.corners
        res1 = p1 + (p1 - p3) * FIELD_MARGIN
        res2 = p2 + (p2 - p4) * FIELD_MARGIN
        res3 = p3 + (p3 - p1) * FIELD_MARGIN
        res4 = p4 + (p4 - p2) * FIELD_MARGIN
        return DetectedField([res1, res2, res3, res4])
