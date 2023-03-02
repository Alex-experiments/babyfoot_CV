import numpy as np
import cv2

from src.stats_computation.interface.classes import *


class Perspective:
    def __init__(self, field: DetectedField):
        pts1 = np.float32(field.corners)
        pts2 = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def __call__(self, pts: List[Coordinates]) -> List[Coordinates]:
        transformed_pts = cv2.perspectiveTransform(
            np.array(pts).reshape(1, -1, 2), self.matrix
        )
        return transformed_pts.reshape(-1, 2)
