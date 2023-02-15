from typing import List, Tuple, Iterator
from dataclasses import dataclass

import numpy as np

# For all coordinates (x, y) and rectangle sizes (lx, ly), the format is the same as
# yolo's one, namely float between 0 and 1


@dataclass
class DetectedObject:
    x: float
    y: float
    lx: float
    ly: float


class DetectedBall(DetectedObject):
    pass


class DetectedRedPlayer(DetectedObject):
    pass


class DetectedBluePlayer(DetectedObject):
    pass


Coordinates = Tuple[float, float]


@dataclass
class DetectedField:
    corners: List[Coordinates]

    def __post_init__(self):
        assert len(self.corners) == 4

    def order(self):
        # 1st corner = leftmost point
        idx1 = 0
        x1, y1 = self.corners[0]
        for idx in range(1, 4):
            if self.corners[idx] < (x1, y1):
                idx1, x1 = idx, self.corners[idx][0]

        # Then in clockwise order
        points = self.corners.copy()
        points.pop(idx1)
        vectors = [(x - x1, y - y1) for x, y in points]
        cos_angles = [
            y / (x**2 + y**2) ** 0.5 for x, y in vectors
        ]  # cosinus of the angle with y axis
        order = [1, 2, 3]
        order.sort(key=lambda x: cos_angles[x - 1])
        res = [self.corners[idx1], *[self.corners[i] for i in order]]

        # Shift if the 1st side is a width instead of a length
        lengths_squared = [
            (res[i][0] - res[(i + 1) % 4][0]) ** 2
            + (res[i][1] - res[(i + 1) % 4][1]) ** 2
            for i in range(4)
        ]
        if (
            lengths_squared[0] + lengths_squared[2]
            <= lengths_squared[1] + lengths_squared[3]
        ):
            head = res.pop()
            res = [head] + res

        self.corners = res


@dataclass
class Detection:
    field: DetectedField | None
    ball: DetectedBall | None
    red_players: List[DetectedRedPlayer]
    blue_players: List[DetectedBluePlayer]

    def __post_init__(self):
        assert len(self.red_players) <= 11
        assert len(self.blue_players) <= 11


class Image(np.ndarray):
    pass


AnnotatedSequence = Iterator[Tuple[Image, Detection]]
