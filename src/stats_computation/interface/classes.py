from typing import List, Tuple, Iterator
from dataclasses import dataclass

import numpy as np

# For all coordinates (x, y) and rectangle sizes (lx, ly), the format is the same as
# yolo's one, namely float between 0 and 1

Coordinates = np.ndarray  # Represents x and y coordinates, array of shape (2,)


@dataclass
class DetectedObject:
    pos: Coordinates
    width: Coordinates


class DetectedBall(DetectedObject):
    pass


class DetectedRedPlayer(DetectedObject):
    pass


class DetectedBluePlayer(DetectedObject):
    pass


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
            if tuple(self.corners[idx]) < (x1, y1):
                idx1, x1 = idx, self.corners[idx][0]

        # Then in clockwise order
        points = self.corners.copy()
        point1 = points.pop(idx1)
        vectors = [point - point1 for point in points]
        angles = [
            vector[1] ** 2 / vector.dot(vector) for vector in vectors
        ]  # square of the cosinus of the angle with y axis
        order = [1, 2, 3]
        order.sort(key=lambda x: angles[x - 1])
        res = [self.corners[idx1], *[self.corners[i] for i in order]]

        # Shift if the 1st side is a width instead of a length
        vectors = [res[i] - res[(i + 1) % 4] for i in range(4)]
        lengths_squared = [vec.dot(vec) for vec in vectors]
        if (
            lengths_squared[0] + lengths_squared[2]
            <= lengths_squared[1] + lengths_squared[3]
        ):
            head = res.pop()
            res = [head] + res

        self.corners = res


@dataclass
class Detection:
    field: DetectedField #| None
    ball: DetectedBall #| None
    red_players: List[DetectedRedPlayer]
    blue_players: List[DetectedBluePlayer]


class Image(np.ndarray):
    pass


DetectionSequence = Iterator[Tuple[Image, Detection]]
