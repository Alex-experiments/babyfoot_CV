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


@dataclass
class Coordinates:
    x: float
    y: float


@dataclass
class DetectedField:
    corners: List[Coordinates]

    def __post_init__(self):
        assert len(self.corners) == 4


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
