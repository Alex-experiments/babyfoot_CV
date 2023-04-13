import numpy as np

from src.stats_computation.utils import *
from src.stats_computation.interface.classes import *

EPSILON = 1e-5


def test_perspective():
    field = DetectedField(
        [
            np.array([1.0, 1.0]),
            np.array([3.0, 3.0]),
            np.array([5.0, 3.0]),
            np.array([7.0, 1.0]),
        ]
    )
    prt = Perspective(field)
    pt = np.array([4.0, 3.0])
    res = prt([pt])[0]
    assert (abs(res - np.array([1.0, 0.5])) < EPSILON).all()
