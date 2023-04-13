import numpy as np

from src.stats_computation.interface.classes import DetectedField


def test_order_field():
    corners = [np.array([0, 0]), np.array([1, 2]), np.array([0, 2]), np.array([1, 0])]
    field = DetectedField(corners)
    field.order()
    assert (field.corners[0] == corners[2]).all()
    assert (field.corners[1] == corners[0]).all()
    assert (field.corners[2] == corners[3]).all()
    assert (field.corners[3] == corners[1]).all()
