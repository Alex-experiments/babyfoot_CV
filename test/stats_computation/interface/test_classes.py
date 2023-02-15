from src.stats_computation.interface.classes import DetectedField


def test_order_field():
    field = DetectedField([(0, 0), (1, 2), (0, 2), (1, 0)])
    field.order()
    assert field.corners[0] == (0, 2)
    assert field.corners[1] == (0, 0)
    assert field.corners[2] == (1, 0)
    assert field.corners[3] == (1, 2)
