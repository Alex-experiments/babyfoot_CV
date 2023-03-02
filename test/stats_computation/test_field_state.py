from src.stats_computation.field_state import *
from test.stats_computation.import_test_sequence import import_test_sequence


def test_draw():
    itr = import_test_sequence()
    for img, ann in itr:
        fs = FieldState(img, ann)
