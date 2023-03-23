from src.stats_computation.animate import *
from test.stats_computation.import_test_sequence import import_test_sequence


def test_draw():
    anim = Animation()
    itr = import_test_sequence()
    for img, ann in itr:
        anim.update(ann, img)
        anim.draw()
