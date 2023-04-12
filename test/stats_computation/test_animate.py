from src.stats_computation.animate import *
from test.stats_computation.import_test_sequence import import_test_sequence, TEST_FPS


def test_draw():
    anim = Animation(fps=TEST_FPS, save_name="test.json")
    itr = import_test_sequence()
    for img, ann in itr:
        anim.update(ann, img)
        anim.draw()
