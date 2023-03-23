from src.stats_computation.game_state import *
from test.stats_computation.import_test_sequence import import_test_sequence


def test_draw():
    itr = import_test_sequence()
    game_state = GameState()
    for img, ann in itr:
        game_state.update(ann)
