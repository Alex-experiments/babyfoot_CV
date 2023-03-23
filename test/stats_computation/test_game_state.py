from src.stats_computation.game_state import *
from test.stats_computation.import_test_sequence import import_test_sequence


def test_game_state():
    game_state = GameState()
    itr = import_test_sequence()
    for img, ann in itr:
        game_state.update(ann)
