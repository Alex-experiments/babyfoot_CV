from src.stats_computation.stats_extraction import *
from test.stats_computation.import_test_sequence import import_test_sequence, TEST_FPS


def test_stats_extraction():
    stats_extraction = StatsExtraction(fps=TEST_FPS, save_name="test.json")
    itr = import_test_sequence()
    for img, ann in itr:
        stats_extraction.update(ann)
    stats_extraction.save()
