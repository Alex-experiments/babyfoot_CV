from src.stats_computation.stats_extraction import *
from test.stats_computation.import_test_sequence import import_test_sequence


def test_stats_extraction():
    stats_extraction = StatsExtraction()
    itr = import_test_sequence()
    for img, ann in itr:
        stats_extraction.update(ann)
        # TODO Add tests
