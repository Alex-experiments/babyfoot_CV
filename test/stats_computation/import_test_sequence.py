import os

from src.stats_computation.interface.classes import AnnotatedSequence, DetectedField
from src.stats_computation.interface.annotations_extraction import (
    extract_detection_from_folder,
)

TEST_FOLDER_PATH = os.path.join("test", "stats_computation", "test_images")
FIELD_CORNERS = [
    (0.37812500000000004, 0.12847222222222224),
    (0.6257812500000001, 0.12708333333333333),
    (0.66953125, 0.8798611111111111),
    (0.32460937500000003, 0.8791666666666667),
]


def import_test_sequence() -> AnnotatedSequence:
    return extract_detection_from_folder(TEST_FOLDER_PATH, DetectedField(FIELD_CORNERS))
