import os

import numpy as np

from src.stats_computation.interface.classes import DetectionSequence, DetectedField
from src.stats_computation.interface.annotations_extraction import (
    extract_detection_from_folder,
)

TEST_FOLDER_PATH = os.path.join("test", "stats_computation", "test_images")
FIELD_CORNERS = [
    np.array([0.37812500000000004, 0.12847222222222224]),
    np.array([0.6257812500000001, 0.12708333333333333]),
    np.array([0.66953125, 0.8798611111111111]),
    np.array([0.32460937500000003, 0.8791666666666667]),
]
TEST_FPS = 30


def import_test_sequence() -> DetectionSequence:
    return extract_detection_from_folder(TEST_FOLDER_PATH, DetectedField(FIELD_CORNERS))


CAMERA_EXAMPLE_FOLDER_PATH = os.path.join("baby-foot-dataset", "camera_example")
CAMERA_EXAMPLE_FPS = 60


def import_camera_example_sequence() -> DetectionSequence:
    return extract_detection_from_folder(
        CAMERA_EXAMPLE_FOLDER_PATH, sep=",", convert=True
    )
