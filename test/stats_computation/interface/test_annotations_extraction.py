import os

import numpy as np

from src.stats_computation.interface.annotations_extraction import (
    extract_detection_from_folder,
)

TEST_FOLDER_PATH = os.path.join("test", "stats_computation", "test_images")
EPSILON = 1e-5


def test_extract_detection_from_folder() -> None:
    itr = extract_detection_from_folder(TEST_FOLDER_PATH)
    for image, annotation in itr:
        assert image.shape == (720, 1280, 3)
        pos = np.array([0.529296875, 0.26944444444444443])
        width = np.array([0.013281250000000001, 0.02777777777777778])
        assert (abs(annotation.ball.pos - pos) < EPSILON).all()
        assert (abs(annotation.ball.width - width) < EPSILON).all()
        assert len(annotation.red_players) == 11
        assert len(annotation.blue_players) == 11
        break
