import os

import cv2

from src.field_detection.chromatic import ChromaticDetection

# try:
#     from src.field_detection.chromatic import ChromaticDetection
# except ModuleNotFoundError:
#     here = os.getcwd()
#     raise Exception("|" + str(os.listdir(here)) + "|")

PATH_TEST_IMAGE = os.path.join("test", "field_detection", "test_images")


def test_chromatic_detection() -> None:
    for file in os.listdir(PATH_TEST_IMAGE):
        image = cv2.imread(os.path.join(PATH_TEST_IMAGE, file))
        cd = ChromaticDetection(intermediate_image_saving=False)
        cd.detect(image)
