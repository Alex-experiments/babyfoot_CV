from typing import List, Tuple
from dataclasses import dataclass

import numpy as np
import cv2
import imutils

from src.field_detection.abstract_class import FieldDetection

DEFAULT_PARAMETERS = {
    "lower_filter": np.array([70, 100, 20]),
    "upper_filter": np.array([90, 255, 255]),
    "nb_blur": 20,
    "kernel_ratio_blur": 0.02,
    "threshold_value": 10,
    "epsilon_contours_approximation": 0.04,
}


@dataclass
class ChromaticDetection(FieldDetection):
    intermediate_image_saving: bool = False
    lower_filter: np.ndarray = DEFAULT_PARAMETERS["lower_filter"]
    upper_filter: np.ndarray = DEFAULT_PARAMETERS["upper_filter"]
    nb_blur: int = DEFAULT_PARAMETERS["nb_blur"]
    kernel_ratio_blur: float = DEFAULT_PARAMETERS["kernel_ratio_blur"]
    threshold_value: int = DEFAULT_PARAMETERS["threshold_value"]
    epsilon_contours_approximation: float = DEFAULT_PARAMETERS[
        "epsilon_contours_approximation"
    ]

    def color_filter(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a color filter to an image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_filter, self.upper_filter)
        filtered = cv2.bitwise_and(image, image, mask=mask)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)

        if self.intermediate_image_saving:
            cv2.imwrite("mask.jpg", mask)
            cv2.imwrite("filtered.jpg", filtered)

        filtered = cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)

        return filtered, mask

    def blur(self, image: np.ndarray) -> np.ndarray:
        """Blur the image to help the detection of the field"""
        blurred = image.copy()
        kernel = (
            int(image.shape[0] * self.kernel_ratio_blur / 2) * 2 + 1,
            int(image.shape[1] * self.kernel_ratio_blur / 2) * 2 + 1,
        )  # There must be ksize.width % 2 == 1 && ksize.height % 2 == 1
        for _ in range(self.nb_blur):
            blurred = cv2.GaussianBlur(blurred, kernel, cv2.BORDER_DEFAULT)

        if self.intermediate_image_saving:
            cv2.imwrite("blurred.jpg", blurred)

        return blurred

    def threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply a threshold to keep the objects and remove noise"""
        threshold = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)[
            1
        ]

        if self.intermediate_image_saving:
            cv2.imwrite("threshold.jpg", threshold)

        return threshold

    def approximate_contours(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find the contours of the image and approximate it
        The output is a contour in the format of cv2: a ndarray of shape (X, 1, 2)"""
        contours = cv2.findContours(
            image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = imutils.grab_contours(contours)
        assert len(contours) == 1  # It must recognize one object only
        contours = contours[0]
        peri = cv2.arcLength(contours, True)
        approx = cv2.approxPolyDP(
            contours, self.epsilon_contours_approximation * peri, True
        )

        if self.intermediate_image_saving:
            saved_image = image.copy()
            saved_image = cv2.cvtColor(saved_image, cv2.COLOR_GRAY2RGB)
            cv2.drawContours(saved_image, [contours], -1, (0, 255, 0), 3)
            cv2.drawContours(saved_image, [approx], -1, (0, 0, 255), 3)
            cv2.imwrite("contours_approximation.jpg", saved_image)

        return contours, approx

    def detect(self, image: np.ndarray) -> List[Tuple[int, int]]:
        filtered, _ = self.color_filter(image)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        blurred = self.blur(gray)
        threshold = self.threshold(blurred)
        contours, approx = self.approximate_contours(threshold)

        if self.intermediate_image_saving:
            saved_image = image.copy()
            cv2.drawContours(saved_image, [contours], -1, (0, 255, 0), 3)
            cv2.drawContours(saved_image, [approx], -1, (0, 0, 255), 3)
            cv2.imwrite("field_detection.jpg", saved_image)

        # Convert approximated contour to List[Tuple[int, int]]
        approx = approx.reshape(-1, 2)
        res = [(x[0], x[1]) for x in approx]
        assert len(res) == 4

        return res


if __name__ == "__main__":
    initial_image = cv2.imread("test/field_detection/test_images/image.jpg")
    cd = ChromaticDetection(intermediate_image_saving=True)
    cd.detect(initial_image)
