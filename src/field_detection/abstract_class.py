from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class FieldDetection(ABC):
    """Abstract class which aims at detecting the four corners of the field and returning their coordinates"""

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Tuple[int, int]]:
        """Given an input image, compute and return the four corner's coordinates of the field.
        The input is an image obtained by cv2.imread in COLOR_BGR
        The output is expected to be of length 4"""
        pass
