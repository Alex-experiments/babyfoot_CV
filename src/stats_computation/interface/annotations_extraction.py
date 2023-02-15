from typing import Tuple
import os

import pandas as pd
import cv2

from src.stats_computation.interface.classes import *

LABEL_BALL = 0
LABEL_RED_PLAYER = 1
LABEL_BLUE_PLAYER = 2


def extract_object_from_annotation(
    annotation: pd.DataFrame, label: int
) -> List[List[float]]:
    objects = annotation.loc[annotation["label"] == label].to_dict("records")
    return [
        [np.array([object["x"], object["y"]]), np.array([object["lx"], object["ly"]])]
        for object in objects
    ]


def extract_detection_from_file(
    filepath: str, field: DetectedField = None
) -> Detection:
    annotation = pd.read_csv(
        filepath,
        sep=" ",
        names=["label", "x", "y", "lx", "ly"],
    )
    # Ball extraction
    balls = extract_object_from_annotation(annotation, LABEL_BALL)
    ball = None if len(balls) == 0 else DetectedBall(*(balls[0]))

    # Red players extraction
    red_players = [
        DetectedRedPlayer(*obj)
        for obj in extract_object_from_annotation(annotation, LABEL_RED_PLAYER)
    ]

    # Blue players extraction
    blue_players = [
        DetectedBluePlayer(*obj)
        for obj in extract_object_from_annotation(annotation, LABEL_BLUE_PLAYER)
    ]

    return Detection(field, ball, red_players, blue_players)


def list_files(path: str) -> Tuple[List[str], List[str]]:
    images = []
    annotations = []
    for file in os.listdir(path):
        if file == "place_holder.txt":
            continue
        if file.endswith(".jpg"):
            images.append(file)
        if file.endswith(".txt"):
            annotations.append(file)
    idx_frame = lambda x: int(x.split("_")[-1].split(".")[0])
    images.sort(key=idx_frame)
    annotations.sort(key=idx_frame)
    return images, annotations


def extract_detection_from_folder(
    path: str, field: DetectedField = None
) -> AnnotatedSequence:
    images, annotations = list_files(path)
    for image, annotation in zip(images, annotations):
        im = cv2.imread(os.path.join(path, image))
        annotation = extract_detection_from_file(os.path.join(path, annotation), field)
        yield im, annotation
