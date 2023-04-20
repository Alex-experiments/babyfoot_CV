from typing import Tuple, Callable
import os

import numpy as np
import pandas as pd
import cv2

from src.stats_computation.interface.classes import *
from src.tracker.tracking_pipeline import track_pipeline

LABEL_BALL = 0
LABEL_RED_PLAYER = 1
LABEL_BLUE_PLAYER = 2


def extract_object_from_annotation(
    annotation: pd.DataFrame,
    label: int,
    convert: bool = False,
    im_size: np.ndarray = None,
) -> List[List[Coordinates]]:
    objects = annotation.loc[annotation["label"] == label].to_dict("records")
    res = [
        [np.array([object["x"], object["y"]]), np.array([object["lx"], object["ly"]])]
        for object in objects
    ]
    if convert:
        res = [convert_coord(x, im_size) for x in res]
    return res


def extract_field_from_file(filepath: str, im_size: np.ndarray) -> DetectedField:
    with open(filepath) as f:
        for line in f:
            if "CORNERS" in line:
                corners = line.split(", [")[1:]
                corners = [corner.split("]")[0] for corner in corners]
                corners = [corner.split(", ") for corner in corners]
                corners = [[int(corner[0]), int(corner[1])] for corner in corners]
                corners = [
                    np.array([corner[0] / im_size[1], corner[1] / im_size[0]])
                    for corner in corners
                ]
                return DetectedField(corners)


def get_skiprows(filepath: str) -> List[int]:
    res = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if "CORNERS" in line:
                res.append(i)
    return res


def convert_coord(
    obj: List[Coordinates],
    im_size: np.ndarray,
) -> List[Coordinates]:
    pos, width = obj
    res = [
        np.array(
            [
                pos[0] * im_size[0] / im_size[1],
                pos[1] * im_size[1] / im_size[0],
            ]
        ),
        np.array(
            [
                width[0] * im_size[0] / im_size[1],
                width[1] * im_size[1] / im_size[0],
            ]
        ),
    ]
    return res


def extract_detection_from_file(
    filepath: str,
    im_size: np.ndarray,
    field: DetectedField = None,
    sep: str = " ",
    convert: bool = False,
) -> Detection:
    # Import from csv
    skiprows = get_skiprows(filepath)
    annotation = pd.read_csv(
        filepath, sep=sep, names=["label", "x", "y", "lx", "ly"], skiprows=skiprows
    )

    # Field extraction
    if field is None:
        field = extract_field_from_file(filepath, im_size)

    # Ball extraction
    balls = extract_object_from_annotation(
        annotation,
        LABEL_BALL,
        convert=convert,
        im_size=im_size,
    )
    if len(balls) > 1:
        ball = None
        print(f"Warning: ball detection problem ({len(balls)} balls detected)")
    elif len(balls) == 0:
        # Not an error because the ball can be hidden or go out of the field
        ball = None
    else:
        ball = DetectedBall(*(balls[0]))

    # Red players extraction
    red_players = [
        DetectedRedPlayer(*obj)
        for obj in extract_object_from_annotation(
            annotation,
            LABEL_RED_PLAYER,
            convert=convert,
            im_size=im_size,
        )
    ]

    # Blue players extraction
    blue_players = [
        DetectedBluePlayer(*obj)
        for obj in extract_object_from_annotation(
            annotation,
            LABEL_BLUE_PLAYER,
            convert=convert,
            im_size=im_size,
        )
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
    path: str, field: DetectedField = None, sep: str = " ", convert: bool = False
) -> DetectionSequence:
    images, annotations = list_files(path)
    for image, annotation in zip(images, annotations):
        im = cv2.imread(os.path.join(path, image))
        im_size = im.shape
        annotation = extract_detection_from_file(
            os.path.join(path, annotation),
            im_size,
            field,
            sep=sep,
            convert=convert,
        )
        yield im, annotation


def convert_track(**kwargs) -> Iterator[Tuple[Image, Detection, float]]:
    track_data = track_pipeline(**kwargs)
    for data in track_data:
        im_size = data["img"].shape
        # Field
        corners = data["corners"]
        corners = [
            np.array([corner[0] / im_size[1], corner[1] / im_size[0]])
            for corner in corners
        ]
        field = DetectedField(corners)
        # Ball
        if data["ball"] is None:
            # Not an error because the ball can be hidden or go out of the field
            ball = None
        else:
            objects = [data["ball"]]
            balls = [
                [
                    np.array([object[0], object[1]]),
                    np.array([object[2], object[3]]),
                ]
                for object in objects
            ]
            balls = [convert_coord(x, im_size) for x in balls]
            if len(balls) > 1:
                ball = None
                print(f"Warning: ball detection problem ({len(balls)} balls detected)")
            else:
                ball = DetectedBall(*(balls[0]))

        # Red players
        objects = data["red_players"]
        red_players = [
            [
                np.array([object[0], object[1]]),
                np.array([object[2], object[3]]),
            ]
            for object in objects
        ]
        red_players = [convert_coord(x, im_size) for x in red_players]
        red_players = [DetectedRedPlayer(*obj) for obj in red_players]
        # Blue players
        objects = data["blue_players"]
        blue_players = [
            [
                np.array([object[0], object[1]]),
                np.array([object[2], object[3]]),
            ]
            for object in objects
        ]
        blue_players = [convert_coord(x, im_size) for x in blue_players]
        blue_players = [DetectedBluePlayer(*obj) for obj in blue_players]

        yield data["img"], Detection(field, ball, red_players, blue_players), data[
            "time"
        ]


def curry_convert_track(
    **kwargs,
) -> Callable[[], Iterator[Tuple[Image, Detection, float]]]:
    return lambda: convert_track(**kwargs)
