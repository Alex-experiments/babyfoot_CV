from typing import Tuple
import os

import numpy as np
import pandas as pd
import cv2

from src.stats_computation.interface.classes import *

LABEL_BALL = 0
LABEL_RED_PLAYER = 1
LABEL_BLUE_PLAYER = 2


def extract_object_from_annotation(
    annotation: pd.DataFrame,
    label: int,
    convert: bool = False,
    im_size: np.ndarray = None,
    raw_corners: List[Coordinates] = None,
) -> List[List[Coordinates]]:
    objects = annotation.loc[annotation["label"] == label].to_dict("records")
    res = [
        [np.array([object["x"], object["y"]]), np.array([object["lx"], object["ly"]])]
        for object in objects
    ]
    # if convert:
    #     res = [convert_coord(x, im_size, raw_corners) for x in res]
    return res


def extract_field_from_file(
    filepath: str, im_size: np.ndarray
) -> Tuple[DetectedField, List[Coordinates]]:
    with open(filepath) as f:
        for line in f:
            if "CORNERS" in line:
                corners = line.split(", [")[1:]
                corners = [corner.split("]")[0] for corner in corners]
                corners = [corner.split(", ") for corner in corners]
                corners = [[int(corner[0]), int(corner[1])] for corner in corners]
                raw_corners = corners
                corners = [
                    np.array([corner[0] / im_size[0], corner[1] / im_size[1]])
                    for corner in corners
                ]
                # corners = [
                #     np.array([corner[0] / im_size[1], corner[1] / im_size[0]])
                #     for corner in corners
                # ]
                return DetectedField(corners), raw_corners


def get_skiprows(filepath: str) -> List[int]:
    res = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if "CORNERS" in line:
                res.append(i)
    return res


def convert_coord(
    pts: List[Coordinates],
    im_size: np.ndarray,
    raw_corners: List[Coordinates],
) -> Coordinates:
    pts1 = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    pts2 = np.float32(raw_corners)
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_pts = cv2.perspectiveTransform(np.array(pts).reshape(1, -1, 2), matrix)
    res = []
    for coord in transformed_pts.reshape(-1, 2):
        res.append(np.array([coord[0] / im_size[1], coord[1] / im_size[0]]))
    print(res)
    return res
    # res = []
    # for coord in pts:
    #     res.append(
    #         np.array(
    #             [coord[0] / im_size[0] * im_size[1], coord[1] / im_size[1] * im_size[0]]
    #         )
    #     )
    # return res


def extract_detection_from_file(
    filepath: str,
    im_size: np.ndarray,
    field: DetectedField = None,
    sep: str = " ",
    convert: bool = False,
) -> Detection:
    skiprows = get_skiprows(filepath)
    annotation = pd.read_csv(
        filepath, sep=sep, names=["label", "x", "y", "lx", "ly"], skiprows=skiprows
    )
    # Field extraction
    if field is None:
        field, raw_corners = extract_field_from_file(filepath, im_size)
    else:
        raw_corners = None

    # Ball extraction
    balls = extract_object_from_annotation(
        annotation,
        LABEL_BALL,
        convert=convert,
        im_size=im_size,
        raw_corners=raw_corners,
    )
    ball = None if len(balls) == 0 else DetectedBall(*(balls[0]))

    # Red players extraction
    red_players = [
        DetectedRedPlayer(*obj)
        for obj in extract_object_from_annotation(
            annotation,
            LABEL_RED_PLAYER,
            convert=convert,
            im_size=im_size,
            raw_corners=raw_corners,
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
            raw_corners=raw_corners,
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
