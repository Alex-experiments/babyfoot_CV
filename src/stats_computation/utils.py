from typing import Dict, List

import numpy as np
import cv2

from src.stats_computation.interface.classes import *
from src.stats_computation.field_measures import *


class Perspective:
    def __init__(self, field: DetectedField):
        pts1 = np.float32(field.corners)
        pts2 = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.matrix = cv2.getPerspectiveTransform(pts1, pts2)

    def __call__(self, pts: List[Coordinates]) -> List[Coordinates]:
        transformed_pts = cv2.perspectiveTransform(
            np.array(pts).reshape(1, -1, 2), self.matrix
        )
        return transformed_pts.reshape(-1, 2)


def shift_field(field: DetectedField, rel_height: float) -> DetectedField:
    """Very experimental"""
    pt1, pt2, pt3, pt4 = field.corners
    d1 = np.linalg.norm(pt1 - pt2)
    d2 = np.linalg.norm(pt2 - pt3)
    d3 = np.linalg.norm(pt3 - pt4)
    d4 = np.linalg.norm(pt4 - pt1)
    tilt = max(d1 / d3, d3 / d1, d2 / d4, d4 / d2) - 1
    shift = np.array([0, tilt * rel_height])
    return DetectedField([pt1 - shift, pt2 - shift, pt3 - shift, pt4 - shift])


def parse_players(players: List[Coordinates], up: bool) -> Dict[str, List[Coordinates]]:
    res = {"goal": [], "defense": [], "middle": [], "attack": []}
    for player in players:
        pos_x = player[1] if up else 1 - player[1]
        if pos_x < (GOAL_REL_X + DEFENSE_REL_X) / 2:
            res["goal"].append(player)
        elif pos_x < (DEFENSE_REL_X + MIDDLE_REL_X) / 2:
            res["defense"].append(player)
        elif pos_x < (MIDDLE_REL_X + ATTACK_REL_X) / 2:
            res["middle"].append(player)
        else:
            res["attack"].append(player)
    assert len(res["goal"]) == 1
    assert len(res["defense"]) == 2
    assert len(res["middle"]) == 5
    assert len(res["attack"]) == 3
    res["goal"] = smooth_players(res["goal"], GOAL_REL_X if up else 1 - GOAL_REL_X, 0)
    res["defense"] = smooth_players(
        res["defense"], DEFENSE_REL_X if up else 1 - DEFENSE_REL_X, DEFENSE_REL_W
    )
    res["middle"] = smooth_players(
        res["middle"], MIDDLE_REL_X if up else 1 - MIDDLE_REL_X, MIDDLE_REL_W
    )
    res["attack"] = smooth_players(
        res["attack"], ATTACK_REL_X if up else 1 - ATTACK_REL_X, ATTACK_REL_W
    )
    return res


def smooth_players(
    players: List[Coordinates], rel_x_bar: float, rel_w_bar: float
) -> List[Coordinates]:
    if len(players) == 1:
        return [np.array([players[0][0], rel_x_bar])]
    mid = np.mean(players, axis=0)[0]
    return [
        np.array([mid - rel_w_bar / 2 + i * rel_w_bar / (len(players) - 1), rel_x_bar])
        for i in range(len(players))
    ]


def distance_cm(pt1: Coordinates, pt2: Coordinates) -> float:
    if pt1 is None or pt2 is None:
        return None
    vect = pt1 - pt2
    res = (vect[0] * FIELD_WIDTH) ** 2 + (vect[1] * FIELD_LENGTH) ** 2
    return res**0.5 * CM_PER_UNIT


def angle_deg(pt1: Coordinates, pt2: Coordinates) -> float:
    if pt1 is None or pt2 is None:
        return None
    vect = pt1 - pt2
    vect_cm = np.array([vect[0] * FIELD_WIDTH, vect[1] * FIELD_LENGTH]) * CM_PER_UNIT
    normalized = vect_cm / np.linalg.norm(vect_cm)
    angle = np.arccos(normalized[0])
    if normalized[1] < 0:
        angle *= -1
    return angle * 180 / np.pi
