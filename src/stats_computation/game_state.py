from typing import Dict, List
from dataclasses import dataclass

from src.stats_computation.interface.classes import *
from src.stats_computation.utils import Perspective, shift_field, parse_players
from src.stats_computation.field_measures import PLAYERS_HEIGHT

FIELD_MARGIN = 0.1


@dataclass
class GameState:
    ball: Coordinates = None  # ball relative position
    # for red or blue team, the key is the position of the players (goal | defense | middle | attack)
    # and the value is the list of relative players' position
    red_players: Dict[str, List[Coordinates]] = None
    blue_players: Dict[str, List[Coordinates]] = None

    def __post_init__(self):
        self.frame_idx = 0  # idx of the current frame
        # Internal variables
        self.margin_field = None  # field with margin, used to ignore what's outside
        self.perspective = (
            None  # used to transform absolute to relative positions in the field
        )
        self.shifted_field = None  # the players are shifted from the field due to perspective, used to correct this
        self.shifted_perspective = None  # used to transform absolute to relative positions for the players (since they are shifted)
        self.is_red_up = (
            None  # True if red team is above (meaning their goal is at y = 0)
        )
        self.history = (
            []
        )  # list of reccords of the form {"ball": ..., "red_players": ..., "blue_players": ...}, the last is of index -1

    def update(self, detection: Detection):
        self.frame_idx += 1
        if self.frame_idx > 1:
            self.update_history()
        self.update_field_margin(detection)
        self.update_perspective(detection)
        self.update_relative_positions(detection)

    def update_field_margin(self, detection: Detection) -> None:
        # TODO use these margins to remove detections outside
        # TODO maybe move this to Detection
        p1, p2, p3, p4 = detection.field.corners
        res1 = p1 + (p1 - p3) * FIELD_MARGIN
        res2 = p2 + (p2 - p4) * FIELD_MARGIN
        res3 = p3 + (p3 - p1) * FIELD_MARGIN
        res4 = p4 + (p4 - p2) * FIELD_MARGIN
        self.margin_field = DetectedField([res1, res2, res3, res4])

    def update_perspective(self, detection: Detection) -> None:
        self.perspective = Perspective(detection.field)
        self.shifted_field = shift_field(detection.field, PLAYERS_HEIGHT)
        self.shifted_perspective = Perspective(self.shifted_field)

    def update_relative_positions(self, detection: Detection) -> None:
        # Ball
        if detection.ball is not None:
            self.ball = self.perspective([detection.ball.pos])[0]
        else:
            self.ball = None
        # Players
        red_players = self.shifted_perspective(
            [player.pos for player in detection.red_players]
        )
        blue_players = self.shifted_perspective(
            [player.pos for player in detection.blue_players]
        )
        self.is_red_up = (
            np.min(red_players, axis=0)[1] < np.min(blue_players, axis=0)[1]
        )
        self.red_players = parse_players(red_players, self.is_red_up)
        self.blue_players = parse_players(blue_players, not self.is_red_up)
        # try:
        #     self.red_players = parse_players(red_players, self.is_red_up)
        # except Exception:
        #     print("error occured while parsing players")
        #     zero = np.array([0.0, 0.0])
        #     self.red_players = {
        #         "goal": [zero],
        #         "defense": [zero] * 2,
        #         "middle": [zero] * 5,
        #         "attack": [zero] * 3,
        #     }

        # try:
        #     self.blue_players = parse_players(blue_players, not self.is_red_up)
        # except Exception:
        #     print("error occured while parsing players")
        #     zero = np.array([0.0, 0.0])
        #     self.blue_players = {
        #         "goal": [zero],
        #         "defense": [zero] * 2,
        #         "middle": [zero] * 5,
        #         "attack": [zero] * 3,
        #     }

    def update_history(self) -> None:
        current_data = {
            "ball": self.ball,
            "red_players": self.red_players,
            "blue_players": self.blue_players,
        }
        self.history.append(current_data)
