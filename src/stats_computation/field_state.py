from dataclasses import dataclass

from src.stats_computation.interface.classes import *
from src.stats_computation.utils import Perspective, shift_field, parse_players
from src.stats_computation.field_measures import FIELD_HEIGHT

FIELD_MARGIN = 0.1


@dataclass
class FieldState:
    image: Image
    detection: Detection

    def __post_init__(self):
        self.margin_field = self.compute_field_margin()
        self.init_perspective()
        self.compute_relative_position()

    def compute_field_margin(self) -> DetectedField:
        p1, p2, p3, p4 = self.detection.field.corners
        res1 = p1 + (p1 - p3) * FIELD_MARGIN
        res2 = p2 + (p2 - p4) * FIELD_MARGIN
        res3 = p3 + (p3 - p1) * FIELD_MARGIN
        res4 = p4 + (p4 - p2) * FIELD_MARGIN
        return DetectedField([res1, res2, res3, res4])

    def init_perspective(self):
        self.perspective = Perspective(self.detection.field)
        self.shifted_field = shift_field(self.detection.field, FIELD_HEIGHT)
        self.shifted_perspective = Perspective(self.shifted_field)

    def compute_relative_position(self):
        # Ball
        if self.detection.ball is not None:
            self.ball = self.perspective([self.detection.ball.pos])[0]
        # Players
        red_players = self.shifted_perspective(
            [player.pos for player in self.detection.red_players]
        )
        blue_players = self.shifted_perspective(
            [player.pos for player in self.detection.blue_players]
        )
        self.is_red_up = (
            np.min(red_players, axis=0)[1] < np.min(blue_players, axis=0)[1]
        )
        self.red_players = parse_players(red_players, self.is_red_up)
        self.blue_players = parse_players(blue_players, not self.is_red_up)
