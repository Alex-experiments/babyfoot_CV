from dataclasses import dataclass
import time

from src.stats_computation.interface.classes import *
from src.stats_computation.game_state import *
from src.stats_computation.field_measures import *
from src.stats_computation.utils import distance_cm

BALL_MOVEMENT_THRESHOLD = 3  # Threshold (in cm/s) to consider the ball in movement


@dataclass
class GlobalStats:
    score_red: int = 0
    score_blue: int = 0
    duration: float = 0.0  # Time in seconds
    ball_total_distance: float = 0.0  # Total dispance covered by the ball (in cm)
    ball_max_speed: float = (
        None  # Maximum speed of the ball between two frames (in cm/s)
    )
    ball_avg_speed: float = None  # Average speed of the ball (in cm/s)


@dataclass
class Events:
    frame: int


@dataclass
class PlayerPosition:
    team: str  # "red" or "blue"
    position: str  # "goal", "defense", "middle" or "attack"
    idx: int  # in range(5)


class StatsExtraction(GameState):
    fps: int = None  # if None time is computed live

    def __post_init__(self):
        super().__post_init__()
        # Initialization variables
        self.t0 = time.time()
        self.global_stats = GlobalStats()
        self.events = []
        # Internal variables (need to be updated each frame)
        self.time = []
        self.ball_displacement = (
            []
        )  # displacement of the ball between two frames (in cm)
        self.ball_speed = []  # speed of the ball between two frames (in cm/s)
        self.ball_movement = []  # True if the ball speed is above a threshold
        self.distance_ball_red_players = (
            []
        )  # distance (in cm) between the ball and every red player in the form {"goal": [...], "defense": [...], ...}
        self.distance_ball_blue_players = (
            []
        )  # distance (in cm) between the ball and every red player in the form {"goal": [...], "defense": [...], ...}
        self.closest_player_distance = (
            []
        )  # distance (in cm) between the ball and the closest player
        self.closest_player_position = (
            []
        )  # closest player from the ball, of type PlayerPosition

    def update(self, detection: Detection):
        super().update(detection)
        self.update_internal_variables()
        self.detect_events()
        self.update_global_stats()
        print(self.global_stats)
        print(self.ball_displacement[-1])
        print(
            self.closest_player_distance[-1],
            self.closest_player_position[-1],
        )
        print(self.ball_speed[-1])
        print(self.ball_movement[-1])

    def update_internal_variables(self) -> None:
        # Duration
        self.time.append(self.get_time())
        # Ball displacement and speed
        self.update_ball_displacement_speed()
        self.update_ball_movement()
        # Distance between the ball and the players
        self.update_distance_ball_players()
        # Closest player
        self.update_closest_player()

    def detect_events(self) -> None:
        pass

    def update_global_stats(self) -> None:
        # Duration
        self.global_stats.duration = self.time[-1]
        # Ball displacement and speed
        self.update_global_ball_displacement_speed()

    def get_time(self) -> float:
        if self.fps is None:
            return time.time() - self.t0
        return self.frame_idx / self.fps

    def update_ball_displacement_speed(self):
        # TODO improvement: with None take the last frame that is not None
        if self.frame_idx > 1:
            # Ball displacement
            self.ball_displacement.append(
                distance_cm(self.ball, self.history[-1]["ball"])
            )
            # Ball speed
            if self.ball_displacement[-1] is not None:
                self.ball_speed.append(
                    self.ball_displacement[-1] / (self.time[-1] - self.time[-2])
                )
        else:
            self.ball_displacement.append(None)
            self.ball_speed.append(None)

    def update_ball_movement(self):
        ball_speed = self.ball_speed[-1]
        if ball_speed is None:
            self.ball_movement.append(None)
        else:
            self.ball_movement.append(ball_speed > BALL_MOVEMENT_THRESHOLD)

    def update_global_ball_displacement_speed(self):
        # Ball displacement
        if self.ball_displacement[-1] is not None:
            self.global_stats.ball_total_distance += self.ball_displacement[-1]
        # Ball speed
        if self.ball_speed[-1] is not None:
            # Ball max speed
            if (
                self.global_stats.ball_max_speed is None
                or self.global_stats.ball_max_speed < self.ball_speed[-1]
            ):
                self.global_stats.ball_max_speed = self.ball_speed[-1]
            # Ball average speed
            if self.global_stats.ball_avg_speed is None:
                self.global_stats.ball_avg_speed = self.ball_speed[-1]
            nb_not_None = len(
                [x for x in self.ball_speed if x is not None]
            )  # TODO can be improved
            self.global_stats.ball_avg_speed = (
                nb_not_None * self.global_stats.ball_avg_speed + self.ball_speed[-1]
            ) / (nb_not_None + 1)

    def update_distance_ball_players(self):
        for (players, distances) in [
            (self.red_players, self.distance_ball_red_players),
            (self.blue_players, self.distance_ball_blue_players),
        ]:
            res = dict()
            for position in ["goal", "defense", "middle", "attack"]:
                res[position] = [
                    distance_cm(self.ball, player) for player in players[position]
                ]
            distances.append(res)

    def update_closest_player(self):
        closest_player_distance = None
        closest_player_position = None
        for (team, distances) in [
            ("red", self.distance_ball_red_players[-1]),
            ("blue", self.distance_ball_blue_players[-1]),
        ]:
            for position in ["goal", "defense", "middle", "attack"]:
                for idx in range(len(distances[position])):
                    d = distances[position][idx]
                    if d is not None and (
                        closest_player_distance is None or d < closest_player_distance
                    ):
                        closest_player_distance = d
                        closest_player_position = PlayerPosition(team, position, idx)
        self.closest_player_distance.append(closest_player_distance)
        self.closest_player_position.append(closest_player_position)
