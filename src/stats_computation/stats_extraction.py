from dataclasses import dataclass
import time

from src.stats_computation.interface.classes import *
from src.stats_computation.game_state import *
from src.stats_computation.field_measures import *
from src.stats_computation.utils import *


SHOT_ACCELERATION_THRESHOLD = (
    5000  # Above this acceleration (in cm/s^2) a shot is triggered
)
DISAPPEAR_BALL_NB_FRAME = 5  # Number of frames after which the ball is considered out


@dataclass
class GlobalStats:
    score_red: int = 0
    score_blue: int = 0
    duration: float = 0.0  # Time in seconds
    ball_total_distance: float = 0.0  # Total dispance covered by the ball (in cm)
    ball_max_speed: float = (
        None  # Maximum speed of the ball between two frames (in cm/s)
    )


@dataclass
class Events:
    frame: int


@dataclass
class PlayerPosition:
    team: str  # "red" or "blue"
    position: str  # "goal", "defense", "middle" or "attack"
    idx: int  # in range(5)


@dataclass
class Shot(Events):
    player: PlayerPosition
    ball_speed: float


@dataclass
class Goal(Events):
    team: str  # "red" or "blue"


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
        self.ball_speed = []  # speed vector of the ball between two frames (in cm/s)
        self.ball_speed_norm = []  # speed of the ball between two frames (in cm/s)
        self.ball_acceleration = (
            []
        )  # acceleration vector of the ball between two frames (in cm/s^2)
        self.ball_acceleration_norm = (
            []
        )  # acceleration of the ball between two frames (in cm/s^2)
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
        self.ball_displacement_angle = (
            []
        )  # Angle (in °) between the displacement vector and [1, 0]
        self.ball_angle_diff = (
            []
        )  # Difference in angle (in °) between the two last displacements

        # Events variables
        self.no_ball_count = -1

    def update(self, detection: Detection):
        super().update(detection)
        self.update_internal_variables()

        print("Déplacement ball :", self.ball_displacement[-1])
        print("Vitesse ball :", self.ball_speed_norm[-1])
        print("Acceleration ball :", self.ball_acceleration_norm[-1])
        print(
            "Closest player:",
            self.closest_player_position[-1],
            self.closest_player_distance[-1],
        )
        print("Angle diff :", self.ball_angle_diff[-1])

        self.detect_events()
        self.update_global_stats()

        print(self.global_stats)

    def update_internal_variables(self) -> None:
        # Duration
        self.time.append(self.get_time())
        # Ball displacement, speed, acceleration
        self.update_ball_displacement()
        self.update_ball_speed()
        self.update_ball_acceleration()
        # Distance between the ball and the players, closest player
        self.update_distance_ball_players()
        self.update_closest_player()
        # Ball displacement angles
        self.update_angles()

    def detect_events(self) -> None:
        self.detect_shot()
        self.detect_goal()

    def update_global_stats(self) -> None:
        # Duration
        self.global_stats.duration = self.time[-1]
        # Ball displacement and speed
        self.update_global_ball_displacement_speed()

    def get_time(self) -> float:
        if self.fps is None:
            return time.time() - self.t0
        return self.frame_idx / self.fps

    def update_ball_displacement(self):
        if self.frame_idx > 1:
            displacement = distance_cm(self.ball, self.history[-1]["ball"])
            self.ball_displacement.append(displacement)
        else:
            self.ball_displacement.append(None)

    def update_ball_speed(self):
        if (
            self.frame_idx > 1
            and self.ball is not None
            and self.history[-1]["ball"] is not None
        ):
            self.ball_speed.append(
                (self.ball - self.history[-1]["ball"]) / (self.time[-1] - self.time[-2])
            )
        else:
            self.ball_speed.append(None)
        self.ball_speed_norm.append(norm_cm(self.ball_speed[-1]))

    def update_ball_acceleration(self):
        if (
            self.frame_idx > 1
            and self.ball_speed[-1] is not None
            and self.ball_speed[-2] is not None
        ):
            self.ball_acceleration.append(
                (self.ball_speed[-1] - self.ball_speed[-2])
                / (self.time[-1] - self.time[-2])
            )
        else:
            self.ball_acceleration.append(None)
        self.ball_acceleration_norm.append(norm_cm(self.ball_acceleration[-1]))

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

    def update_angles(self):
        if self.frame_idx > 1:
            angle = angle_deg(self.ball, self.history[-1]["ball"])
            self.ball_displacement_angle.append(angle)
            # Difference in angles
            last_angle = self.ball_displacement_angle[-2]
            if angle is not None and last_angle is not None:
                diff_angle = angle - last_angle
                if diff_angle > 180:
                    diff_angle -= 360
                elif diff_angle <= -180:
                    diff_angle += 360
                self.ball_angle_diff.append(diff_angle)
            else:
                self.ball_angle_diff.append(None)
        else:
            self.ball_displacement_angle.append(None)
            self.ball_angle_diff.append(None)

    def change_closest_player(self) -> bool:
        if self.frame_idx > 1:
            previous = self.closest_player_position[-1]
            next = self.closest_player_position[-2]
            if previous is None or next is None:
                return False
            return previous != next
        return False

    def get_disappear_ball(self) -> bool:
        if self.no_ball_count == -1:
            # It means the ball was out: wait until it reappears
            if self.ball is not None:
                self.no_ball_count = 0
            return False
        if self.ball is None:
            self.no_ball_count += 1
        if self.no_ball_count == DISAPPEAR_BALL_NB_FRAME:
            self.no_ball_count = -1
            return True
        return False

    def detect_shot(self):
        if self.change_closest_player():
            if self.ball_acceleration_norm[-1] > SHOT_ACCELERATION_THRESHOLD:
                shot = Shot(
                    self.frame_idx,
                    self.closest_player_position[-2],
                    self.ball_speed_norm[-1],
                )
                print(f"===== {shot} =====")
                self.events.append(shot)

    def detect_goal(self):
        if self.get_disappear_ball():
            last_ball = self.history[-DISAPPEAR_BALL_NB_FRAME]["ball"]
            # Test if last position inside goal
            up1, down1 = in_goal_up_down(last_ball)
            # Test if last position + displacement inside goal
            if self.frame_idx > DISAPPEAR_BALL_NB_FRAME + 1:
                penult_ball = self.history[-DISAPPEAR_BALL_NB_FRAME - 1]["ball"]
                if penult_ball is not None:
                    up2, down2 = in_goal_up_down(last_ball + (last_ball - penult_ball))
                    up1 = up1 or up2
                    down1 = down1 or down2
            # Trigger event
            if up1 or down1:
                if up1 and self.is_red_up:
                    team = "blue"
                    self.global_stats.score_blue += 1
                else:
                    team = "red"
                    self.global_stats.score_red += 1
            goal = Goal(self.frame_idx, team)
            print(f"===== {goal} =====")

    def update_global_ball_displacement_speed(self):
        # Ball displacement
        if self.ball_displacement[-1] is not None:
            self.global_stats.ball_total_distance += self.ball_displacement[-1]
        # Ball speed
        if self.ball_speed_norm[-1] is not None:
            # Ball max speed
            if (
                self.global_stats.ball_max_speed is None
                or self.global_stats.ball_max_speed < self.ball_speed_norm[-1]
            ):
                self.global_stats.ball_max_speed = self.ball_speed_norm[-1]
