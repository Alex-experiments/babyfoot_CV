from typing import Tuple, Callable
from dataclasses import dataclass
import time

import numpy as np
import cv2

from src.stats_computation.interface.classes import *
from src.stats_computation.stats_extraction import *
from src.stats_computation.field_measures import *


def animate(
    itr_fn: Callable[[], DetectionSequence],
    fps: int = 30,
    save_name: str = None,
    scroll: bool = False,
    loop: bool = False,
):
    itr = itr_fn()
    t0 = time.time()
    anim = Animation(fps=fps, save_name=save_name)

    print("Press q to close the windows")
    if scroll:
        print("Press any key to get the next frame")

    while True:
        if time.time() - t0 > 1 / fps:
            t0 = time.time()

            try:
                data = next(itr)
            except StopIteration:
                anim.save()
                if loop:
                    itr = itr_fn()
                    data = next(itr)
                else:
                    break

            img, ann = data[:2]
            if len(data) == 3:
                current_time = data[2]
            else:
                current_time = None

            anim.update(ann, img, current_time=current_time)
            anim.draw()
            anim.show()

            delta_t = time.time() - t0
            # print(f"Update time: {delta_t}, faster than fps: {delta_t < 1 / fps}\n")

        if scroll:
            key = cv2.waitKey(0)
        else:
            key = cv2.waitKey(1)

        if key == ord("q"):
            # press q to terminate the loop
            cv2.destroyAllWindows()
            break

    anim.show_stats()


# Default parameters of AnimationFrame

RPR_FIELD_LENGTH = 600
RPR_MARGIN = 0.1

COLOR_BALL = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_FIELD = (0, 255, 0)

RECTANGLE_THICKNESS = 1


@dataclass
class Animation(StatsExtraction):
    rpr_field_length: int = RPR_FIELD_LENGTH
    rpr_field_width: int = int(RPR_FIELD_LENGTH * FIELD_WIDTH / FIELD_LENGTH)
    rpr_ball_radius: int = int(RPR_FIELD_LENGTH * BALL_RADIUS / FIELD_LENGTH)
    rpr_margin: float = RPR_MARGIN
    color_ball: Tuple[int, int, int] = COLOR_BALL
    color_red: Tuple[int, int, int] = COLOR_RED
    color_blue: Tuple[int, int, int] = COLOR_BLUE
    color_field: Tuple[int, int, int] = COLOR_FIELD
    rectangle_thickness: int = RECTANGLE_THICKNESS

    def __post_init__(self):
        super().__post_init__()
        # Internal variables
        self.detection = None  # current detection
        self.main_img = (
            None  # main window image, representing the image from the webcam
        )
        self.main_size = None  # main window image size
        self.rpr_img = None  # representation image of the game (2d projection, obtained with relative positions)
        self.rpr_size = None  # representation image size

    def update(self, detection: Detection, image: Image, current_time: float = None):
        super().update(detection, current_time=current_time)
        self.detection = detection
        # Main window
        self.main_img = image
        self.main_size = np.array(self.main_img.shape[:-1][::-1])

        # Representation window
        self.rpr_img = np.zeros(
            shape=[
                int(self.rpr_field_length * (1 + 2 * self.rpr_margin)),
                int(self.rpr_field_width * (1 + 2 * self.rpr_margin)),
                3,
            ],
            dtype=np.uint8,
        )
        self.rpr_size = np.array([self.rpr_field_width, self.rpr_field_length])
        self.rpr_margin_vect = np.array(
            [
                self.rpr_field_width * self.rpr_margin,
                self.rpr_field_length * self.rpr_margin,
            ]
        )

    def draw(self):
        self.draw_main()
        self.draw_representation()

    def show(self):
        cv2.imshow("Animation", self.main_img)
        cv2.imshow("Representation", self.rpr_img)

    def convert_rectangles(
        self, obj: DetectedObject
    ) -> Tuple[Coordinates, Coordinates]:
        """Return (start_point, end_point) in the format of cv2.rectangle()"""
        pos, width = obj.pos, obj.width
        start = pos - width / 2
        end = pos + width / 2
        start = (start * self.main_size).astype(int)
        end = (end * self.main_size).astype(int)
        return start, end

    def draw_rectangles_ball(self) -> None:
        if self.detection.ball is not None:
            cv2.rectangle(
                self.main_img,
                *self.convert_rectangles(self.detection.ball),
                self.color_ball,
                self.rectangle_thickness,
            )

    def draw_rectangles_players(self) -> None:
        for (color, players) in zip(
            [self.color_red, self.color_blue],
            [self.detection.red_players, self.detection.blue_players],
        ):
            for player in players:
                cv2.rectangle(
                    self.main_img,
                    *self.convert_rectangles(player),
                    color,
                    self.rectangle_thickness,
                )

    def draw_rectangles_field(self) -> None:
        for field in [
            self.detection.field,
            self.margin_field,
            self.shifted_field,
        ]:
            pts = np.array(
                [(corner * self.main_size).astype(int) for corner in field.corners],
            ).reshape(-1, 1, 2)
            cv2.polylines(
                self.main_img,
                [pts],
                True,
                self.color_field,
                self.rectangle_thickness,
            )

    def draw_main(self) -> None:
        self.draw_rectangles_ball()
        self.draw_rectangles_players()
        self.draw_rectangles_field()

    def draw_bar_representation(
        self, rel_x: float, color: Tuple[int, int, int]
    ) -> None:
        pt1 = np.array([0, rel_x])
        pt2 = np.array([1, rel_x])
        cv2.line(
            self.rpr_img,
            self.convert_relative_position(pt1),
            self.convert_relative_position(pt2),
            color,
            self.rectangle_thickness,
        )

    def draw_field_representation(self) -> None:
        pt1 = np.array([0, 0])
        pt2 = np.array([1, 1])
        cv2.rectangle(
            self.rpr_img,
            self.convert_relative_position(pt1),
            self.convert_relative_position(pt2),
            self.color_field,
            self.rectangle_thickness,
        )
        self.draw_bar_representation(0.5, self.color_field)

        if self.is_red_up:
            color_up = self.color_red
            color_down = self.color_blue
        else:
            color_up = self.color_blue
            color_down = self.color_red

        # Team on top
        self.draw_bar_representation(GOAL_REL_X, color_up)
        self.draw_bar_representation(DEFENSE_REL_X, color_up)
        self.draw_bar_representation(MIDDLE_REL_X, color_up)
        self.draw_bar_representation(ATTACK_REL_X, color_up)

        # Team on bottom
        self.draw_bar_representation(1 - GOAL_REL_X, color_down)
        self.draw_bar_representation(1 - DEFENSE_REL_X, color_down)
        self.draw_bar_representation(1 - MIDDLE_REL_X, color_down)
        self.draw_bar_representation(1 - ATTACK_REL_X, color_down)

    def convert_relative_position(self, rel_pos: Coordinates) -> Coordinates:
        return (rel_pos * self.rpr_size + self.rpr_margin_vect).astype(int)

    def draw_ball_representation(self) -> None:
        if self.ball is not None:
            cv2.circle(
                self.rpr_img,
                self.convert_relative_position(self.ball),
                self.rpr_ball_radius,
                self.color_ball,
                -1,
            )

    def draw_players_representation(self) -> None:
        for (color, players) in zip(
            [self.color_red, self.color_blue],
            [self.red_players, self.blue_players],
        ):
            for position in players:
                for player in players[position]:
                    pts = [
                        self.convert_relative_position(player)
                        + coord * self.rpr_ball_radius
                        for coord in [
                            np.array([-1, -1]),
                            np.array([1, 1]),
                        ]
                    ]
                    cv2.rectangle(self.rpr_img, *pts, color, -1)

    def draw_representation(self) -> None:
        self.draw_field_representation()
        self.draw_ball_representation()
        self.draw_players_representation()


if __name__ == "__main__":
    from test.stats_computation.import_test_sequence import *

    itr_fn = import_test_sequence
    fps = TEST_FPS
    # itr_fn = import_camera_example_sequence
    # fps = CAMERA_EXAMPLE_FPS
    animate(itr_fn, fps=fps, save_name="experiment.json", scroll=False, loop=False)
