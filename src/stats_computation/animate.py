from typing import Tuple
from dataclasses import dataclass
import time

import numpy as np
import cv2

from src.stats_computation.interface.classes import *
from src.stats_computation.field_state import FieldState


def animate(
    itr: AnnotatedSequence,
    fps: int = 30,
):
    i = 0
    seq = [x for x in itr]
    t0 = time.time()

    print("Press q to close the windows")

    while True:
        if time.time() - t0 > 1 / fps:
            t0 = time.time()
            i += 1

            img, ann = seq[i % len(seq)]
            fs = FieldState(img, ann)
            af = AnimationFrame(fs)
            af.draw()
            af.show()

        if cv2.waitKey(1) == ord("q"):
            # press q to terminate the loop
            cv2.destroyAllWindows()
            break


# Default parameters of AnimationFrame

COLOR_BALL = (0, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_FIELD = (0, 255, 0)

RECTANGLE_THICKNESS = 1


@dataclass
class AnimationFrame:
    fs: FieldState
    color_ball: Tuple[int, int, int] = COLOR_BALL
    color_red: Tuple[int, int, int] = COLOR_RED
    color_blue: Tuple[int, int, int] = COLOR_BLUE
    color_field: Tuple[int, int, int] = COLOR_FIELD
    rectangle_thickness: int = RECTANGLE_THICKNESS

    def __post_init__(self):
        self.main_img = self.fs.image
        self.main_size = self.main_img.shape[:-1][::-1]

    def draw(self):
        self.draw_rectangles_all()

    def show(self):
        cv2.imshow("animation", self.main_img)

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
        if self.fs.detection.ball is not None:
            cv2.rectangle(
                self.main_img,
                *self.convert_rectangles(self.fs.detection.ball),
                self.color_ball,
                self.rectangle_thickness
            )

    def draw_rectangles_players(self) -> None:
        for (color, players) in zip(
            [self.color_red, self.color_blue],
            [self.fs.detection.red_players, self.fs.detection.blue_players],
        ):
            for player in players:
                cv2.rectangle(
                    self.main_img,
                    *self.convert_rectangles(player),
                    color,
                    self.rectangle_thickness
                )

    def draw_rectangles_field(self) -> None:
        for corners in [self.fs.detection.field.corners, self.fs.margin_field.corners]:
            pts = np.array(
                [(corner * self.main_size).astype(int) for corner in corners],
            ).reshape((-1, 1, 2))
            cv2.polylines(
                self.main_img,
                [pts],
                True,
                self.color_field,
                self.rectangle_thickness,
            )

    def draw_rectangles_all(self) -> None:
        self.draw_rectangles_ball()
        self.draw_rectangles_players()
        self.draw_rectangles_field()


if __name__ == "__main__":
    from test.stats_computation.import_test_sequence import import_test_sequence

    itr = import_test_sequence()
    animate(itr)