from typing import Callable, Iterator, Tuple

from src.stats_computation.interface.classes import *
from src.tracker.tracking_pipeline import track_pipeline
from src.stats_computation.animate import animate
from src.stats_computation.interface.annotations_extraction import *


def convert_track(
    track_fn: Callable, **kwargs
) -> Iterator[Tuple[Image, Detection, float]]:
    track_data = track_fn(**kwargs)
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


def main(
    from_cam: bool = True,
    vis: bool = False,
    check_field_every: int = 50,
    batch_size: int = 25,
) -> None:
    itr_fn = lambda: convert_track(
        track_pipeline,
        from_cam=from_cam,
        vis=vis,
        check_field_every=check_field_every,
        batch_size=batch_size,
    )
    animate(itr_fn, fps=None, save_name="main.json", scroll=False, loop=False)


if __name__ == "__main__":
    main(vis=False, from_cam=True)
