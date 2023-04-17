from typing import Callable, Iterator, Tuple

from src.stats_computation.interface.classes import Image, Detection
from tracking_pipeline import track_pipeline
from src.stats_computation.animate import animate


def convert_track(
    track_fn: Callable, **kwargs
) -> Iterator[Tuple[Image, Detection, float]]:
    track_data = track_fn(**kwargs)
    for data in track_data:
        yield Detection(
            data["corners"], data["ball"], data["red_players"], data["blue_players"]
        ), data["img"], data["time"]


def main(
    from_cam: bool = True,
    vis: bool = False,
    check_field_every: int = 50,
    batch_size: int = 25,
) -> None:
    itr_fn = lambda: convert_track(
        track_pipeline,
        rom_cam=from_cam,
        vis=vis,
        check_field_every=check_field_every,
        batch_size=batch_size,
    )
    animate(itr_fn, fps=None, save_name="main.json", scroll=False, loop=False)


if __name__ == "__main__":
    main(vis=False, from_cam=True)
