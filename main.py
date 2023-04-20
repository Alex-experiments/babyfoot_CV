from src.stats_computation.animate import animate
from src.stats_computation.interface.annotations_extraction import *


def main(
    from_cam: bool = True,
    vis: bool = False,
    check_field_every: int = 50,
    batch_size: int = 25,
) -> None:
    itr_fn = curry_convert_track(
        from_cam=from_cam,
        vis=vis,
        check_field_every=check_field_every,
        batch_size=batch_size,
    )
    animate(itr_fn, fps=None, save_name="main.json", scroll=False, loop=False)


if __name__ == "__main__":
    main(vis=False, from_cam=True)
