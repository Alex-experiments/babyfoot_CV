import os
import random

import tqdm
import cv2

DEFAULT_INPUT_FOLDER_PATH = os.path.join("dataset_creation", "videos")
DEFAULT_OUTPUT_FOLDER_PATH = os.path.join("dataset_creation", "images")
DEFAULT_NB_FRAMES = 100


def extract_frames_single_video(
    input_file_path: str, output_folder_path: str, nb_frames: int
) -> None:
    _, tail = os.path.split(input_file_path)
    file_name = tail.split(".")[0]
    used_frames = []
    for file in os.listdir(output_folder_path):
        if file.startswith(file_name) and file.endswith(".jpg"):
            used_frames.append(int(file.split("_")[-1][:-4]))  # -4 to remove .jpg

    vidcap = cv2.VideoCapture(input_file_path)
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    available_frames = set(range(video_length)) - set(used_frames)
    if len(available_frames) < nb_frames:
        nb_frames = len(available_frames)
        print(
            "Warning: The number of frames to extract exceeds the number of available frames, only the available ones have been extracted."
        )
    kept_frames = random.sample(available_frames, k=nb_frames)

    for count, frame in enumerate(tqdm.tqdm(range(max(kept_frames) + 1))):
        _, image = vidcap.read()
        if frame in kept_frames:
            cv2.imwrite(
                os.path.join(output_folder_path, f"{file_name}_{count}.jpg"), image
            )  # save frame as JPEG file


def extract_frames_all_videos(
    input_folder_path: str = DEFAULT_INPUT_FOLDER_PATH,
    output_folder_path: str = DEFAULT_OUTPUT_FOLDER_PATH,
    nb_frames: int = DEFAULT_NB_FRAMES,
) -> None:
    for file in os.listdir(input_folder_path):
        if file.endswith(".mp4"):
            extract_frames_single_video(
                os.path.join(input_folder_path, file), output_folder_path, nb_frames
            )


if __name__ == "__main__":
    extract_frames_all_videos()
