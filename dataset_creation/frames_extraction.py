import os
import random

import tqdm
import cv2

DEFAULT_INPUT_FOLDER_PATH = os.path.join("dataset_creation", "videos")
DEFAULT_OUTPUT_FOLDER_PATH = os.path.join("dataset_creation", "images")
DEFAULT_NB_FRAMES = 100

def extract_frames_single_video(input_file_path: str, output_folder_path: str, nb_frames: int) -> None:
    _, tail = os.path.split(input_file_path)
    file_name = tail.split(".")[0]
    count = 0
    for file in os.listdir(output_folder_path):
        if file.startswith(file_name) and file.endswith(".jpg"):
            count += 1
    print(count)
    
    vidcap = cv2.VideoCapture(input_file_path)
    video_length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    kept_frames = set(random.sample(range(video_length), k=nb_frames))

    for frame in tqdm.tqdm(range(max(kept_frames) + 1)):
        _, image = vidcap.read()
        if frame in kept_frames: 
            cv2.imwrite(os.path.join(output_folder_path, f"{file_name}_{count}.jpg"), image) # save frame as JPEG file
            count += 1

def extract_frames_all_videos(
    input_folder_path: str = DEFAULT_INPUT_FOLDER_PATH,
    output_folder_path: str = DEFAULT_OUTPUT_FOLDER_PATH,
    nb_frames: int = DEFAULT_NB_FRAMES,
) -> None:
    for file in os.listdir(input_folder_path):
        if file.endswith(".mp4"):
            extract_frames_single_video(os.path.join(input_folder_path, file), output_folder_path, nb_frames)

if __name__=="__main__":
    extract_frames_all_videos()
