from typing import Tuple, List

import cv2
import os
import field_detector
import ball_tracker
import torch
from time import time
from imutils import rotate
import numpy as np


def extract_continuous_frames(input_file_path:str, nb_frames:int = None, size:Tuple[int]=(640,480))-> Tuple[str, str, float]:
    """
    Extract frames from video, and resizes them to the desired size.
    Frames are saved in the same folder as the video.
    Returns the output dir, the video name and the frame rate.
    """
    t0 = time()
    print("Exctracting frames:")
    dir_path, tail = os.path.split(input_file_path)
    file_name = tail.split(".")[0]
    vidcap = cv2.VideoCapture(input_file_path)
    vid_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_rate = vidcap.get(cv2.CAP_PROP_FPS)

    dir_out = os.path.join(dir_path, file_name)
    os.makedirs(dir_out, exist_ok=True)

    count = 0
    retval, image = vidcap.read()
    while retval and (nb_frames is None or count<nb_frames):
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(dir_out, f"{file_name}_{count}.jpg"), image) # save frame as JPEG file
        retval, image = vidcap.read()
        count+=1
        if count % 10 == 0:
            if nb_frames is not None:
                print(f"{(100*count)//nb_frames}%", end="\r")
            else:
                print(f"{(100*count)//vid_len}%", end="\r")
    
    print(f"Frames extracted in {time()-t0:.2f}s!")
    return dir_out, file_name, frame_rate

def create_vid(dir_frames:str, video_name:str, frame_rate:float):
    """Creates video from images in dir_frames"""
    t0=time()
    print("Creating video:")
    images = [im for im in os.listdir(dir_frames) if im.endswith(".jpg")]
    size = cv2.imread(os.path.join(dir_frames, images[0])).shape[:2]
    n_images = len(images)
    video = cv2.VideoWriter(os.path.join(dir_frames, f'{video_name}_tracked.avi'), cv2.VideoWriter_fourcc(*'DIVX') , frame_rate, size[::-1])

    for count in range(n_images):
        if count%10 == 0:
            print(f"{(100*count)//n_images}%", end="\r")
        video.write(cv2.imread(os.path.join(dir_frames, f"{video_name}_{count}.jpg")))
    cv2.destroyAllWindows() 
    video.release()

    for im in images:
        os.remove(os.path.join(dir_frames, im))
    
    print(f"Video created in {time()-t0:.2f}s!")
    

def track_on_frames(dir_frames:str, batch_size:int = 50, check_field_every:int = 20):
    t0=time()
    print("Tracking ball and players on frames:")
    fd = field_detector.ChromaticDetection()
    tracker = ball_tracker.Ball_tracker()
    player_tracker = PlayerTracker("../exp17/weights/best.pt", device="cuda:0")
    t1=time()
    images = [im for im in os.listdir(dir_frames) if im.endswith(".jpg")]
    images = sorted(images, key=lambda name: int(name[:-4].split('_')[-1])) #important de les lire dans l'ordre comme on regarde l'orientation du terrain toutes les X frames
    
    corners, angle = None, None

    buffer={}

    # ~ 0.025s -> 40fps
    for i, img_name in enumerate(images):
        if i%10 == 0:
            print(f"{(100*i)//len(images)}%", end="\r")

        # Opening the image and deleting it ~ 8ms
        img = cv2.imread(os.path.join(dir_frames, img_name))
        os.remove(os.path.join(dir_frames, img_name))        

        #Field detection ~ 1ms
        if i%check_field_every==0:
            corners, useless_rotated_img, angle= fd.get_field_corners(img, expand_by=25)
            angle=int(angle)       #limiter les variations à 5° pres ?
            #print(angle, "    ")
        if angle != 0:
            img = rotate(img, angle)
        cropped_img, shift_x, shift_y = fd.mask_and_crop_field(img, corners)

        #Ball tracking ~ 0.5ms
        x, y, w, h = tracker.track(cropped_img, shift_x=shift_x, shift_y=shift_y)
        img = cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color=(0,0,255), thickness=2)
        
        # Batched player tracking ~ 5ms 
        if len(buffer) <= batch_size:
            buffer[img_name] = (img, cropped_img)
        if len(buffer)==batch_size:
            players_batch = player_tracker.track_batch([img[1][..., ::-1] for img in buffer.values()])      #BGR TO RGB

            for idx, [img_name, [img, _]] in enumerate(buffer.items()):
                # Plotting players and field bounding boxes ~ 5ms
                for xmin, ymin, xmax, ymax, conf, n_class in players_batch[idx]:
                    img = cv2.rectangle(img, (int(xmin)+shift_x, int(ymin)+shift_y), (int(xmax)+shift_x, int(ymax)+shift_y), color=(0,0,255) if n_class==0 else (255,0,0), thickness=1)
                
                cv2.polylines(img, [np.array(corners)], True, color=(0,255,0), thickness=3)

                # Saving the modified frame ~ 6ms
                cv2.imwrite(os.path.join(dir_frames, img_name), img)
            
            buffer={}        
        
    print(f"Tracking finished in {time()-t0:.2f}s ({t1-t0:.2f}s to load yolov5s)!")

class PlayerTracker:

    def __init__(self, model_path, device="cuda:0", use_local_repo=False):
        self.device = device
        if use_local_repo:
            self.model = torch.hub.load('../yolov5', 'custom', model_path, device=self.device, _verbose=False, verbose=False, source="local")
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, device=self.device, _verbose=False, verbose=False)
        self.model.max_det = 22

    def track_batch(self, frames:List[np.ndarray])->List[List[int]]:
        """
        Takes a list of frames in RGB as input, applies inference and returns the predictions of the batched images in a list.
        Each sublist is a list of the bounding boxes in Yolo format
        """
        return [pred.detach() for pred in self.model(frames, size=240).xyxy] 

        
if __name__ == "__main__":
    dir_frames, video_name, frame_rate = extract_continuous_frames("./dataset_creation/videos/ITSF_2020_cropped.mp4", nb_frames=500)
    track_on_frames(dir_frames)
    create_vid(dir_frames, video_name, frame_rate)
