from typing import Tuple, List

import cv2
import os

from field_detector import FieldTracker
from ball_tracker import BallTracker
from player_tracker import PlayerTracker

from time import time
from imutils import rotate
import numpy as np
from copy import deepcopy
from datetime import datetime



class CamReader:
    def __init__(self, idx = 0):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 120)
        if not self.cap.isOpened():
            raise Exception(f"Can't open cam at index {idx}")
        
        d = datetime.now()
        self.name = f"{d.year}_{d.month}_{d.day}_{d.hour}h_{d.minute}m_{d.second}s"
        
    def get_name_and_path(self):
        return self.name, "."
    
    def frames(self, img_size:Tuple[int] = (640, 480)):
        t0=time()
        while 1:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield cv2.resize(frame, img_size, interpolation=cv2.INTER_AREA), time()-t0


class VideoReader:
    def __init__(self, input_file_path:str):
        self.dir_path, tail = os.path.split(input_file_path)
        self.file_name = tail.split(".")[0]

        self.cap = cv2.VideoCapture(input_file_path)
        self.vid_len = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

        self.frame_rate = None
    
    def get_name_and_path(self):
        return self.file_name, self.dir_path

    def get_frame_rate(self):
        if self.frame_rate is None:
            raise Exception("To get the video frame rate, you must call the frames() method first: the parameter half will determine its value.")
        return self.frame_rate

    def frames(self, nb_frames:int = None, size:Tuple[int]=(640,480), half:bool=False):
        """       
        A generator to extract frames from video and resize them to the desired size.
        If half is True, it will skip every other image.
        """

        self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS) / (1 + half)
        count = 0
        retval = True
        while retval and (nb_frames is None or count<nb_frames):
            retval, image = self.cap.read()
            yield cv2.resize(image, size, interpolation=cv2.INTER_AREA), count * (1 + half)/ self.frame_rate
            count+=1
            if half:
                self.cap.read()
        

def track_pipeline(from_cam = True, vis = False, check_field_every=50, batch_size=25):
    """
    A generator that returns a dictionary with:
    -A "time" key associated with the time at which the frame was captured
    -An "image" key with the image with rotation correction
    -A "ball" key -> None if the ball wasn't detected or the coordinates of the ball in Yolo format
    -"red_players" and "blue_players" keys with lists of coordinates in Yolo format
    -"corners" key -> list of the pixel coordinates of the fields corners starting from the top left one
    """
    t0=time()
    
    if from_cam:
        reader = CamReader()
    else:
        reader = VideoReader("../babyfoot_CV/dataset_creation/videos/perso_6.avi")

    fd = FieldTracker()
    tracker = BallTracker()
    player_tracker = PlayerTracker("../exp18/weights/best.pt", device="cuda:0")
    print(f"Loaded yolov5s in {time()-t0:.2f}s!")

    if vis:
        recording_name, dir = reader.get_name_and_path()
        dir_out = os.path.join(dir, recording_name)
        os.makedirs(dir_out, exist_ok=True)
    
    if from_cam:
        f = reader.frames()
    else:
        f = reader.frames(half=False, nb_frames=500)

    corners, angle = None, None
    rolling_buffer = [None]*batch_size
    sending_buffer = [None]*batch_size
    
    for i, [img, timestamp] in enumerate(f):

        #Field detection ~ 1ms
        if i%check_field_every==0:
            corners, _, angle= fd.get_field_corners(img)
            angle=int(angle)
        if angle != 0:
            img = rotate(img, angle)

        cropped_img, shift_x, shift_y = fd.mask_and_crop_field(img, corners, expand_by=25)

        rolling_buffer[i%batch_size] = (img, cropped_img, shift_x, shift_y, deepcopy(corners))
        sending_buffer[i%batch_size] = {"time":timestamp, "img":img}

        if (i+1)%batch_size == 0:
            players_batch = player_tracker.track_batch([img[1][..., ::-1] for img in rolling_buffer])      #BGR TO RGB

            for idx, [img, cropped_img, s_x, s_y, corn] in enumerate(rolling_buffer):
                #Ball tracking ~ 0.5ms
                x, y, w, h = tracker.track(cropped_img, shift_x=s_x, shift_y=s_y)
                if x < 0:   #ball not detected
                    sending_buffer[idx]["ball"] = None
                else:
                    sending_buffer[idx]["ball"] = (x/img.shape[0], y/img.shape[1], w/img.shape[0], h/img.shape[1])
                if vis:
                    img = cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), color=(0,0,255), thickness=2)

                sending_buffer[idx]["red_players"] = []
                sending_buffer[idx]["blue_players"] = []
                for xmin, ymin, xmax, ymax, conf, n_class in players_batch[idx]:
                    if vis:
                        img = cv2.rectangle(img, (int(xmin)+s_x, int(ymin)+s_y), (int(xmax)+s_x, int(ymax)+s_y), color=(0,0,255) if n_class==0 else (255,0,0), thickness=1)
                    if n_class==0:
                        sending_buffer[idx]["red_players"].append( ( ((xmin+xmax)/2 + s_x)/img.shape[0], ((ymin+ymax)/2 + s_y)/img.shape[1], (xmax-xmin)/img.shape[0], (ymax-ymin)/img.shape[1]) )
                    else:
                        sending_buffer[idx]["blue_players"].append( ( ((xmin+xmax)/2 + s_x)/img.shape[0], ((ymin+ymax)/2 + s_y)/img.shape[1], (xmax-xmin)/img.shape[0], (ymax-ymin)/img.shape[1]) )
                    
                if vis:
                    cv2.polylines(img, [np.array(corn)], True, color=(0,255,0), thickness=3)
                sending_buffer[idx]["corners"] = corn

                # Saving the modified frame ~ 6ms
                if vis:
                    cv2.imwrite(os.path.join(dir_out, f"{recording_name}_{i-batch_size+idx+1}.jpg"), img)

        if i>=batch_size:
            #print((i-batch_size+1), sending_buffer[(i+1)%batch_size]["time"],  len(sending_buffer[(i+1)%batch_size]["red_players"]))
            yield sending_buffer[(i+1)%batch_size]

        if vis and i >=batch_size:
            cv2.imshow('frame', rolling_buffer[(i+1)%batch_size][0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    track_data = track_pipeline(vis=True, from_cam=True)
    for i, data in enumerate(track_data):
        print(i, data["time"])



