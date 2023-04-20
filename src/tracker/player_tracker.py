from typing import Tuple, List

import torch
import numpy as np

class PlayerTracker:

    def __init__(self, model_path, device="cuda:0", use_local_repo=True, local_repo_path = '../yolov5'):
        if use_local_repo:
            self.model = torch.hub.load(local_repo_path, 'custom', model_path, device=device, _verbose=False, verbose=False, source="local")
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', model_path, device=device, _verbose=False, verbose=False)
        
        self.model.max_det = 22     # 11 players for each side

    def track_batch(self, frames:List[np.ndarray], size:int=240)->List[List[int]]:
        """
        Takes a list of frames in RGB as input, applies inference and returns the predictions of the batched images in a list.
        Each sublist returned is a list of the bounding boxes in Yolo format
        """
        return [pred.detach().cpu().numpy() for pred in self.model(frames, size=size).xyxy] 