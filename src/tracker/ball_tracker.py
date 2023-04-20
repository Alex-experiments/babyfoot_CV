from typing import Tuple

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

DEFAULT_PARAMETERS = {
    #"lower_filter": np.array([20, 0, 118]),      #Custom footage: 19.5<h<=43.5 118.5<v
    #"upper_filter": np.array([44, 255, 255]),  
    "lower_filter": np.array([25, 20, 113]),        #YT dataset : 24<=h<=54 97<=s
    "upper_filter": np.array([54, 255, 255]),       #27<=h<=54 90<=s 113<=v
}

class BallTracker():
    DEBUG_MODE = False

    def __init__(self):
        self.get_ball_hsv()

    def track(self, img:np.ndarray, truth:np.ndarray=None, shift_x=0, shift_y=0)-> Tuple[int]:
        """Main function of the class, returns x,y,w,h of the ball bounding box (all values are equal to -1 if the ball couldn't be detected).
        img: the frame in BGR on which we want to predict the position of the ball,
        truth: the true label (optional)
        shift_x, shift_y: the shifting due to a potential cropping on the image"""

        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #We work with HSV to have a more resilient prediction

        x, y, w, h = self.chroma_key(img)
        if not x<0: #if the ball was detected we apply the shifting, else we just keep all value at -1
            x += shift_x
            y += shift_y

        if self.DEBUG_MODE:
            #print("pred:", x, y, w, h)
            if truth is not None:
                print(f"Pred: {x}, {y}, {w}, {h}, Truth: {truth}")
            else:
                print(f"Pred: {x}, {y}, {w}, {h}")

            self.plot(x, y, w, h, img, truth)
        return x, y, w, h
        
    def plot(self, x, y, w, h, img, truth = None):
        """A helper function to plot the detected bounding box and the truth if it is given"""
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_HSV2RGB))
        plt.gca().add_patch(Rectangle((x-w/2, y-h/2), w, h, linewidth=2, edgecolor='r', facecolor='none'))
        if truth is not None:
            plt.gca().add_patch(Rectangle((truth[0]-truth[2]/2, truth[1]-truth[3]/2), truth[2], truth[3], linewidth=2, edgecolor='g', facecolor='none'))
        plt.show()

    def get_ball_hsv(self):
        """Plan would be to ask the player to hold the ball in the center of the table,
        then grab the color of the ball.
        But first step: pre-defined color 
        """
        self.ball_color = np.array([30, 170, 255])
        #self.ball_color = np.array([21, 122, 224])
        self.inter =  np.array([7, 100, 100])
        self.lower_bound = self.ball_color - self.inter
        self.upper_bound = self.ball_color + self.inter
        self.lower_bound = DEFAULT_PARAMETERS["lower_filter"]
        self.upper_bound = DEFAULT_PARAMETERS["upper_filter"]

    def chroma_key(self, img:np.ndarray) -> Tuple[int]:  #potentiellement virer img des args, sinon envoie une copie
        ball_mask = cv2.inRange(img, self.lower_bound, self.upper_bound)

        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))

        contours, _ = cv2.findContours(ball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            if self.DEBUG_MODE:
                print("Unable to detect any ball")
            return -1,-1,-1,-1
        
        
        if self.DEBUG_MODE:
            plt.subplot(1, 3, 1)
            plt.imshow(img)
            plt.subplot(1, 3, 2)
            plt.imshow(ball_mask)
            plt.subplot(1, 3, 3)
            temp=np.zeros_like(img)
            cv2.drawContours(temp, contours, -1, (255, 0, 0), 3)
            plt.imshow(temp)
            plt.show()        
        

        #if multiple contours were detected, we return the one that is the most round
        max_score = 0
        best_pred = 0
        if len(contours) > 1: 
            for i, cont in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cont)
                s = self.roundness_score(ball_mask, y, x, h, w)
                if s > max_score:
                    best_pred = i
                    max_score = s
            if self.DEBUG_MODE:
                print("Best roundness score found: ", max_score)
        
        x, y, w, h = cv2.boundingRect(contours[best_pred])  #returns pos of upper left corner and width and height
        return x+w//2, y+h//2, w, h                         #converts to pos of center and width and height

    def roundness_score(self, mask: np.array, x: float, y: float, w: float, h: float) -> float :
        """The idea is we will calculate the area of the shape (the number of pixel that are equal to 255),
        then calculate the radius of a perfect circle of this area. The score will be the ratio of the intersection of both areas by the area of the shape."""
        points_x, points_y = (mask[x:x+w, y:y+h] == 255).nonzero()

        area = points_x.shape[0]
        if area <= 10:       #if we find less than 10 pixels, this shouldn't be considered as a potential ball
            return 0
        radius_sq = area/np.pi

        center_x, center_y = points_x.mean(), points_y.mean()
        #print(f"theoric center: ({center_x+x}, {center_y+y}), area: {area}, radius: {np.sqrt(radius_sq)}")
        area_inside_perfect_circle = 0
        for x, y in zip(points_x, points_y):
            if (x-center_x)**2 + (y-center_y)**2 <= radius_sq:
                area_inside_perfect_circle += 1
        return float(area_inside_perfect_circle)/area


def test_roundness_score():
    """This will test the roundness score on a circle shape and on a rectangle shape"""
    def get_shape(circle:bool)-> tuple[np.ndarray, list[int, int, int, int]]:
        mask = np.zeros((500, 400))
        center = (200, 300)
        r1, r2 = 50, 30

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if circle and (x-center[0])**2 + (y-center[1])**2 < r1**2 or not circle and abs(x-center[0]) < r1 and abs(y-center[1]) < r2:
                    mask[x, y] = 255
        
        if circle:
            r2 = r1
        return mask, [center[0]-r1, center[1]-r2, 2*r1, 2*r2]

    bt = BallTracker()

    circle_mask, circle_cont = get_shape(circle=True)
    rect_mask, rect_cont = get_shape(circle=False)
    circle_score = bt.roundness_score(circle_mask, *circle_cont)
    rect_score = bt.roundness_score(rect_mask, *rect_cont)
    print(f"Circle score: {circle_score}, rectangle score: {rect_score}")


def get_metrics(path="../dataset_ball_tracking/cropped_data"):
    import os

    dist = []
    n_correct = 0
    n_undetected = 0
    n_missed_pred = 0
    n_no_truth_no_pred = 0

    bt = BallTracker()

    for img_name in os.listdir(path):
        if not img_name.endswith(".jpg"):
            continue
        frame = cv2.imread(os.path.join(path, img_name))
        truth = None
        with open(os.path.join(path, img_name.replace(".jpg", ".txt")), "r") as f:
            for line in f.readlines():
                if line.startswith("0"):
                    truth = [float(elem) for elem in line.strip().split()[1:]]
                    break

        if truth is not None:
            truth[0] = int(truth[0] * frame.shape[1])
            truth[2] = int(truth[2] * frame.shape[1])
            truth[1] = int(truth[1] * frame.shape[0])
            truth[3] = int(truth[3] * frame.shape[0])

        #bt.DEBUG_MODE = img_name=="cropped_perso_7_3937.jpg"
        if bt.DEBUG_MODE:
            print(img_name)

        pred = bt.track(frame, truth=truth)
        
        if pred[0] < 0:
            if truth is not None:
                n_undetected += 1
            else:
                n_no_truth_no_pred += 1
        elif pred[0] >= 0:
            if truth is None:
                n_missed_pred += 1
                print(img_name)
            else:
                dist.append( np.sqrt((pred[0]-truth[0])*(pred[0]-truth[0]) + (pred[1]-truth[1])*(pred[1]-truth[1])) )
                if dist[-1] < 10 :
                    n_correct += 1
                else:
                    print(dist[-1], img_name)
    print(f"\nOn {len(dist)+n_undetected+n_missed_pred+n_no_truth_no_pred} annotated examples, {len(dist)} where detected ({n_correct} with dist<10), predicted mean sq distance is {np.mean(dist):.3f}")
    print(f"{n_undetected} images where ball couldn't be detected, {n_missed_pred} where a ball was predicted while not on the field, {n_no_truth_no_pred} images without ball in truth and in prediction")

if __name__=="__main__":
    test_roundness_score()
    b_tracker = BallTracker()
    #b_tracker.DEBUG_MODE = True
    #frame = cv2.imread("d:/downloads/image.jpg")
    #frame = cv2.imread("./dataset_creation/balle_jaune.jpg")
    frame = cv2.imread("./cropped_image.jpg")
    #frame = cv2.imread("./dataset_creation/perso_3_1390.jpg")
    #frame = cv2.imread("./dataset_creation/rotated_field_cropped.jpg")
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv_frame)
    plt.show()
    b_tracker.track(frame)

    get_metrics()
    
        

        



    
