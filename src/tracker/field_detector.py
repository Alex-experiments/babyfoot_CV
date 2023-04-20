from typing import List, Tuple

import numpy as np
import cv2
from imutils import rotate
from copy import deepcopy

#METHOD = "RED"
METHOD = "GREEN"

DEFAULT_PARAMETERS = {
    #"lower_filter": np.array([70, 100, 50]),
    #"upper_filter": np.array([80, 255, 255]),  
    "GREEN": {"lower_filter": np.array([85, 85, 70]),       #Pour données perso, mettre r=8, margin=15
            "upper_filter": np.array([105, 150, 255]),
            "ray": 8,
            "margin": 25
            },
    "RED":  {"lower_filter": np.array([0, 140, 200]),        #Pour rouge, mettre r=3, margin=25
            "upper_filter": np.array([20, 210, 255]),
            "ray": 3,
            "margin": 25
            },
    #"lower_filter": np.array([33, 0, 44]),      #Custom footage: 43.5< v <= 118.5  41.5<h <= 107.5
    #"upper_filter": np.array([114, 255, 137]),                  #118.5<v <=136.5  33<h<=114.5
    "threshold_value": 10,
}


class FieldTracker():
    intermediate_image_saving: bool = False
    lower_filter: np.ndarray = DEFAULT_PARAMETERS[METHOD]["lower_filter"]
    upper_filter: np.ndarray = DEFAULT_PARAMETERS[METHOD]["upper_filter"]
    threshold_value: int = DEFAULT_PARAMETERS["threshold_value"]

    def __init__(self, intermediate_image_saving=False):
        self.intermediate_image_saving = intermediate_image_saving
        r = DEFAULT_PARAMETERS[METHOD]["ray"]
        self.circular_kernel = np.fromfunction(lambda x, y: ((x-r)**2 + (y-r)**2 <= r**2)*1, (2*r+1, 2*r+1), dtype=int).astype(np.uint8)
    

    def color_filter(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply a color filter to an image"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_filter, self.upper_filter)
        filtered = cv2.bitwise_and(image, image, mask=mask)
        filtered = cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)

        if self.intermediate_image_saving:
            cv2.imwrite("mask.jpg", mask)
            cv2.imwrite("filtered.jpg", filtered)

        filtered = cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)

        return filtered, mask

    def erosion_dilation(self, image: np.ndarray) -> np.ndarray:
        """Apply erosion and dilation to the image to help the detection of the field by filtering noise"""
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel=self.circular_kernel)

    def threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply a threshold to keep the objects and remove noise"""
        threshold = cv2.threshold(image, self.threshold_value, 255, cv2.THRESH_BINARY)[1]

        if self.intermediate_image_saving:
            cv2.imwrite("threshold.jpg", threshold)

        return threshold

    def approximate_contours(self, image: np.ndarray) -> List[List[int]]:
        """Find the contours of the image and approximate it
        The output is a list of the position of the corners: a list containing 4 list of two integers [pos_x, pos_y]
        The corners are returned in a clockwise order starting from the top left
        """
        
        temp = np.where(image>0)[0]
        if temp.shape[0] == 0:  #aucun background n'a été détecté
            return [[0,0], [image.shape[1], 0], [image.shape[1], image.shape[0]], [0, image.shape[0]]]
        
        field_lb = temp.min()
        field_ub = temp.max()

        margin = DEFAULT_PARAMETERS[METHOD]["margin"]
        temp = np.where(image[field_lb:field_lb+margin]>0)[1]
        field_lower_corners = [[temp.min(), field_lb], [temp.max(), field_lb]]

        temp = np.where(image[field_ub-margin:field_ub]>0)[1]
        field_upper_corners = [[temp.min(), field_ub], [temp.max(), field_ub]]

        corners = field_lower_corners+field_upper_corners[::-1]

        if self.intermediate_image_saving:
            #saved_image = image.copy()
            saved_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cv2.polylines(saved_image, [np.array(corners)], True, color=(0,0,255), thickness=3)
            cv2.imwrite("contours_approximation.jpg", saved_image)

        return corners

    def detect(self, image: np.ndarray) -> Tuple[List[List[int]], np.ndarray, float]:
        """Main function of the class
        Takes an image in BGR format and returns the positions of the corners of the field, the rotated image and the angle of rotation needed to align vertically the field.
        Corners are returned in [x, y] format in clockwise order starting from top left.
        """

        filtered, _ = self.color_filter(image)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        gray = self.erosion_dilation(gray)
        threshold = self.threshold(gray)
        angle = self.get_orientation(threshold)
        #print(f"Angle: {angle}")
        #rotate par rapport au centre du field
        rotated_image = rotate(image, angle)
        threshold = rotate(threshold, angle)
        if self.intermediate_image_saving:
            cv2.imwrite("corrected_rotation.jpg", rotated_image)
            cv2.imwrite("corrected_threshold.jpg", threshold)
            

        corners = self.approximate_contours(threshold)
        
        if self.intermediate_image_saving:
            saved_image = rotated_image.copy()
            cv2.polylines(saved_image, [np.array(corners)], True, color=(0,0,255), thickness=3)
            cv2.imwrite("field_detection.jpg", saved_image)

        return corners, rotated_image, angle
    
    def expand_corners(self, corners: List[List[int]], expand_by: int, img_size:np.ndarray)->List[List[int]]:
        """Carefull, inplace expanding -> will modify the list corners"""
        #corners are in clockwise order starting with top left
        corners[0][0] -= expand_by
        corners[0][1] -= expand_by
        corners[1][0] += expand_by
        corners[1][1] -= expand_by

        corners[2][0] += expand_by
        corners[2][1] += expand_by
        corners[3][0] -= expand_by
        corners[3][1] += expand_by

        for i in range(len(corners)):       #bound corners values to be in the img, can deform the rect so not the best method
            corners[i][0] = min(img_size[1], max(corners[i][0], 0))
            corners[i][1] = min(img_size[0], max(corners[i][1], 0))
    
    def get_field_corners(self, image:np.ndarray)->Tuple[List[List[int]], np.ndarray]:
        corners, rotated_img, angle = self.detect(image)        
        return corners, rotated_img, angle

    def mask_and_crop_field(self, rotated_image: np.ndarray, corners:List[List[int]], expand_by:int = 0) -> Tuple[np.ndarray, int, int]:
        """Crop the image and returns the shifting"""

        corners = deepcopy(corners)

        if expand_by != 0:
            self.expand_corners(corners, expand_by, img_size=rotated_image.shape)
 
        min_x, max_x = min(corners[0][0], corners[3][0]), max(corners[1][0], corners[2][0])
        min_y, max_y = min(corners[0][1], corners[1][1]), max(corners[2][1], corners[3][1])

        mask = np.zeros(rotated_image.shape[:2], dtype=np.int8)
        cv2.fillPoly(mask, [np.array(corners)], (255,255,255))

        return cv2.bitwise_and(rotated_image, rotated_image, mask=mask)[min_y:max_y, min_x:max_x, :], min_x, min_y


    def get_orientation(self, image: np.ndarray) -> float:
        rotrect = cv2.minAreaRect(np.argwhere(image>0))

        if self.intermediate_image_saving:
            box = np.intp(cv2.boxPoints(rotrect))[:, ::-1]
            boxed_thresh = image.copy()
            cv2.drawContours(boxed_thresh, [box], 0, 125, 2)
            cv2.imwrite("thresh_boxed.jpg", boxed_thresh)

        if rotrect[1][0] <= rotrect[1][1]:   #if rectangle width < height
            return 90 - rotrect[-1]
        return -rotrect[-1]


if __name__ == "__main__":
    #initial_image = cv2.imread("image.jpg")
    #initial_image = cv2.imread("./dataset_creation/balle_jaune.jpg")
    #initial_image = cv2.imread("../dataset/train/ITSF_2020_cropped_1667.jpg")
    #initial_image = cv2.imread("d:/downloads/image.jpg")
    #initial_image = cv2.imread("./rotated_field.png")
    initial_image = cv2.imread("./dataset_creation/perso_7_202.jpg")
    #initial_image = cv2.imread("./dataset_creation/perso_3_1390.jpg")
    #initial_image = rotate(initial_image, 160) 
    fd = FieldTracker(intermediate_image_saving=True)
    corners, rotated_image, angle = fd.get_field_corners(initial_image)
    cropped, _, _ = fd.mask_and_crop_field(rotated_image, corners, expand_by=25)
    cv2.imwrite("cropped_image.jpg", cropped)
