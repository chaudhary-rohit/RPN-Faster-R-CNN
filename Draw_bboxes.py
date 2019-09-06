import cv2
import numpy as np


def Draw_bboxes(bboxes, image):
    
    image = image
    for box in bboxes[:]:
        figure = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,255,0), 1)
    
    cv2.imshow('Image with Bounding-Boxes', figure)    
    cv2.waitKey(0)   
    cv2.destroyAllWindows()
    
    
    
