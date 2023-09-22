from ultralytics import YOLO
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
# Load a model
# model = YOLO('yolov8n-seg.pt')  # load an official model
model = YOLO('model/best.pt')  # load a custom model
# Predict with the model
results = model('forest.jpg')  # predict on an image
masks = results[0].masks


print(masks.data)


# img = cv2.imread('forest.jpg')

# contours = []

# for mask in masks:
#   contour = cv2.findContours(np.squeeze(np.array(mask)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#   contours.append(contour[0])

# cv2.drawContours(img, contours, -1, (0,255,0), 2)

# cv2.imwrite('converted_img.jpg', img)