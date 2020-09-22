# OpenCV Lab4 by: Christian Prather

# Threshold values used
# Hue 158
# Hue 255
# Saturation 89
# Saturation 255
# Value 52
# Value 255

# Morph Kernel size (4x4)

import cv2
import numpy as np

# Default to found values
min_thresholds = [158,89,52]
max_thresholds = [255,255,255]
windowNames = ["Hue", "Saturation", "Value"]
kernel = np.ones((4,4), np.uint8)

# Callback
def calculate(x):
    threshold_image = np.full((input_image.shape[0], input_image.shape[1]), 255, dtype=np.uint8)
    for i in range(3):
        low_value = cv2.getTrackbarPos("Low", windowNames[i])
        high_value = cv2.getTrackbarPos("High", windowNames[i])
        print (windowNames[i], low_value)
        print(windowNames[i], high_value)

        _, low_img = cv2.threshold(splits[i], low_value, 255, cv2.THRESH_BINARY)
        _, high_img = cv2.threshold(splits[i], high_value, 255, cv2.THRESH_BINARY_INV)

        threshold_combo = cv2.bitwise_and(low_img, high_img)
        cv2.imshow(windowNames[i], threshold_combo)
        
        threshold_image = cv2.bitwise_and(threshold_image, threshold_combo)
    cv2.imshow("Final", threshold_image)
    morph = cv2.morphologyEx(threshold_image, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    cv2.imshow("Morph", morph)
    # Manually relate saved file to input
    cv2.imwrite("stop3_morph.jpg", morph)

# Manually enter input image (should loop eventually)
input_image = cv2.imread("stop3.jpg")

# Convert to HSV
hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
splits = cv2.split(hsv_image)
for name in windowNames:
    cv2.namedWindow(name)
# cv2.imshow("Segmented",input_image)

# Add tracbar
for i in range(3):
    cv2.createTrackbar("Low", windowNames[i], min_thresholds[i], 255, calculate)
    cv2.createTrackbar("High", windowNames[i], max_thresholds[i], 255, calculate)


cv2.waitKey(0)

