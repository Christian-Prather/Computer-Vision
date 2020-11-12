# Lab 10 Computer Vision 
# Christian Prather
# Referenced hough tutorial in slides

import cv2
import numpy as np
from vanishing import find_vanishing_point_directions

# Set Globals that will be used for the processing
# Note: While globals some do get recomputed dynamically
# Max width of image
MAX_DIMS = 1000
# Blur to soften image
SIGMA_BLUR = 1.0
# Edge globals
MIN_FRACT_EDGES = 0.05
MAX_FRACT_EDGES = 0.08
# Line globals
MIN_HOUGH_VOTES_FRACTION = 0.08
MIN_LINE_LENGTH_FRACTION = 0.03
# Image name to load
image_name = "corridor1"

def main():
    # Read in image (change for corridor3.png)
    color_image = cv2.imread("{}.jpg".format(image_name))
    # Resize if to big
    if color_image.shape[1] > MAX_DIMS:
        scale = MAX_DIMS / color_image.shape[1]
        color_image = cv2.resize(color_image, dsize=None, fx=scale, fy=scale)
    print("Dims:", color_image.shape)
    # Smooth with gausian blur
    gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(src= gray_image, ksize=(0,0), sigmaX= SIGMA_BLUR, sigmaY= SIGMA_BLUR)
    
    # Starting point
    thresh_canny = 1.0
    edge_image = cv2.Canny(image= gray_image, apertureSize=3, threshold1 = thresh_canny, threshold2 = thresh_canny * 3, L2gradient=True)

    # Edge detection
    while np.sum(edge_image)/255 < MIN_FRACT_EDGES *(edge_image.shape[1] * edge_image.shape[0]):
        print("Decreasing the threshold...")
        thresh_canny *= 0.9
        edge_image = cv2.Canny(image= gray_image, apertureSize=3, threshold1 = thresh_canny, threshold2 = thresh_canny * 3, L2gradient=True)
    
    while np.sum(edge_image)/255 > MIN_FRACT_EDGES *(edge_image.shape[1] * edge_image.shape[0]):
        print("Increasing the threshold...")
        thresh_canny *= 1.1
        edge_image = cv2.Canny(image= gray_image, apertureSize=3, threshold1 = thresh_canny, threshold2 = thresh_canny * 3, L2gradient=True)

    # Houg Line
    houghThreshold = int(edge_image.shape[1] * MIN_HOUGH_VOTES_FRACTION)
    hough_lines = cv2.HoughLinesP(image = edge_image, rho = 1, theta = np.pi/ 180, threshold = houghThreshold, lines=None,
     minLineLength= int(edge_image.shape[1] * MIN_LINE_LENGTH_FRACTION),
     maxLineGap=10)

    print("Found {} line segments".format(len(hough_lines)))
    # Uncomment to see line array
    # print(hough_lines)

    # Line displaying
    line_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    for i in range(0, len(hough_lines)):
        line = hough_lines[i][0]
        cv2.line(line_image, (line[0], line[1]),(line[2], line[3]), (0,0,255), thickness=2, lineType=cv2.LINE_AA)

    # Find vanishing points
    vanishing_directions = find_vanishing_point_directions(hough_lines, color_image)

    # Show all images
    cv2.imshow("Color image", color_image)
    cv2.imshow("Gray Image", gray_image)
    cv2.imshow("Edge Image", edge_image)
    cv2.imshow("Lines Image", line_image)

    # Save key images
    cv2.imwrite("edge_image_{}.jpg".format(image_name), edge_image)
    cv2.imwrite("hough_lines_{}.jpg".format(image_name), line_image)
    cv2.waitKey(0)
if __name__ == "__main__":
    main()