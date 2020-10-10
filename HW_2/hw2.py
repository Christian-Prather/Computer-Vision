# Question 2
import cv2
import sys

import numpy as np


# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        param.append((x, y))
def load_image(path):
    image = cv2.imread(path)
    return image

def question_2():
    x_coor = 450
    y_coor = 211
    coordinates = []
 
    # Read in first frame for template selection
    image = load_image("textsample.tif")
    print(image.shape)
    q2_window_name = "Question_2"
    cv2.imshow(q2_window_name, image)

    # uncomment if setting new tmeplate
    #cv2.setMouseCallback(q2_window_name, get_xy, coordinates)
    
    # Once selected user needs to hit enter to progress
    # cv2.waitKey(0)

    # Uncomment if wanting to change template
    # Get Template as img
    # x_coor, y_coor = coordinates[0]
    print("P1 {} P2 {}".format((x_coor-10, y_coor-10), (x_coor+10, y_coor+10)))
    cropped = image[y_coor-10:y_coor+10, x_coor-10:x_coor+10]
    cropped_x, cropped_y, _ = cropped.shape
    print("T_x {} T_y {}".format(cropped_x, cropped_y))
    cv2.imwrite("template.png", cropped)

    # Show rectangle around template on first frame
    cv2.rectangle(image, (x_coor-10, y_coor-10), (x_coor+10, y_coor+10),(255,0,0), thickness=1, lineType=8)
    cv2.imshow(q2_window_name, image)

    # Must hit enter again to show rectange on video
    cv2.waitKey(0)

    scores = cv2.matchTemplate(image, cropped, cv2.TM_CCOEFF_NORMED)
    print(scores)

    # Local thresholding code from OpenCV matchTemplate documentation
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_template_matching/py_template_matching.html
    # Threshold experimantally derived 
    threshold = 0.60
    match_points = np.where(scores >= threshold)
    y_coordinates = match_points[0]
    x_coordinates = match_points[1]
    points = zip(x_coordinates, y_coordinates)

    number_of_matches = 0
    for x_coor, y_coor in points:
        number_of_matches += 1
        #print("Point", x_coor, y_coor)

        # Add offset to center the boundry rectangel
        x_coor += int(cropped_x/2)
        y_coor += int(cropped_y/2)
        cv2.rectangle(image, (x_coor-10, y_coor-10), (x_coor+10, y_coor+10),(0,255,0), thickness=1, lineType=8)
    cv2.imshow(q2_window_name, image)
    cv2.imwrite("output.png", image)
    cv2.waitKey(0)
    print("A's Found:", number_of_matches)

def distance(point_a, point_b):
    return np.sqrt((point_b[0]-point_a[0])**2 + (point_b[1]-point_a[1])**2)
def question_4():
    kernel_size_black = 3
    kernel_size_white = 3
    kernel_black = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_black, kernel_size_black))
    kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_white, kernel_size_white))

    frame_number = 1
    video_capture = cv2.VideoCapture("fiveCCC.avi")
    can_read, frame = video_capture.read()
    if not can_read:
        print("Error reading file check path and format")
        sys.exit()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("fiveCCC_output.avi", fourcc=fourcc, fps=30.0, frameSize=(641, 481))
   
    while True:
        can_read, frame = video_capture.read()
        if not can_read:
            break
        
        # Threshold frame
        # Convert to gray scale
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Otsu's Algorithm thresholding
        threshold, binary_image = cv2.threshold(gray_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # binary_image = cv2.adaptiveThreshold(src=gray_image, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 51, C= -10)
        # threshold = 300
        # Filtered image
        filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_white)
        filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel_white)

        # Connected Components
        num_while_labels, labels_white_image, white_stats, white_centroids = cv2.connectedComponentsWithStats(filtered_image)
        # num_black_labels, labels_black_image, black_stats, black_centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(filtered_image))
        labels_white_display = cv2.normalize(src = labels_white_image, dst= None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Otsu's Algorithm thresholding
        threshold, binary_image = cv2.threshold(gray_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # binary_image = cv2.adaptiveThreshold(src=gray_image, maxValue = 255, adaptiveMethod = cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType = cv2.THRESH_BINARY, blockSize = 51, C= -10)
        # threshold = 300
        # Filtered image
        filtered_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_black)
        filtered_image = cv2.morphologyEx(filtered_image, cv2.MORPH_CLOSE, kernel_black)
        num_black_labels, labels_black_image, black_stats, black_centroids = cv2.connectedComponentsWithStats(filtered_image)

        labels_black_display = cv2.normalize(src = labels_black_image, dst= None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Find CCC
        centrod_image = frame
        for white_stat, white_centroid in zip(white_stats, white_centroids):
            white_x, white_y = white_centroid
            # print(white_x, white_y)
            for black_stat, black_centroid in zip(black_stats, black_centroids):
                black_x, black_y = black_centroid
                point_distance = distance((white_x, white_y), (black_x, black_y))
                print(point_distance)
                if(point_distance < threshold):
                    if (white_stat[cv2.CC_STAT_AREA] < black_stat[cv2.CC_STAT_AREA]):
                        # Possible point
                        x0 = white_stat[cv2.CC_STAT_LEFT]
                        y0 = white_stat[cv2.CC_STAT_TOP]
                        width = white_stat[cv2.CC_STAT_WIDTH]
                        height = white_stat[cv2.CC_STAT_HEIGHT]
                        
                        x0_black = black_stat[cv2.CC_STAT_LEFT]
                        y0_black = black_stat[cv2.CC_STAT_TOP]
                        width_black = black_stat[cv2.CC_STAT_WIDTH]
                        height_black = black_stat[cv2.CC_STAT_HEIGHT]
                        if (x0_black < x0) and (y0_black < y0) and ((x0_black + width_black) > (x0 + width)) and ((y0_black + height_black) > (y0 + height)):
                            centrod_image = cv2.rectangle(img=centrod_image, pt1=(x0_black,y0_black), pt2=(x0_black+width_black, y0_black+height_black), color=(0,255,0), thickness=1)


        cv2.putText(centrod_image, "Frame: {}".format(str(frame_number)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Video", frame)
        cv2.imshow("Threshold", binary_image)
        cv2.imshow("Filtered", filtered_image)
        cv2.imshow("White Labels", labels_white_display)
        cv2.imshow("Black Labels", labels_black_display)
        cv2.imshow("Final", centrod_image)

        frame_number+=1

        # Save to video
        videoWriter.write(centrod_image)

        cv2.waitKey(30)
    videoWriter.release()

def main():
    # question_2()
    question_4()

if __name__ == "__main__":
    main()
