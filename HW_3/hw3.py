import cv2
import numpy as np
from order_targets import order_targets

# Load in video
video_capture = cv2.VideoCapture("fiveCCC.avi")
can_read, frame = video_capture.read()
if not can_read:
    print("Error reading file check path and format")
    sys.exit()
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter("fiveCCC_output.avi", fourcc=fourcc, fps=30.0, frameSize=(641, 481))


# Finds Euclidian distance of points
def distance(point_a, point_b):
    return np.sqrt((point_b[0]-point_a[0])**2 + (point_b[1]-point_a[1])**2)

# Find CCC function pulled from past HW submission
def find_CCC(frame, frame_number):
    areas = []
    centroids_locations = []
    points = []
    # frame = cv2.imread("CCCtarget.jpg")

    # Morph Configs
    kernel_size_black = 2
    kernel_size_white = 2
    kernel_black = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_black, kernel_size_black))
    kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_white, kernel_size_white))

    # Threshold frame
    # Convert to gray scale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Otsu's Algorithm thresholding
    threshold, binary_image = cv2.threshold(gray_image, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   
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
    centrod_image = frame.copy()
    for white_stat, white_centroid in zip(white_stats, white_centroids):
        white_x, white_y = white_centroid
        # print(white_x, white_y)
        for black_stat, black_centroid in zip(black_stats, black_centroids):
            black_x, black_y = black_centroid
            point_distance = distance((white_x, white_y), (black_x, black_y))
            # print(point_distance)
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
                        if((white_stat[cv2.CC_STAT_AREA] < 350) and (black_stat[cv2.CC_STAT_AREA] < 400)):
                            # centrod_image = cv2.rectangle(img=centrod_image, pt1=(x0_black,y0_black), pt2=(x0_black+width_black, y0_black+height_black), color=(0,255,0), thickness=1)
                            # print("Area", black_stat[cv2.CC_STAT_AREA])
                            # Append center of centroid to list
                            centroids_locations.append(np.array([(x0_black + 0.5*width_black) ,(y0_black + 0.5*height_black)]))
                            areas.append(black_stat[cv2.CC_STAT_AREA])
                            points.append(((x0_black,y0_black), (x0_black+width_black, y0_black+height_black)))
    #Area filtering (needed due to issue with ccc false positives) 
    # Reference https://medium.com/datadriveninvestor/finding-outliers-in-dataset-using-python-efc3fce6ce32
    
    if len(centroids_locations) > 5:
        indexs = []
        print(len(areas))
        area_threshold = 3
        mean = np.mean(areas)
        std_area = np.std(areas)

        for i in range(len(areas)):
            z_score = (areas[i] - mean)/std_area
            if np.abs(z_score) > area_threshold:
                print("Removed...")
                indexs.append(i)

        indexs.sort(reverse=True)
        for index in indexs:
            _= centroids_locations.pop(index)
            _ = points.pop(index)
    print("P {} C {}".format(len(points), len(centroids_locations)))
    # for point in points:
    #     centrod_image = cv2.rectangle(img=centrod_image, pt1=point[0], pt2=point[1], color=(0,255,0), thickness=1)

    #Debugging Tools
    # cv2.imshow("Input", frame)
    # cv2.imshow("Threshold", binary_image)
    # cv2.imshow("Filtered", filtered_image)
    # cv2.imshow("White Labels", labels_white_display)
    # cv2.imshow("Black Labels", labels_black_display)
    cv2.putText(centrod_image, "Frame: {}".format(str(frame_number)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
 

    # print(centroids_locations)
    # print()
    ordered_centroids = order_targets(centroids_locations)
    print(ordered_centroids)

    for point in ordered_centroids:
        print("Point", point[0])
        centrod_image = cv2.rectangle(img=centrod_image, pt1=(int(point[0]) - 10, int(point[1]) -10), pt2=( int(point[0]) + 10, int(point[1]) + 10), color=(0,255,0), thickness=1)
    # print(ordered_centrods)
    image_points = np.asarray(ordered_centroids, dtype=np.float32)
    # print(image_points)
    # print(image_points.shape)
    # Make sure ordering worked
    # if (len(ordered_centroids) != 5):
    #     print("Error ordering Centroids...{}".format(len(ordered_centroids)))
    #     exit()
     # Print ordered labels
    for i in range(len(ordered_centroids)):
        point = (int(ordered_centroids[i][0]), int(ordered_centroids[i][1]))
        # print(point)
        cv2.putText(centrod_image, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)

    cv2.imshow("Final", centrod_image)
    cv2.waitKey(30)
def main():
    frame_number = 1

    while True:
        can_read, frame = video_capture.read()
        if not can_read:
            break
        find_CCC(frame, frame_number)
        frame_number +=1


if __name__ == "__main__":
    main()