import cv2
import numpy as np
from order_targets import order_targets

# Load in video
video_capture = cv2.VideoCapture("fiveCCC.avi")
can_read, frame = video_capture.read()
if not can_read:
    print("Error reading file check path and format")
    sys.exit()
# Save video
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
videoWriter = cv2.VideoWriter("hw3_output.avi", fourcc=fourcc, fps=30.0, frameSize=(int(video_capture.get(3)), int(video_capture.get(4))))

# Global defines for the points and input image
model_points = np.array([[0,0,1],[3.7,0,1], [7.4,0,1], [0,4.55,1], [7.4,4.55,1]], dtype=np.float32)
bgr_image = cv2.imread("naked_mole_rat.jpg")
input_image_height = bgr_image.shape[0]
input_image_width = bgr_image.shape[1]
# Coordinates of new image to use for mapping to CCC video
coordinates = [(0,0), (input_image_width/2, 0), (input_image_width, 0), (0, input_image_height), (input_image_width, input_image_height)]



# Intrinsics
f = 531.0
cx = 320.0
cy = 240.0
K = np.array(((f, 0.0, cx), (0.0, f, cy), (0.0, 0.0, 1.0)))
def pose(frame, image_points):

    # Pose calculations
    isPoseFound, rvec, tvec = cv2.solvePnP(objectPoints=model_points, imagePoints=image_points, cameraMatrix = K, distCoeffs = None)
    pImg, Jacobian = cv2.projectPoints(objectPoints=model_points, rvec = rvec, tvec = tvec, cameraMatrix = K, distCoeffs = None)

    # Axis plotting (referenced slides)
    W = np.amax(model_points, axis=0)
    L = np.linalg.norm(W)
    d = L/5

    pImg = pImg.reshape(-1,2)
    cv2.line(frame, tuple(np.int32(pImg[0])),
            tuple(np.int32(pImg[1])), (0, 0, 255), 2)  # x
    cv2.line(frame, tuple(np.int32(pImg[0])),
            tuple(np.int32(pImg[2])), (0, 255, 0), 2)  # y
    cv2.line(frame, tuple(np.int32(pImg[0])),
            tuple(np.int32(pImg[3])), (255, 0, 0), 2)  # z


    # Translation and Rotation values (vectors)
    cv2.putText(frame, "rvec :{}".format(rvec), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, "tvec :{}".format(tvec), (50,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    #Show frame wait for input
    # cv2.imshow("POSE", frame)
def createNamedWindow(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h = image.shape[0]
    w = image.shape[1]

    WIN_MAX_SIZE = 1000
    if max(w,h) >WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE/max(w,h)
    else:
        scale = 1
    cv2.resizeWindow(winname=window_name, width=int(w * scale), height = int(h * scale))

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        param.append((x, y))

def warpImage(frame, image_points):
    # Use the coordinates of new image for projecting
    coor_points = np.asarray(coordinates, dtype=np.float32)

    # Find the projections
    H_0_1, _ = cv2.findHomography(coor_points, image_points, cv2.RANSAC)
    bgr_output = cv2.warpPerspective(bgr_image, H_0_1, (frame.shape[1], frame.shape[0]))

    # Zero out where the new image will be placed on the original
    fill_points = np.asarray([image_points[0], image_points[2], image_points[4], image_points[3]], dtype= np.float32)
    print(fill_points)
    cv2.fillConvexPoly(frame, fill_points.astype(int), 0)

    # Make sure both images are same size
    bgr_output = cv2.resize(bgr_output, (frame.shape[1], frame.shape[0]))
    overlay = cv2.bitwise_or(frame, bgr_output)
    return overlay

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
    # Area filtering (needed due to issue with ccc false positives) 
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
    # print("P {} C {}".format(len(points), len(centroids_locations)))
    # Debugging Tools
    # cv2.imshow("Input", frame)
    # cv2.imshow("Threshold", binary_image)
    # cv2.imshow("Filtered", filtered_image)
    # cv2.imshow("White Labels", labels_white_display)
    # cv2.imshow("Black Labels", labels_black_display)
    cv2.putText(centrod_image, "Frame: {}".format(str(frame_number)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # order targets
    ordered_centroids = order_targets(centroids_locations)
    # print(ordered_centroids)
    # Center points
    for point in ordered_centroids:
        # print("Point", point[0])
        centrod_image = cv2.rectangle(img=centrod_image, pt1=(int(point[0]) - 10, int(point[1]) -10), pt2=( int(point[0]) + 10, int(point[1]) + 10), color=(0,255,0), thickness=1)
    # print(ordered_centrods)
    image_points = np.asarray(ordered_centroids, dtype=np.float32)
    
    for i in range(len(ordered_centroids)):
        point = (int(ordered_centroids[i][0]), int(ordered_centroids[i][1]))
        # print(point)
        cv2.putText(centrod_image, str(i), point, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 1, cv2.LINE_AA)
    # If there are enough found CCC in frame fun projection
    if len(image_points) == 5:
        pose(centrod_image, image_points)
        centrod_image = warpImage(centrod_image, image_points)
    
    cv2.imshow("Final", centrod_image)
    videoWriter.write(centrod_image)

    cv2.waitKey(30)



def main():

    frame_number = 1

    while True:
        can_read, frame = video_capture.read()
        if not can_read:
            break
        find_CCC(frame, frame_number)
        frame_number +=1
    videoWriter.release()


if __name__ == "__main__":
    main()