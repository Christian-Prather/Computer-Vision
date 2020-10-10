import cv2
import numpy as np
from order_targets import order_targets

def distance(point_a, point_b):
    return np.sqrt((point_b[0]-point_a[0])**2 + (point_b[1]-point_a[1])**2)

def find_CCC():

    centroids_locations = []
    frame = cv2.imread("CCCtarget.jpg")

    kernel_size_black = 1
    kernel_size_white = 1
    kernel_black = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_black, kernel_size_black))
    kernel_white = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_white, kernel_size_white))

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
                        if(black_stat[cv2.CC_STAT_AREA] < 400):
                            centrod_image = cv2.rectangle(img=centrod_image, pt1=(x0_black,y0_black), pt2=(x0_black+width_black, y0_black+height_black), color=(0,255,0), thickness=1)
                            print("Area", black_stat[cv2.CC_STAT_AREA])
                            centroids_locations.append([x0_black,y0_black])

    # cv2.imshow("Input", frame)
    # cv2.imshow("Threshold", binary_image)
    # cv2.imshow("Filtered", filtered_image)
    # cv2.imshow("White Labels", labels_white_display)
    # cv2.imshow("Black Labels", labels_black_display)
    # cv2.imshow("Final", centrod_image)
    print(centroids_locations)
    # ordered_centroids = order_targets(centroids_locations)
    # if (len(ordered_centroids) < 5):
    #     print("Error ordering Centroids...")
    #     exit()
    ordered_centroids = np.array([
        [343, 179], 
        [360,162], 
        [379,145], 
        [364,200], 
        [401,165]], dtype=np.float32)
    model_points = np.array([[0,0,1],[3.7,0,1], [7.4,0,1], [0,4.55,1], [7.4,4.55,1]], dtype=np.float32)
    # N = model_points.shape[1]
    print(model_points.shape)
    print(ordered_centroids.shape)
    f = 531.0
    cx = 320.0
    cy = 240.0
    K = np.array(((f, 0.0, cx), (0.0, f, cy), (0.0, 0.0, 1.0)))

    isPoseFound, rvec, tvec = cv2.solvePnP(objectPoints=model_points, imagePoints=ordered_centroids, cameraMatrix = K, distCoeffs = None)
    pImg, Jacobian = cv2.projectPoints(objectPoints=model_points, rvec = rvec, tvec = tvec, cameraMatrix = K, distCoeffs = None)

    W = np.amax(model_points, axis=0)
    L = np.linalg.norm(W)
    d = L/5

    pImg = pImg.reshape(-1,2)
    cv2.line(frame, tuple(np.int32(pImg[0])),
             tuple(np.int32(pImg[1])), (0, 0, 255), 3)  # x
    cv2.line(frame, tuple(np.int32(pImg[0])),
             tuple(np.int32(pImg[2])), (0, 255, 0), 3)  # y
    cv2.line(frame, tuple(np.int32(pImg[0])),
             tuple(np.int32(pImg[3])), (255, 0, 0), 3)  # z


    cv2.putText(frame, "rvec :{}".format(rvec), (50,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, "tvec :{}".format(tvec), (50,450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    cv2.imshow("POSE", frame)

    cv2.waitKey(0)

def main():
    find_CCC()

if __name__ == "__main__":
    main()