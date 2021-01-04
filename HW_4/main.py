import cv2
import numpy as np

# Load aruco tag dictionary
arucoDictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
# Size of aruco tag
MARKER_LENGTH = 2
# Adjust for camera distrotion
dist_coeff = 0

# Read in video frame at a time
# video_capture = cv2.VideoCapture("hw4.avi")
video_capture = cv2.VideoCapture("mine.MOV")

# Save video
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
# videoWriter = cv2.VideoWriter("hw4_output.avi", fourcc=fourcc, fps=30.0, frameSize=(int(video_capture.get(3)), int(video_capture.get(4))))
videoWriter = cv2.VideoWriter("hw4_custom_output.avi", fourcc=fourcc, fps=30.0, frameSize=(int(video_capture.get(3)), int(video_capture.get(4))))


# Camera intrinsics assuming no distrotion
f = 675
# cx = 320
# cy = 240
# mine
cx = 960
cy = 540

K = np.array(((f, 0.0, cx), (0.0, f, cy), (0.0, 0.0, 1.0)))

# Find the aruco pose in frame from aruco tag corners
# Returns: modified frame, rotation and translation vector
def compute_aruco_pose(corners, frame):
    rvecs,tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners=corners, markerLength=MARKER_LENGTH,
                    cameraMatrix = K, distCoeffs=dist_coeff
    )

    # Rotation/ translationvector of single aruco tag
    rvec_m_c_ = rvecs[0]
    tm_c = tvecs[0]

    cv2.aruco.drawAxis(image=frame, cameraMatrix=K, distCoeffs=dist_coeff, 
                        rvec= rvec_m_c_, tvec =tm_c, length= MARKER_LENGTH)

    return frame, rvec_m_c_, tm_c

# Helper to list all aruco tags in dictionary
def print_all_tags():
    for id in range(0,100):
        img = cv2.aruco.drawMarker(dictionary=arucoDictionary, id=id, sidePixels=200)
        cv2.imshow("IMG", img)
        # cv2.imwrite("5.jpg", img)
        cv2.waitKey(30)

# Locate the aruco tag in frame 
def find_aruco(frame):
    corners, ids, _ = cv2.aruco.detectMarkers(image=frame, dictionary=arucoDictionary)
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0,0,255))
        frame, rvec, tvec = compute_aruco_pose(corners, frame)
        frame = augmentation(rvec,tvec,ids[0], frame)
    videoWriter.write(frame)
    # Show modified image
    cv2.imshow("Tags", frame)



# Project pyramid points on to image
def augmentation(rvec, tvec, pose_id, frame):
    px = -14.0
    py = -14.0
    pz = 1.0
    # Manual place points in reference to tip at a (+- (2,1,1) x,y,z offest)
    switch_points_1 = np.array([[-2.5, -2.0, -5.0], [-4.5,-1,-6], [-4.5,-1,-4], [-4.5,-3,-6], [-4.5, -3, -4]])
    switch_points_0 = np.array([[2.5, -2.0, -1.0], [3.5, -1, -2],[3.5,-1, 0],[3.5,-3,-2],[3.5,-3,0]])
    switch_points_5 = np.array([[px, py, pz], [px -1.0, py+1.0, pz +2.0],[px +1.0,py +1.0, pz +2.0], [px -1.0,py -1.0, pz+2.0],[px +1.0, py -1.0, pz +2.0]])

    if pose_id == 1:
        # Find the points projections from the aruco tag pose
        pImg, _ = cv2.projectPoints(objectPoints = switch_points_1, rvec = rvec, tvec = tvec, cameraMatrix= K, distCoeffs = None)
        pImg = pImg.reshape(-1,2)
        # print("im", pImg[0])
        # Silly mehtos for printing lines connection pyramid points onto image
        for point in pImg:
            cv2.line(frame, tuple(np.int32(point)), tuple(np.int32(pImg[0])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[1])), tuple(np.int32(pImg[2])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[2])), tuple(np.int32(pImg[4])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[3])), tuple(np.int32(pImg[4])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[3])), tuple(np.int32(pImg[1])), (255,255,255))

    # Referencing aruco tag 0
    elif pose_id == 0:
        # Find projection of pyramid to image
        pImg, _ = cv2.projectPoints(objectPoints = switch_points_0, rvec = rvec, tvec = tvec, cameraMatrix= K, distCoeffs = None)
        pImg = pImg.reshape(-1,2)
         # Silly mehtos for printing lines connection pyramid points onto image
        for point in pImg:
            cv2.line(frame, tuple(np.int32(point)), tuple(np.int32(pImg[0])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[1])), tuple(np.int32(pImg[2])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[2])), tuple(np.int32(pImg[4])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[3])), tuple(np.int32(pImg[4])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[3])), tuple(np.int32(pImg[1])), (255,255,255))

    # Referencing aruco tag 0
    elif pose_id == 5:
        # Find projection of pyramid to image
        pImg, _ = cv2.projectPoints(objectPoints = switch_points_5, rvec = rvec, tvec = tvec, cameraMatrix= K, distCoeffs = None)
        pImg = pImg.reshape(-1,2)
         # Silly mehtos for printing lines connection pyramid points onto image
        for point in pImg:
            cv2.line(frame, tuple(np.int32(point)), tuple(np.int32(pImg[0])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[1])), tuple(np.int32(pImg[2])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[2])), tuple(np.int32(pImg[4])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[3])), tuple(np.int32(pImg[4])), (255,255,255))
        cv2.line(frame, tuple(np.int32(pImg[3])), tuple(np.int32(pImg[1])), (255,255,255))

    # No tag in frame
    else:
        pass
    return frame
    
# Loop through video frame at a time getting aruco tag pose and mapping virtual
# object to image
def main():
    # print_all_tags()
    while True:
        can_read, frame = video_capture.read()
        if not can_read:
            print("Error reading file check path and format")
            sys.exit()
        cv2.imshow("Source", frame)
        find_aruco(frame) 

        # threshold_frame(frame)
        cv2.waitKey(100) 


if __name__ == "__main__":
    main()
