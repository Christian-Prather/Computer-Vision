import cv2
import sys

# Mouse callback function. Appends the x,y location of mouse click to a list.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        param.append((x, y))


def main():
    offset = 0
    frame_number = 0
    coordinates = []
    video_capture = cv2.VideoCapture("./building.avi")
    can_read, frame = video_capture.read()
    if not can_read:
        print("Error reading file check path and format")
        sys.exit()
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWriter = cv2.VideoWriter("Lab3.avi", fourcc=fourcc, fps=30.0, frameSize=(641, 481))
   
    # Read in first frame for template selection
    can_read, first_frame = video_capture.read()
    print(first_frame.shape)
    cv2.imshow("Video", first_frame)
    cv2.setMouseCallback("Video", get_xy, coordinates)
    # Once selected user needs to hit enter to progress
    cv2.waitKey(0)
    frame_number+=1

    # Get Template as img
    x_coor, y_coor = coordinates[0]
    print("P1 {} P2 {}".format((y_coor-15, x_coor-15), (y_coor+15, x_coor+15)))
    cropped = first_frame[y_coor-15:y_coor+15, x_coor-15:x_coor+15]
    cropped_x, cropped_y, _ = cropped.shape
    print("T_x {} T_y {}".format(cropped_x, cropped_y))

    # Show rectangle around template on first frame
    cv2.rectangle(first_frame, (x_coor-15, y_coor-15), (x_coor+15, y_coor+15),(255,255,255), thickness=1, lineType=8)
    cv2.imshow("Video", first_frame)
    videoWriter.write(first_frame)

    # cv2.imshow("Template",cropped)

    # Must hit enter again to show rectange on video
    cv2.waitKey(0)

    while True:
        can_read, frame = video_capture.read()
        if not can_read:
            break
        
        # Calculate highestr score
        scores = cv2.matchTemplate(frame, cropped, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(scores)

        # Show rectangle around highest scoring point
        print("Max Location: {} Coorelation Score {}".format(max_loc, max_val))
        x_coor, y_coor = max_loc
        x_coor += int(cropped_x/2)
        y_coor += int(cropped_y/2)
        cv2.rectangle(frame, (x_coor-15, y_coor-15), (x_coor+15, y_coor+15),(255,255,255), thickness=1, lineType=8)

        cv2.putText(frame, "Frame: {}".format(str(frame_number)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Video", frame)
        frame_number+=1

        # Save to video
        videoWriter.write(frame)

        cv2.waitKey(30)
    videoWriter.release()

if __name__ == "__main__":
    main()
