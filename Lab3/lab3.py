import cv2

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
    videoWriter = cv2.VideoWriter("Lab3.avi", fourcc=fourcc, fps=30.0, frameSize=(960, 540))
   
    # First Frame
    can_read, first_frame = video_capture.read()
    cv2.imshow("Video", frame)
    cv2.setMouseCallback("Video", get_xy, coordinates)
    cv2.waitKey(0)
    frame_number+=1


    while True:
        can_read, frame = video_capture.read()
        if not can_read:
            break
        cv2.putText(frame, "Frame: {}".format(str(frame_number)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Video", frame)
        frame_number+=1

        videoWriter.write(frame)

        cv2.waitKey(30)
    videoWriter.release()

if __name__ == "__main__":
    main()
