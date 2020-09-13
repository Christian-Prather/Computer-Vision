import cv2 
import sys

FOCAL_LENGTH = 500

points = [(-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (1.0, 1.0, 1.0), (-1.0, 1.0, 1.0)]

def calculate_points(point_x,point_y,point_z, offset):
	x = (FOCAL_LENGTH)*(point_x / ((point_z + offset)))
	y = (FOCAL_LENGTH)*(point_y / ((point_z + offset)))

	return x,y


def main():
	offset = 0
	frame_number = 0

	video_capture = cv2.VideoCapture("/home/christian/Downloads/earth.wmv")
	can_read, frame = video_capture.read()
	if not can_read:
		print("Error reading file check path and format")
		sys.exit()
	fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
	videoWriter = cv2.VideoWriter("Lab1.avi", fourcc = fourcc, fps = 30.0, frameSize= (960, 540))

	while True:
		can_read, frame = video_capture.read()
		if not can_read:
			break;
		for point in points:
			# Get each point and pass it to the cal function
			x, y = calculate_points(point[0], point[1], point[2], offset)
			print ("X:{} Y:{}".format(x,y))
			cv2.circle(frame, (int(x) + 480, int(y) + 270), 3, (0,0,255))

		cv2.putText(frame, "Frame: {}".format(str(frame_number)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		cv2.imshow("Video", frame)
		frame_number+=1

		videoWriter.write(frame)

		print("////////////////////////////////////////////////////")
		offset += 0.1
		cv2.waitKey(30)
		print(frame_number)
	videoWriter.release()
if __name__ == "__main__":
	main()