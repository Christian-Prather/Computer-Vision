import numpy as np
import cv2

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z



# radian rotations
ax, ay, az = 1.1, -0.5, 0.1

# cos and sin defines
sx, sy, sz = np.sin(ax), np.sin(ay), np.sin(az)
cx, cy, cz = np.cos(ax), np.cos(ay), np.cos(az)

# Define 2D rotation matricies
Rot_x = np.array(((1,0,0), (0,cx,-sx), (0,sx,cx)))
Rot_y = np.array(((cy,0,sy), (0,1,0), (-sy,0,cy)))
Rot_z = np.array(((cz,-sz,0), (sz,cz,0), (0,0,1)))

# Combine all 3 in (x -> y -> z) order
Rot = Rot_z @ Rot_y @ Rot_x
print("ROTATION:")
print(Rot)

# Inverse of Rot 
Rot_Inv = np.linalg.inv(Rot)

# Transpose
Rot_Tran = Rot.transpose()

print("\nInverse = Transpose")
print(Rot_Inv)
print()
print(Rot_Tran)

# ZYX fixed angles
Rot_ZYX = Rot_x @ Rot_y @ Rot_z
print("ZYX:")
print(Rot_ZYX)

###################################################################
cam_origin_world = np.array([[10, -25, 40]]).transpose()
print(cam_origin_world)

# Homogeneous transformation matrix camera -> world 
H_cam_to_world = np.block([[Rot, cam_origin_world], [0,0,0,1]])

print("C->W")
print(H_cam_to_world)
print()
# Homogeneous transformation matrix world -> camera
H_world_to_cam = np.linalg.inv(H_cam_to_world)

print("W->C")
print(H_world_to_cam)
print()

# Intrinsic Camera Calibration matrix
focal_length = 400
fx = focal_length
fy = focal_length

img_width = 256
img_height = 170 

# Optical center at image center
cx = img_width / 2
cy = img_height / 2


K = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)))
print(K)
print()

# Projecting points (world coordinates) to image
points = []
point1 = Point(6.8158, -35.1954, 43.0640)
points.append(point1)
point2 = Point(7.8493, -36.1723, 43.7815)
points.append(point2)
point3 = Point(9.9579, -25.2799, 40.1151)
points.append(point3)
point4 = Point(8.8219, -38.3767, 46.6153)
points.append(point4)
point5 = Point(9.5890, -28.8402, 42.2858)
points.append(point5)
point6 = Point(10.8082, -48.8146, 56.1475)
points.append(point6)
point7 = Point(13.2690, -58.0988, 59.1422)
points.append(point7)

# Blank Image 
blank = np.zeros(shape = [img_height, img_width, 3], dtype=np.uint8)

M_ext = np.delete(H_world_to_cam, 3, 0)

# Iterate over the points
img_points = []
n = 0
for point in points:
    P_w = np.array([point.x, point.y, point.z, 1])
    p = K @ M_ext @ P_w
    p = p / p[2]
    img_point = (int(p[0]), int(p[1]))
    img_points.append(img_point)
    blank = cv2.circle(blank, img_point, 2, (255,255,255), 2)
    print(p)
    print()

# Draw lines
for img_point in range(len(img_points) -1):
    blank = cv2.line(blank, img_points[n], img_points[n+1], (255,255,255))
    n+=1

cv2.imshow("Lab 2", blank)

cv2.waitKey(0)
cv2.destroyAllWindows()