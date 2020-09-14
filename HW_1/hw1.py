import helper
import numpy as np
import cv2

# Found in slides
# Draw three 3D line segments, representing xyz unit axes, onto the axis figure ax.
# H is the 4x4 transformation matrix representing the pose of the coordinate frame.
def draw_coordinate_axes(ax, H, label):
    p = H[0:3, 3]      # Origin of the coordinate frame
    ux = H @ np.array([1,0,0,1])   # Tip of the x axis
    uy = H @ np.array([0,1,0,1])   # Tip of the y axis
    uz = H @ np.array([0,0,1,1])   # Tip of the z axis
    ax.plot(xs=[p[0], ux[0]], ys=[p[1], ux[1]], zs=[p[2], ux[2]], c='r')   # x axis
    ax.plot(xs=[p[0], uy[0]], ys=[p[1], uy[1]], zs=[p[2], uy[2]], c='g')   # y axis
    ax.plot(xs=[p[0], uz[0]], ys=[p[1], uz[1]], zs=[p[2], uz[2]], c='b')   # z axis
    ax.text(p[0], p[1], p[2], label)   # Also draw the label of the coordinate frame

############################################################
# Utility function for 3D plots.
def setAxesEqual(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    ax, ay, az = 0.0, 0.0, 0.5235
    Rot = helper.rotationMatrix(ax, ay, az)

    vehicle_to_world = np.array([[6, -8, 1]]).transpose()
    
    # Homogeneous transformation matrix vehicle -> world 
    H_veh_to_world = np.block([[Rot, vehicle_to_world], [0,0,0,1]])

    ##############################################################
    # Camera to Mount 

    #Rotation of camera to mount frame
    ax, ay, az = 0.0, 0.0, 0.0
    cam_to_mount_rotation = helper.rotationMatrix(ax,ay,az)
    cam_to_mount_translation = np.array([[0, -1.5, 0]]).transpose()

    # Homogeneous transformation matrix camera -> mount 
    H_cam_to_mount = np.block([[cam_to_mount_rotation, cam_to_mount_translation], [0,0,0,1]])


    ##############################################################
    # Mount to Vehicle 

    #Rotation of camera to mount frame
    ax, ay, az = -2.0944, 0.0, 0.0
    mount_to_veh_rotation = helper.rotationMatrix(ax,ay,az)
    mount_to_veh_translation = np.array([[0, 0, 3]]).transpose()

    # Homogeneous transformation matrix mount -> veh 
    H_mount_to_veh = np.block([[mount_to_veh_rotation, mount_to_veh_translation], [0,0,0,1]])


    # Find Homogeneous transformation from Camera to World 
    # Combine all 3 in (c-m -> m-v -> v-w) order
    H_cam_to_world = H_veh_to_world @ H_mount_to_veh @ H_cam_to_mount

    H_world_to_cam = np.linalg.inv(H_cam_to_world)

    # Extrinsic parameters
    M_ext = np.delete(H_world_to_cam, 3, 0)

    # Instrinisc parameters
    # Intrinsic Camera Calibration matrix
    focal_length = 600
    fx = focal_length
    fy = focal_length

    img_width = 640
    img_height = 480 

    # Optical center at image center
    cx = img_width / 2
    cy = img_height / 2
    K = np.array(((fx, 0, cx), (0, fy, cy), (0, 0, 1)))

    # Points 
    P_w = np.array(((-1,1,1,-1,0), (-1,-1,1,1,0), (0,0,0,0,3), (1,1,1,1,1)))
    print(P_w)

    p_img = K @ M_ext @ P_w
    p_img = p_img / p_img[2]
    print(p_img)

    image_points = []
    # Blank Image 
    blank = np.zeros(shape = [img_height, img_width, 3], dtype=np.uint8)
    for i in range(p_img.shape[1]):
        print(i)
        x_img = int(round(p_img[0][i]))
        y_img = int(round(p_img[1][i]))

        point = (x_img, y_img)
        image_points.append(point)
        print(point)
        blank = cv2.circle(blank, point, 2, (255,255,255), 2)

    # Draw lines
    n = 0
    current_biggest = 1000
    top_point = image_points[0]
    for img_point in range(len(image_points)):
        # Top point has smallest y val
        if image_points[n][1] < current_biggest:
            current_biggest = image_points[n][1]
            top_point = image_points[n]
        # blank = cv2.line(blank, image_points[n], image_points[n+1], (255,255,255))
        n+=1
    n = 0
    for img_point in range(len(image_points) -1):
        blank = cv2.line(blank, image_points[n], top_point, (255,255,255))
        blank = cv2.line(blank, image_points[n], image_points[n+1], (255,255,255))


        n+=1
    blank = cv2.line(blank, image_points[len(image_points)-2], image_points[0], (255,255,255))
    
    cv2.imshow("HW 1", blank)

    cv2.waitKey(0)
    cv2.destroyAllWindows()    

    # Plot 
    import matplotlib.pyplot as plt    
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Need all reference to world frame
    H_mount_to_world = H_veh_to_world @ H_mount_to_veh
    print(H_mount_to_world)

    draw_coordinate_axes(ax, np.eye(4), 'W')
    draw_coordinate_axes(ax, H_mount_to_world, 'M')
    draw_coordinate_axes(ax, H_veh_to_world, 'V')
    draw_coordinate_axes(ax, H_cam_to_world, 'C')
    # Really not sure if the pyramid is correct
    draw_coordinate_axes(ax,np.delete(P_w, -1, axis=1), 'P')

    setAxesEqual(ax)
    plt.show()
if __name__ == "__main__":
    main()