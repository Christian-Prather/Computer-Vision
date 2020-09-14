import numpy as np

# Create XYZ rotation matrix from ax, ay, az (radian) rotations
def rotationMatrix(ax, ay, az):
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
    return Rot
