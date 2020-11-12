import os
import cv2
import numpy as np  
QUERY_DIR = "query_images"
MIN_MATCHES_NEEDED = 7

KEY_POINTS = np.float32([[362,238,1], 
                         [72,233,1],
                         [330,368,1]])

print(KEY_POINTS.shape)

def calc_homography_transformation(mathces_in_subset, kp_train, kp_query):
    if len(mathces_in_subset) < 3:
        return None, None
    src_pts = np.float32([kp_train[m.trainIdx].pt for m in mathces_in_subset]).reshape(
        -1,1,2
    )
    dest_pts = np.float32([kp_query[m.queryIdx].pt for m in mathces_in_subset]).reshape(
        -1,1,2
    )
    # A_train_query, inliners = cv2.estimateAffine2D(
    #     src_pts, dest_pts,
    #     method=cv2.RANSAC,
    #     ransacReprojThreshold=4,
    #     maxIters=2500,
    #     confidence=0.95,
    #     refineIters=10
    # )
    # return A_train_query, inliners

def detect_features(image, show_features=False):
    # detector = cv2.xfeatures2d.SURF_create(
    #     hessianThreshold=100,
    #     nOctaves=4,
    #     nOctaveLayer=3,
    #     extend=False,
    #     upright=False
    # )
    detector = cv2.ORB_create(nfeatures=2800)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_scale, mask=None)

    return keypoints, descriptors

def main():
    training_img = cv2.imread("printer_001.png")
    for point in KEY_POINTS:
        point = (point[0], point[1])
        display_image = cv2.circle(training_img, point, radius=10, color=(0,255,0), thickness=5)
    cv2.imshow("Training img", display_image)
    cv2.imwrite("Training_im.jpg", display_image)

    kp_train, desc_train = detect_features(training_img, show_features=False)

    for file in os.listdir(QUERY_DIR):
        querry_img = cv2.imread("query_images/{}".format(file))
        # cv2.imshow("Querry", querry_img)
        # cv2.waitKey(0)
        # Detect features
        kp_query, desc_query = detect_features(querry_img, show_features=False)
        match  = cv2.BFMatcher.create(cv2.NORM_L2)

        matches = match.knnMatch(desc_query,desc_train, k=2)
        valid = []
        for m,n in matches:
            if m.distance < 0.8 * n.distance:
                valid.append(m)
            matches = valid

        A_train_query, inliners = calc_affine_transformation(matches, kp_train, kp_query)
        matches = [matches[i] for i in range(len(matches)) if inliners[i] == 1]
        if A_train_query is not None and sum(inliners) >= MIN_MATCHES_NEEDED:
            print("Obj found...")
            print(file)

            rotation  = np.r_[A_train_query,[[0,0,1]]]
            # print(rotation)
            # print(rotation.shape)

            # warped_points = cv2.perspectiveTransform(src=KEY_POINTS, m=rotation)
            for point in KEY_POINTS:
                warped_point = np.matmul(rotation, point)
            # warped_points = cv2.warpAffine(src=KEY_POINTS, M=A_train_query, dsize=(KEY_POINTS.shape[1], KEY_POINTS.shape[0]))
            # print(warped_points)
            # for point in warped_points:
                new_point = (int(warped_point[0]), int(warped_point[1]))
                querry_display_image = cv2.circle(querry_img, new_point, radius=10, color=(0,255,0), thickness=5)

            cv2.imshow("Warped", querry_display_image)
            cv2.waitKey(0)
            cv2.imwrite("{}_out.jpg".format(file), querry_display_image)
if __name__ == "__main__":
    main()