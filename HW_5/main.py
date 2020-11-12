import cv2
import numpy as np
import os

FINAL_IMAGE_WIDTH = 3000
FINAL_IMAGE_HEIGHT = 1000
MIN_MATCHES_NEEDED = 20
QUARY_DIR = "personal_images"


def create_named_window(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h = image.shape[0]
    w = image.shape[1]

    WIN_MAX_SIZE = 1500
    if max(w,h) > WIN_MAX_SIZE:
        scale = WIN_MAX_SIZE / max(w,h)
    else:
        scale = 1
    cv2.resizeWindow(winname = window_name, width = int(w * scale), height = int(h * scale))

def calc_homography_transformation(mathces_in_subset, kp_train, kp_query):
    if len(mathces_in_subset) < MIN_MATCHES_NEEDED:
        return None, None
    src_pts = np.float32([kp_train[m.trainIdx].pt for m in mathces_in_subset]).reshape(
        -1,1,2
    )
    dest_pts = np.float32([kp_query[m.queryIdx].pt for m in mathces_in_subset]).reshape(
        -1,1,2
    )

    H, _ = cv2.findHomography(srcPoints=src_pts, dstPoints=dest_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    # A_train_query, inliners = cv2.estimateAffine2D(
    #     src_pts, dest_pts,
    #     method=cv2.RANSAC,
    #     ransacReprojThreshold=4,
    #     maxIters=2500,
    #     confidence=0.95,
    #     refineIters=10
    # )
    # return A_train_query, inliners
    return H

def detect_features(image, show_features=False):
    detector = cv2.ORB_create(nfeatures=3500)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_scale, mask=None)

    return keypoints, descriptors




def findBaseHomography():
    color_image1 = cv2.imread("{}/image01.JPG".format(QUARY_DIR))
    # For Mosaic
    # source_points = np.array([[153,118],[383,68], [23,480], [308,504]])
    # ortho_points = np.array([[50,50], [369,50], [50, 486], [369,486]])
    source_points = np.array([[2145,803],[3883,616], [2074,2844], [3955,2871]])
    ortho_points = np.array([[350,50], [669,50], [350, 486], [669,486]])
    display_image = color_image1.copy()
    for x,y in source_points:
        cv2.drawMarker(img= display_image, position=(x,y), color= (255,0,0), markerType=cv2.MARKER_DIAMOND,
                        thickness=3)
    H1, _ = cv2.findHomography(srcPoints=source_points, dstPoints=ortho_points)
    print(H1)


    # image_1_ortho = cv2.warpPerspective(color_image1, H1, (FINAL_IMAGE_WIDTH,FINAL_IMAGE_HEIGHT))
    # create_named_window("Warped Image", image_1_ortho)
    # # cv2.imshow("Source Points", display_image)
    # cv2.imshow("Warped Image", image_1_ortho)
    return H1, color_image1

def fuse_color_images(A,B):
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)

    C = np.zeros(A.shape, dtype=np.uint8)

    A_mask = np.sum(A, axis=2) >0
    B_mask = np.sum(B,axis=2) > 0

    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask

    # cv2.imshow("A", A_only)
    # cv2.waitKey(0)

    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]

    return C

def main():
    warped_images = []
    homography_previous_2_mosaic, start_image = findBaseHomography()
    start_warpped_image = cv2.warpPerspective(start_image, homography_previous_2_mosaic, (FINAL_IMAGE_WIDTH,FINAL_IMAGE_HEIGHT))
    warped_images.append(start_warpped_image)

    prior = "01".zfill(2)
    # Set me for how many images in folder
    for i in range(2, 11):
        # if file != "mural01.jpg":
        file_name = str(i).zfill(2)
        prior_image = cv2.imread("{}/image{}.JPG".format(QUARY_DIR, prior))
        current_image = cv2.imread("{}/image{}.JPG".format(QUARY_DIR, file_name))

        kp_train, desc_train = detect_features(current_image, show_features=False)
        kp_query, desc_query = detect_features(prior_image, show_features=False)

        match = cv2.BFMatcher.create(cv2.NORM_L2)
        matches = match.knnMatch(desc_query, desc_train, k = 2)

        # prior_image = cv2.imread("mural01.jpg")
        # current_image = cv2.imread("mural02.jpg")

        valid = []
        for m,n in matches:
            if m.distance < 0.8 * n.distance:
                valid.append(m)

        homography_current_2_previous = calc_homography_transformation(valid, kp_train, kp_query)
        
        # To mosaic
        homography_current_2_mosaic = np.dot( homography_previous_2_mosaic, homography_current_2_previous)
        image_mosaic = cv2.warpPerspective(current_image, homography_current_2_mosaic, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))
        warped_images.append(image_mosaic)
        print("Prior {} Current {}".format(prior, file_name))

        # # Stich
        # final_output_image = fuse_color_images(final_output_image, image_mosaic)
    
        # Update prior variables
        homography_previous_2_mosaic = homography_current_2_mosaic
        prior = file_name


        # create_named_window("Image Mosaic {}".format(file_name), image_mosaic)
        # cv2.imshow("Image Mosaic {}".format(file_name), image_mosaic)
        # cv2.waitKey(0)

    # create_named_window("Final Image", final_output_image)
    final_output_image = np.zeros((FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, 3), dtype=np.uint8)
    for image_mosaic in warped_images:
        # Stich
        final_output_image = fuse_color_images(final_output_image, image_mosaic)
        cv2.imshow("M", image_mosaic)
        cv2.imshow("TMP", final_output_image)
        cv2.waitKey(0)

    create_named_window("Final Image", final_output_image)
    cv2.imshow("Final Image", final_output_image)

    # counter = 1
    # for image in warped_images:
    #     cv2.imshow("Image {}".format(counter), image)
    #     counter+=1
    
    cv2.waitKey(0)
    cv2.imwrite("Final_personal.jpg", final_output_image)

    # Stich all images together


if __name__ == "__main__":
    main()