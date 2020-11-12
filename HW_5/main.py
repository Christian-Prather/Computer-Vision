import cv2
import numpy as np
FINAL_IMAGE_WIDTH = 2000
FINAL_IMAGE_HEIGHT = 500
MIN_MATCHES_NEEDED = 10
QUARY_DIR = "images"




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
    detector = cv2.ORB_create(nfeatures=3000)
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_scale, mask=None)

    return keypoints, descriptors




def findBaseHomography():
    color_image1 = cv2.imread("mural01.jpg")
    source_points = np.array([[153,118],[383,68], [23,480], [308,504]])
    ortho_points = np.array([[50,50], [369,50], [50, 486], [369,486]])
    display_image = color_image1.copy()
    for x,y in source_points:
        cv2.drawMarker(img= display_image, position=(x,y), color= (255,0,0), markerType=cv2.MARKER_DIAMOND,
                        thickness=3)
    H1, _ = cv2.findHomography(srcPoints=source_points, dstPoints=ortho_points)
    print(H1)


    image_1_ortho = cv2.warpPerspective(color_image1, H1, (FINAL_IMAGE_WIDTH,FINAL_IMAGE_HEIGHT))
    # cv2.imshow("Source Points", display_image)
    cv2.imshow("Warped Image", image_1_ortho)
    return H1

def main():
    homography_previous_2_mosaic = findBaseHomography()
    
    prior_image = cv2.imread("mural01.jpg")
    current_image = cv2.imread("mural02.jpg")

    kp_train, desc_train = detect_features(current_image, show_features=False)
    kp_query, desc_query = detect_features(prior_image, show_features=False)

    match = cv2.BFMatcher.create(cv2.NORM_L2)
    matches = match.knnMatch(desc_query, desc_train, k = 2)
    
    valid = []
    for m,n in matches:
        if m.distance < 0.9 * n.distance:
            valid.append(m)

    homography_current_2_previous = calc_homography_transformation(valid, kp_train, kp_query)
    # matches = [matches[i] for i in range(len(matches)) if inliners[i] == 1]
    image2_ortho = cv2.warpPerspective(current_image, homography_current_2_previous, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))        
    cv2.imshow("Second", image2_ortho)
    
    # To mosaic
    homography_current_2_mosaic = np.dot( homography_previous_2_mosaic, homography_current_2_previous)
    image2_mosaic = cv2.warpPerspective(current_image, homography_current_2_mosaic, (FINAL_IMAGE_WIDTH, FINAL_IMAGE_HEIGHT))

    cv2.imshow("Image2 Mosaic", image2_mosaic)
    
    
    
    
    cv2.waitKey(0)



if __name__ == "__main__":
    main()