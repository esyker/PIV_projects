import cv2
import numpy as np
from numpy.core.fromnumeric import shape, size
from numpy import reshape
import matplotlib.pyplot as plt

# Path of the image and the template:
query_image_path = ('Input_Image/frame17.png')              # image to compute
train_image_path = ('Dataset/template2_fewArucos.png')      #template

# Import the images : 
image_1 = cv2.imread(query_image_path, cv2.COLOR_BGR2GRAY)
image_2 = cv2.imread(train_image_path, cv2.COLOR_BGR2GRAY)

def compute_SIFT(image_1, image_2):
    """
    Change the point of view of a image, the goal is to have the same one from the template
    Use of the OpenCV library. 
    Input : image_1 : image to compute 	: type numpy array
            image_2 : the template 	: type numpy array
    Outout : None
    """

    # Minimun number of good matches
    MIN_MATCH_COUNT = 7

    # Compute the detector with the feature SIFT
    detector = cv2.xfeatures2d.SIFT_create()

    # Find the keys and the descriptors with SIFT
    key_1, des_1 = detector.detectAndCompute(image_1, None)
    key_2, des_2 = detector.detectAndCompute(image_2, None)

    FLANN_INDEX_KDTREE = 0
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)

    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)

    # Find all the matches
    matches = flann.knnMatch(des_1, des_2, k = 2)
    
    # Take all the good matches
    good_matches = []
    for m,n in matches:
        if m.distance < 0.82*n.distance:
            good_matches.append(m)
    
    # To compute the function, we need a minimum of efficient matches
    if len(good_matches) > MIN_MATCH_COUNT:
        # Get the point from the match of the image and the template
        src_points = np.float32([key_1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_points = np.float32([key_2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    
        # COmpute the homography with the Ransac method
        H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        # Get the points to compute bring back straight the image 
        h = image_1.shape[0]
        w = image_1.shape[1]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0 ] ]).reshape(-1,1,2)

        # Applying the perspectiveTransform() function to transform the perspective of 
        # the given source image to the corresponding points in the destination image
        result_matrix = cv2.perspectiveTransform(pts,H)

        # Use this function to plot the line around the interesting object
        image_2 = cv2.polylines(image_2, [np.int32(result_matrix)], True,255,3, cv2.LINE_AA)
        
        # Configure the parameters 
        draw_params = dict(matchColor = (0,255,0),          # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask,      # draw only inliers
                            flags = 2)

        # Draw the line between the matches of the 2 images and plot the result
        image_3 = cv2.drawMatches(image_1, key_1, image_2, key_2, good_matches, None, **draw_params) 
        plt.imshow(image_3, 'gray'), plt.show()

        # Compute the perspective changing with warpPerspective()
        rotated = cv2.warpPerspective(image_1, H, (image_2.shape[1], image_2.shape[0]))
        frame = cv2.resize(image_1, None, fx = 0.2, fy = 0.2)
        rotated = cv2.resize(rotated, None, fx = 0.2, fy = 0.2)

        # Plot the orginal image and after the homography
        cv2.imshow("origin", frame)
        cv2.imshow("homography", rotated)
        cv2.waitKey(0)

    # If the numeber of good matches is not good enough, do not compute the homography
    # Print an error message
    else :
        print('not enough good matches')
        matchesMask = None


compute_SIFT(image_1, image_2)