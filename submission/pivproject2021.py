import cv2
import numpy as np
import sys
import os

""""
***********************
Functions used for Task1
************************
"""

def initArucoPos(template, aruco_dict, arucoParameters):
    """
    Returns the corners of the Aruco markers in the template image. Note that 
    the corners sequence is the same as the ids sequence. 
    Example: if ids[0] has value 2 then corners[0] has corners of Aruco marker 
    with id=2.
    :param template: template image
    :param aruco_dict: dictionary of Aruco codes used in the template image
    """
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_template, aruco_dict, parameters=arucoParameters)
    if len(corners) == 0:
        print("getCorners: Could not detect Aruco markers. Exiting.")
        exit(0)
    return corners, ids

def getArucos(img):
    """
    Returns the Aruco markers' coordinates (in pixels)
    and ids. The order of the coordinates is the same as the order of the ids. 
    Example: if ids[0] has value 2 then corners[0] has corners of Aruco marker 
    with id=2.
    :param img: input image
    """
    if img is None:
        print("getCorners: Unable to read the template.")
        exit(-1)
    # The arucos in the template must be the ones in this dictionary
    dict4_7by7 = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_50)
    arucoParameters = cv2.aruco.DetectorParameters_create()
    corners, ids = initArucoPos(img, dict4_7by7, arucoParameters)
    dict = {"corners": corners, "ids": ids}
    return dict

def getSourceCorners(arucos):
    """
    Returns the corners' points from the detected arucos data structure obtained
    using getArucos
    :param arucos: Aruco markers' coordinates (in pixels) and ids.
    """
    sourceCorners = []
    numb_arucos = len(arucos["corners"])
    for i in range(numb_arucos):
        for j in range(arucos["corners"][i][0].shape[0]):
            sourceCorners.append(np.array(arucos["corners"][i][0][j]))
    return np.array(sourceCorners)

def getDestCorners(sourceArucosIDs,referenceCorners):
    """
    Returns an array with the arucos' points whose ID is present in sourceArucosIds 
    by getting those points from referenceCorners. Used to get the points of the 
    template associated to the detected arucos points.
    :param sourceIDs: ids of the corners detected in the source image
    :param referenceCorners: dictionary with the template's arucos ids as keys 
        and the respective arucos points as entries
    """
    destCorners=[]
    for corner_id in sourceArucosIDs:
        corner = referenceCorners[corner_id[0]]
        for point in corner: 
            destCorners.append(point)
    return np.array(destCorners)

def getReferenceCorners(referenceArucos):
    """
    Returns a dictionary with the template's arucos ids as keys and the respective arucos
    points as entries. Used to get a reference of the arucos points in the template
    that are used to obtain the homography
    :param referenceArucos: Aruco markers' coordinates (in pixels) and ids
    """
    referenceCorners={}
    numbArucos=len(referenceArucos["corners"])
    for i in range(numbArucos):
        referenceCorners[referenceArucos["ids"][i][0]] = [corner for corner 
                                                          in referenceArucos["corners"][i][0]]
    return referenceCorners

def findHomography(sourcePoints, destPoints):
    A=[]
    for i in range(len(sourcePoints)):
        x1=sourcePoints[i][0]
        y1=sourcePoints[i][1]
        x2=destPoints[i][0]
        y2=destPoints[i][1]
        line1=[x1,y1,1,0,0,0,-x1*x2,-y1*x2,-x2]
        line2=[0,0,0,x1,y1,1,-x1*y2,-y1*y2,-y2]
        A.append(line1)
        A.append(line2)
    u, s, vh = np.linalg.svd(A,full_matrices=True)#svd decomposition
    h=vh[-1]
    h=h.reshape((3,3))
    return h

""""
***********************
Functions used for Task2
************************
"""

def compute_SIFT(image_1, image_2, des_2, key_2, detector, flann, MIN_MATCH_COUNT=7, ratio_tresh=0.7):
    """
    Change the point of view of a image, the goal is to have the same one from 
    the template
    Use of the OpenCV library. 
    Input : image_1 : image to compute 	: type numpy array
            image_2 : the template 	: type numpy array
            MIN_MATCH_COUNT: Minimun number of good matches : type int
            des_2 : SIFT descriptors of the template : np.array
            key_2 : SIFT keys of the template : np.array
    """
    # Find the keys and the descriptors with SIFT
    key_1, des_1 = detector.detectAndCompute(image_1, None)
    # Find all the matches
    matches = flann.knnMatch(des_1, des_2, k = 2)
    # Take all the good matches
    good_matches = []
    # Filter matches using the Lowe's ratio test
    for m,n in matches:
        if m.distance < ratio_tresh*n.distance:
            good_matches.append(m)
    # To compute the function, we need a minimum of efficient matches
    if len(good_matches) > MIN_MATCH_COUNT:
        # Get the point from the match of the image and the template
        src_points = np.float32([key_1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_points = np.float32([key_2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        return src_points, dst_points
    else:
        # Not enough good matches
        return np.array([]), np.array([])

    
""""
***********************
Functions used for Task4
************************
"""

def get_mask_skin(frame):
    """
    Returns a mask for all pixels in the given image that match the specified
    range for skin tones. Used for hand removal.
    """
    # Target color range in HSV color space
    min_skin_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_skin_HSV = np.array([33, 255, 255], dtype = "uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_skin_HSV, max_skin_HSV)
    # Fill in holes in the mask using morpholoical closing
    skinRegionHSV = cv2.morphologyEx(skinRegionHSV, cv2.MORPH_CLOSE, np.ones((121,121), np.uint8))
    # Smooth out the final mask using a Gaussian filter
    mask = cv2.GaussianBlur(skinRegionHSV, (0,0), 9, 9)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    return mask

def get_mask_sleeve(frame):
    """
    Returns a mask for all pixels in the given image that match the specified
    range for objects like sleeves and pens. Used for hand removal.
    """
    # Target color range in HSV color space
    min_sleeve_HSV = np.array([10,0,0],dtype="uint8")
    max_sleeve_HSV = np.array([120,255,80],dtype="uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sleeveRegionHSV = cv2.inRange(imageHSV, min_sleeve_HSV, max_sleeve_HSV)
    # Fill in holes in the mask using morpholoical closing
    sleeveRegionHSV = cv2.morphologyEx(sleeveRegionHSV, cv2.MORPH_CLOSE, np.ones((121,121), np.uint8))
    # Smooth out the final mask using a Gaussian filter
    mask = cv2.GaussianBlur(sleeveRegionHSV, (0,0), 9, 9)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    return mask


def check_homography(H, img, scale_factor):
    """
    Returns true if the given homography (H) is reasonable for the input image 
    (img) and the parameters of the current transformation problem, false otherwise.
    """
    # If the determinant of the rotation submatrix is negative, the orientation
    # of the image is flipped, and so the homography can be immediately discarded
    det2 = H[0,0] * H[1,1] - H[0,1] * H[1,0]
    if (det2 <= 0.1):
        return False
    # If the determinant of the homography matrix is negative, the orientation
    # of the image is flipped, and so the homography can be immediately discarded
    det3 = np.linalg.det(H)
    if (det3 <= 0.1):
        return False
    
    # To verify if the homography is reasonable, apply the transformation to the
    # 4 corner points of the image
    h,w,d = img.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    [[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]] = dst
    # Compute the original area of the image
    og = img.shape[0] * img.shape[1]
    # Compute the area of the transformed image
    area = (x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1)
    area = abs(area * 0.5)
    # Compute the ratio between the two areas
    ratio = area / og
    
    # Define the margin of error for the following checks
    MAX_SCALE = 4
    # Considering the previous values as areas and volumes, convert to length scale
    ratio_len = np.sqrt(ratio)
    det2_len = np.sqrt(det2)
    det3_len = np.cbrt(det3)
    # The calculated scale factor for the area of the image should be similar to 
    # the scale factor between the original image and the target template size 
    # (considering that the paper template should take up a significat portion of the image)
    if (ratio_len < (scale_factor/MAX_SCALE)) or (ratio_len > (scale_factor*MAX_SCALE)):
        return False
    # The scale factors for the transformation should be similar to the scale factor
    # with the template size
    if (det2_len < (scale_factor/MAX_SCALE)) or (det2_len > (scale_factor*MAX_SCALE)):
        return False
    if (det3_len < (scale_factor/MAX_SCALE)) or (det3_len > (scale_factor*MAX_SCALE)):
        return False
    # The scale factors for the transformation should be similar to the calculated
    # scale factor for the area of the image (compared separately to make sure that
    # the margin of error doesn't stack up)
    if (det2_len < (ratio_len/MAX_SCALE)) or (det2_len > (ratio_len*MAX_SCALE)):
        return False
    if (det3_len < (ratio_len/MAX_SCALE)) or (det3_len > (ratio_len*MAX_SCALE)):
        return False
    # If all scale factor checks are passed, the homography can be used
    return True

"""
***********************
Main
***********************
"""
task = int(sys.argv[1])
template_path = sys.argv[2]
output_path = sys.argv[3]

if task==1:
    input_images_path = sys.argv[4]
    img_template = cv2.imread(template_path)
    #Arucos from the template
    referenceArucos = getArucos(img_template)
    #Coordinates from the Arucos' corners in the template
    referenceCorners=getReferenceCorners(referenceArucos)
    input_images = os.listdir(input_images_path)
    for i in range(len(input_images)):
        img_name = input_images[i]
        frame = cv2.imread(input_images_path+"/"+img_name)
        arucos=getArucos(frame)
        if(len(arucos["ids"])>0):
            #Corners from the frame
            corners = getSourceCorners(arucos)
            #Corresponding corners from the template
            destCorners = getDestCorners(arucos["ids"],referenceCorners)
            H = findHomography(corners, destCorners)
        rotated = cv2.warpPerspective(frame,H, (img_template.shape[1],img_template.shape[0]))
        cv2.imwrite(output_path+"/"+img_name,rotated)

elif task==2:
    input_images_path = sys.argv[4]
    input_images = os.listdir(input_images_path)
    img_template = cv2.imread(template_path)
    # Normalize the template for more consistent outputs
    template_norm = cv2.normalize(img_template, None, 0, 255, cv2.NORM_MINMAX)
    
    # SIFT detector
    detector = cv2.SIFT_create()
    key_template, des_template = detector.detectAndCompute(img_template, None)
    
    # FLANN Matcher
    FLANN_INDEX_KDTREE = 0
    RATIO_TRESH = 0.85
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)

    START = 0
    # Compute the scale factor between the image and template lenghts, used as 
    # an approximation of the scale factor for the final transformation
    scale_factor = img_template.shape[0] / cv2.imread(input_images_path+"/"+input_images[0]).shape[0]
    # The more the image needs to be scaled up to match the template, the higher
    # the margin of error for the matches should be
    reproj_thresh = int(15 * scale_factor)
    
    missed = False
    prev_frame = template_norm
    
    for i in range(START, len(input_images)):
        img_name = input_images[i]
        print("\n"+img_name)
        frame = cv2.imread(input_images_path+"/"+img_name)
        # Find dst_points and src_points using SIFT
        src_points, dst_points = compute_SIFT(frame, img_template, 
                                              des_template, key_template, 
                                              detector, flann, 
                                              ratio_tresh=RATIO_TRESH)

        # Minimum number of matching points required to compute the homography
        POINTS_THRESH = 4
        # Overestimation for the number of iterations needed to find a good match
        MAX_ITERS = 300
        
        if (len(dst_points)>POINTS_THRESH):
            # If the numeber of good matches is good enough, compute the homography
            # Compute the homography with the Ransac method
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 
                                         reproj_thresh, maxIters=MAX_ITERS)
            good = False
            if (H is not None):
                # Check if the homography is acceptable
                good = check_homography(H, frame, scale_factor)
            if (H is None) or (not good):
                # If the homography can't be found or is not acceptable, do not use it
                print('### No H')
                missed = True
            else:
                missed = False
        else :
            # If the numeber of good matches is not good enough, do not compute the homography
            print('### No matches')
            missed = True
    
        if (H is None) or (missed):
            # If there is no acceptable homography, hold the last good frame
            cv2.imwrite(output_path+"/"+img_name, prev_frame)
            continue
        
        # Compute the final frame and normalize it
        rotated = cv2.warpPerspective(frame, H, (img_template.shape[1], 
                                                 img_template.shape[0]))
        rotated = cv2.normalize(rotated, None, 0, 255, cv2.NORM_MINMAX)
        
        cv2.imwrite(output_path+"/"+img_name, rotated)
        # Update the last good frame
        prev_frame = rotated

elif task == 4:
    camera1_images_path = sys.argv[4]
    camera2_images_path = sys.argv[5]
    camera1_images = os.listdir(camera1_images_path)
    camera2_images = os.listdir(camera2_images_path)
    img_template = cv2.imread(template_path)
    template_norm = cv2.normalize(img_template, None, 0, 255, cv2.NORM_MINMAX)
    
    # SIFT detector
    detector = cv2.SIFT_create()
    key_template, des_template = detector.detectAndCompute(img_template, None)
    
    #FLANN Matcher
    FLANN_INDEX_KDTREE = 0
    RATIO_TRESH = 0.85
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    
    START = 0
    prev_frame = template_norm
    
    # Compute the scale factors for the two inputs
    data1_sample =  cv2.imread(camera1_images_path+"/"+camera1_images[0])
    scale_factor1 = img_template.shape[0] / data1_sample.shape[0]
    reproj_thresh1 = int(15 * scale_factor1)
    data2_sample = cv2.imread(camera2_images_path+"/"+camera2_images[0])
    scale_factor2 = img_template.shape[0] / data2_sample.shape[0]
    reproj_thresh2 = int(15 * scale_factor2)
    
    for i in range(START, len(camera1_images)):
        img1_name = camera1_images[i]
        img2_name = camera2_images[i]
        print(img1_name)
        img1 = cv2.imread(camera1_images_path+"/"+img1_name)
        img2 = cv2.imread(camera2_images_path+"/"+img2_name)

        # Find dst_points and src_points using SIFT
        src_points1, dst_points1 = compute_SIFT(img1, img_template, 
                                                des_template, key_template, 
                                                detector, flann, 
                                                ratio_tresh=RATIO_TRESH)
        src_points2, dst_points2 = compute_SIFT(img2, img_template, 
                                                des_template, key_template, 
                                                detector, flann, 
                                                ratio_tresh=RATIO_TRESH)
        
        POINTS_THRESH = 4
        MAX_ITERS = 300
        
        # If no good homography is found, the value stays as None
        rotated1 = None
        # Find and compute the two homographies the same way as Task 2
        if(len(dst_points1)>POINTS_THRESH):
            H1, mask1 = cv2.findHomography(src_points1, dst_points1, cv2.RANSAC, 
                                           reproj_thresh1, maxIters=MAX_ITERS)
            good1 = False
            if (H1 is not None):
                good1 = check_homography(H1, img1, scale_factor1)
            if (H1 is not None) and good1:
                rotated1 = cv2.warpPerspective(img1, H1, (img_template.shape[1], 
                                                          img_template.shape[0]))
                rotated1 = cv2.normalize(rotated1, None, 0, 255, cv2.NORM_MINMAX)
            else:
                print('### No H1')
        else:
            print('### No matches 1')
            
        rotated2 = None
        if(len(dst_points2)>POINTS_THRESH):
            H2, mask2 = cv2.findHomography(src_points2, dst_points2, cv2.RANSAC, 
                                           reproj_thresh2, maxIters=MAX_ITERS)
            good2 = False
            if (H2 is not None):
                good2 = check_homography(H2, img2, scale_factor2)
            if (H2 is not None) and good2:
                rotated2 = cv2.warpPerspective(img2, H2, (img_template.shape[1], 
                                                          img_template.shape[0]))
                rotated2 = cv2.normalize(rotated2, None, 0, 255, cv2.NORM_MINMAX)
            else:
                print('No H2')
        else:
            print('No matches 2')
            
        curr = None
        
        if rotated1 is not None:
            curr = rotated1
        if rotated2 is not None:
            if curr is not None:
                # If both inputs have given an acceptable result, merge the two together
                # Mask out the skin and other objects areas
                mask_skin = get_mask_skin(curr)
                mask_diff = get_mask_sleeve(curr)
                # Mask out the black regions where the images had no data
                bw = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                mask_white = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY_INV)[1]
                mask_black = cv2.bitwise_not(mask_white)
                # Combine the masks
                mask = cv2.bitwise_and(mask_skin, mask_diff)
                mask = cv2.bitwise_and(mask, mask_black)
                mask_inv = cv2.bitwise_not(mask)
                # Apply the mask to the first image and fill the remaining areas
                # with the second one
                curr_fg = cv2.bitwise_and(curr, curr, mask=mask)
                new_bg = cv2.bitwise_and(rotated2, rotated2, mask=mask_inv)
                curr = cv2.add(curr_fg, new_bg)
            else:
                curr = rotated2
        if curr is not None:
            # If any of the inputs have given an acceptable result, fill in the remaining
            # empty areas with the original template
            # Same merging operations as the previous step
            mask_skin = get_mask_skin(curr)
            mask_diff = get_mask_sleeve(curr)
            bw = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            mask_white = cv2.threshold(bw, 0, 255, cv2.THRESH_BINARY_INV)[1]
            mask_black = cv2.bitwise_not(mask_white)
            mask = cv2.bitwise_and(mask_skin, mask_diff)
            mask = cv2.bitwise_and(mask, mask_black)
            mask_inv = cv2.bitwise_not(mask)
            curr_fg = cv2.bitwise_and(curr, curr, mask=mask)
            new_bg = cv2.bitwise_and(template_norm, template_norm, mask=mask_inv)
            curr = cv2.add(curr_fg, new_bg)
        else:
            # If no acceptable result was found, hold the last good frame
            curr = prev_frame

        cv2.imwrite(output_path+"/"+img1_name, curr)
        # Update the last good frame
        prev_frame = curr
