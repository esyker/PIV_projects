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
    Returns the corners of the Aruco markers in the template image. Note that the corners sequence is the same as
    the ids sequence. Example: if ids[0] has value 2 then corners[0] has corners of Aruco marker with id=2.
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
    and ids. The order of the coordinates is the same as the order of the ids. Example: if ids[0] has value 2 then
    corners[0] has corners of Aruco marker with id=2.
    :param img: input image
    """
    if img is None:
        print("getCorners: Unable to read the template.")
        exit(-1)
    #The arucos in the template must be the ones in this dictionary
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
    :param referenceCorners: dictionary with the template's arucos ids as keys and the respective arucos
    points as entries
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
    Change the point of view of a image, the goal is to have the same one from the template
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
    #Filter matches using the Lowe's ratio test
    for m,n in matches:
        if m.distance < ratio_tresh*n.distance:
            good_matches.append(m)
    # To compute the function, we need a minimum of efficient matches
    if len(good_matches) > MIN_MATCH_COUNT:
        # Get the point from the match of the image and the template
        src_points = np.float32([key_1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_points = np.float32([key_2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        return src_points, dst_points
    else:#Not enough good matches
        return np.array([]), np.array([])

""""
***********************
Functions used for Task4
************************
"""
def compute_SIFT_img(image_1, image_2, detector, flann, MIN_MATCH_COUNT=7, ratio_tresh=0.7):
    """
    Change the point of view of a image, the goal is to have the same one from the template
    Use of the OpenCV library. 
    Input : image_1 : image to compute 	: type numpy array
            image_2 : the template 	: type numpy array
            MIN_MATCH_COUNT: Minimun number of good matches : type int
            des_2 : SIFT descriptors of the template : np.array
            key_2 : SIFT keys of the template : np.array
    """
    # Find the keys and the descriptors with SIFT
    key_1, des_1 = detector.detectAndCompute(image_1, None)
    key_2, des_2 =detector.detectAndCompute(image_2, None)
    # Find all the matches
    matches = flann.knnMatch(des_1, des_2, k = 2)
    # Take all the good matches
    good_matches = []
    #Filter matches using the Lowe's ratio test
    for m,n in matches:
        if m.distance < ratio_tresh*n.distance:
            good_matches.append(m)
    # To compute the function, we need a minimum of efficient matches
    if len(good_matches) > MIN_MATCH_COUNT:
        # Get the point from the match of the image and the template
        src_points = np.float32([key_1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_points = np.float32([key_2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        return src_points, dst_points
    else:#Not enough good matches
        return np.array([]), np.array([])

def calculate_depth_map(img1_undistorted,img2_undistorted):
    # ------------------------------------------------------------
    # CALCULATE DISPARITY (DEPTH MAP)
    # Adapted from: https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py
    # and: https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html
    
    # StereoSGBM Parameter explanations:
    # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html
    
    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 11
    min_disp = -128
    max_disp = 128
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 200
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 2
    disp12MaxDiff = 0
    
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        uniquenessRatio=uniquenessRatio,
        speckleWindowSize=speckleWindowSize,
        speckleRange=speckleRange,
        disp12MaxDiff=disp12MaxDiff,
        P1=8 * 1 * block_size * block_size,
        P2=32 * 1 * block_size * block_size,
    )
    disparity_SGBM = stereo.compute(img1_undistorted, img2_undistorted)
    
    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_SGBM = np.uint8(disparity_SGBM)
    cv2.imshow("Disparity", disparity_SGBM)
    #cv2.imwrite("disparity_SGBM_norm.png", disparity_SGBM)
    
def find_depth(circle_right, circle_left, frame_right, frame_left, baseline,f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        f_pixel = (width_right * 0.5) / np.tan(alpha * 0.5 * np.pi/180)

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = circle_right[0]
    x_left = circle_left[0]

    # CALCULATE THE DISPARITY:
    disparity = x_left-x_right      #Displacement between left and right frames [pixels]

    # CALCULATE DEPTH z:
    zDepth = (baseline*f_pixel)/disparity             #Depth in [cm]

    return abs(zDepth)
    


"""
***********************
Main
***********************
"""
task = int(sys.argv[1])
template_path = sys.argv[2]#path_to_template
output_path = sys.argv[3]#path_to_output_folder

#Run with for example: 1 Dataset/template2_fewArucos.png Output_Images Input_Images_Small
if task==1:
    input_images_path = sys.argv[4]
    img_template = cv2.imread(template_path)
    referenceArucos = getArucos(img_template)
    referenceCorners=getReferenceCorners(referenceArucos)
    input_images = os.listdir(input_images_path)
    for i in range(len(input_images)):
        img_name = input_images[i]
        frame = cv2.imread(input_images_path+"/"+img_name)
        arucos=getArucos(frame)
        if(len(arucos["ids"])>0):
            corners=getSourceCorners(arucos)
            destCorners= getDestCorners(arucos["ids"],referenceCorners)
            H= findHomography(corners, destCorners)
        rotated = cv2.warpPerspective(frame,H, (img_template.shape[1],img_template.shape[0]))
        cv2.imwrite(output_path+"/"+img_name,rotated)

#Run with for example: 2 Dataset/template2_fewArucos.png Output_Images Input_Images_Small
#Run with for example: 2 Dataset/formsns/templateSNS.jpg Output_Images Dataset/formsns/receitaSNS
#Run with for example: 2 Dataset/GoogleGlass/template_glass.jpg Output_Images Dataset/GoogleGlass/glass
#Run with for example: 2 Dataset/GoogleGlass/template_glass.jpg Output_Images Dataset/GoogleGlass/nexus
#Run with for example: 2 Dataset/Gehry/Template_Gehry.jpg Output_Images Dataset/Gehry/images
elif task==2:
    input_images_path = sys.argv[4]
    img_template = cv2.imread(template_path)
    input_images = os.listdir(input_images_path)
    detector = cv2.xfeatures2d.SIFT_create()# SIFT detector
    key_template, des_template = detector.detectAndCompute(img_template, None)
    #FLANN Matcher
    FLANN_INDEX_KDTREE = 0
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    #img_template = cv2.medianBlur(img_template,7)
    for i in range(len(input_images)):
        print(i)
        img_name = input_images[i]
        print(img_name)
        frame = cv2.imread(input_images_path+"/"+img_name, cv2.COLOR_BGR2GRAY)
        #frame_median = cv2.medianBlur(frame, 7) # Add median filter to image
        frame_median = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
        #Find dst_points and src_points using SIFT
        src_points, dst_points = compute_SIFT(frame_median, img_template, des_template, key_template, 
                                                      detector, flann)
        print("src: ",src_points.shape," ",src_points.dtype)
        print("dst: ",dst_points.shape," ",dst_points.dtype)
        if(len(dst_points)>0):
            # If the numeber of good matches is good enough, compute the homography
            # Compute the homography with the Ransac method
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5, maxIters=3000)
            if(H is not None):
                rotated = cv2.warpPerspective(frame, H, (img_template.shape[1], img_template.shape[0]))
                cv2.imwrite(output_path+"/"+img_name,rotated)
            else:
                print('Could not find homography matrix')
        else :
            # If the numeber of good matches is not good enough, do not compute the homography
            # Print an error message
            print('not enough good matches')

#Run with for example: 4 Dataset/TwoCameras/ulisboatemplate.jpg Output_Images Dataset/TwoCameras/ulisboa1/phone Dataset/TwoCameras/ulisboa1/photo
elif task == 4:
    img_template = cv2.imread(template_path)
    camera1_images_path = sys.argv[4]
    camera2_images_path = sys.argv[5]
    camera1_images = os.listdir(camera1_images_path)
    camera2_images = os.listdir(camera2_images_path)
    detector = cv2.xfeatures2d.SIFT_create()# SIFT detector
    key_template, des_template = detector.detectAndCompute(img_template, None)
    #FLANN Matcher
    FLANN_INDEX_KDTREE = 0
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    #Calibrate the camera
    for i in range(len(camera1_images)):
        img1_name = img2_name = camera1_images[i]
        img1= cv2.imread(camera1_images_path+"/"+img1_name)
        img2 = cv2.imread(camera2_images_path+"/"+img2_name)
        cv2.imshow('1',img1)
        cv2.imshow('2',img2)
        
        src_points1, dst_points1 = compute_SIFT(img1, img_template, des_template, key_template, 
                                                      detector, flann)
        src_points2, dst_points2 =compute_SIFT(img2, img_template, des_template, key_template, 
                                                      detector, flann)
        
        H1, mask1 = cv2.findHomography(src_points1, dst_points1, cv2.RANSAC, 5, maxIters=3000)
        H2, mask2 = cv2.findHomography(src_points1, dst_points1, cv2.RANSAC, 5, maxIters=3000)
        h1=img1.shape[0]
        w1 = img1.shape[1]
        h2=img2.shape[0]
        w2 = img2.shape[1]
        #img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
        #img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))
        img1_rectified = cv2.warpPerspective(img1, H1, (img_template.shape[1], img_template.shape[0]))
        img2_rectified = cv2.warpPerspective(img2, H2, (img_template.shape[1], img_template.shape[0]))
        cv2.imshow('r1',img1_rectified)
        cv2.imshow('r2',img2_rectified)
        calculate_depth_map(img1_rectified,img2_rectified)
        break
    
    


