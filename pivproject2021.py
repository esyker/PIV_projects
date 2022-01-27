import cv2
import numpy as np
import sys
import os
import math

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
"""
def remove_skin_sleeve_hsv(frame):
    min_skin_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_skin_HSV = np.array([33, 255, 255], dtype = "uint8")
    min_sleeve_HSV = np.array([10,0,0],dtype="uint8")
    max_sleeve_HSV = np.array([120,255,120],dtype="uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_skin_HSV, max_skin_HSV)
    sleeveRegionHSV = cv2.inRange(imageHSV, min_sleeve_HSV, max_sleeve_HSV)
    noSkinHSV = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(cv2.bitwise_or(skinRegionHSV,sleeveRegionHSV)))    
    return noSkinHSV
"""
def remove_skin_sleeve_hsv(frame):
    min_skin_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_skin_HSV = np.array([33, 255, 255], dtype = "uint8")
    min_sleeve_HSV = np.array([10,0,0],dtype="uint8")
    max_sleeve_HSV = np.array([120,255,80],dtype="uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_skin_HSV, max_skin_HSV)
    closedSkin = cv2.morphologyEx(skinRegionHSV, cv2.MORPH_CLOSE, np.ones((121,121),np.uint8))
    sleeveRegionHSV = cv2.inRange(imageHSV, min_sleeve_HSV, max_sleeve_HSV)
    noSleeveNoSkinHSV = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(cv2.bitwise_or(closedSkin,
                                                                                            sleeveRegionHSV)))
    return noSleeveNoSkinHSV


def check_homography(H, img, scale_factor):
    det2 = H[0,0] * H[1,1] - H[0,1] * H[1,0]
    if (det2 <= 0.1):
        return False
    det3 = np.linalg.det(H)
    if (det3 <= 0.1):
        return False
    h,w,d = img.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, H)
    [[x1, y1]], [[x2, y2]], [[x3, y3]], [[x4, y4]] = dst
    og = img.shape[0] * img.shape[1]
    area = (x1*y2 + x2*y3 + x3*y4 + x4*y1) - (y1*x2 + y2*x3 + y3*x4 + y4*x1)
    area = abs(area * 0.5)
    ratio = area / og
    
    MAX_SCALE = 4
    ratio_len = np.sqrt(ratio)
    det2_len = np.sqrt(det2)
    det3_len = np.cbrt(det3)
    print(ratio_len, det2_len, det3_len)
    if (ratio_len < (scale_factor/MAX_SCALE)) or (ratio_len > (scale_factor*MAX_SCALE)):
        return False
    if (det2_len < (scale_factor/MAX_SCALE)) or (det2_len > (scale_factor*MAX_SCALE)):
        return False
    if (det3_len < (scale_factor/MAX_SCALE)) or (det3_len > (scale_factor*MAX_SCALE)):
        return False
    return True

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
    
    detector = cv2.SIFT_create()# SIFT detector
    key_template, des_template = detector.detectAndCompute(img_template, None)
    
    #FLANN Matcher
    FLANN_INDEX_KDTREE = 0
    RATIO_TRESH = 0.85
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    
    H_prev = None

    START = 0
    scale_factor = img_template.shape[0] / cv2.imread(input_images_path+"/"+input_images[0]).shape[0]
    reproj_thresh = int(15 * scale_factor)
    
    for i in range(START, len(input_images)):
        img_name = input_images[i]
        print("\n"+img_name)
        frame = cv2.imread(input_images_path+"/"+img_name)
        #Find dst_points and src_points using SIFT
        src_points, dst_points = compute_SIFT(frame, img_template, 
                                              des_template, key_template, 
                                              detector, flann, 
                                              ratio_tresh=RATIO_TRESH)

        POINTS_THRESH = 4
        MAX_ITERS = 200
        
        if (len(dst_points)>POINTS_THRESH):
            # If the numeber of good matches is good enough, compute the homography
            # Compute the homography with the Ransac method
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 
                                         reproj_thresh, maxIters=MAX_ITERS)
            good = False
            if (H is not None):
                good = check_homography(H, frame, scale_factor)
            if (H is None) or (not good):
                print('### No H')
                H = H_prev
        else :
            # If the numeber of good matches is not good enough, do not compute the homography
            # Print an error message
            print('### No matches')
            H = H_prev
    
        if (H is None):
            continue
        rotated = cv2.warpPerspective(frame, H, (img_template.shape[1], 
                                                 img_template.shape[0]))
        cv2.imwrite(output_path+"/"+img_name, rotated)
        H_prev = H

#Run with for example: 4 Dataset/TwoCameras/ulisboatemplate.jpg Output_Images Dataset/TwoCameras/ulisboa1/phone Dataset/TwoCameras/ulisboa1/photo
#Run with for example: 4 Dataset/GoogleGlass/template_glass.jpg Output_Images Dataset/GoogleGlass/nexus Dataset/GoogleGlass/glass
elif task == 4:
    img_template = cv2.imread(template_path)
    camera1_images_path = sys.argv[4]
    camera2_images_path = sys.argv[5]
    camera1_images = os.listdir(camera1_images_path)
    camera2_images = os.listdir(camera2_images_path)
    
    detector = cv2.SIFT_create()# SIFT detector
    key_template, des_template = detector.detectAndCompute(img_template, None)
    
    #FLANN Matcher
    FLANN_INDEX_KDTREE = 0
    RATIO_TRESH = 0.85
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    
    prev_frame = img_template
    
    START = 225
    
    data1_sample =  cv2.imread(camera1_images_path+"/"+camera1_images[0])
    scale_factor1 = img_template.shape[0] / data1_sample.shape[0]
    reproj_thresh1 = int(15 * scale_factor1)
    data2_sample = cv2.imread(camera2_images_path+"/"+camera2_images[0])
    scale_factor2 = img_template.shape[0] / data2_sample.shape[0]
    reproj_thresh2 = int(15 * scale_factor2)
    
    for i in range(START, len(camera1_images)):
        rotated1 = None
        rotated2 = None
        img1_name = camera1_images[i]
        img2_name = camera2_images[i*2 + 1]
        print(img1_name)
        img1 = cv2.imread(camera1_images_path+"/"+img1_name)
        img2 = cv2.imread(camera2_images_path+"/"+img2_name)

        img1_noskin = remove_skin_sleeve_hsv(img1)
        img2_noskin = remove_skin_sleeve_hsv(img2)

        src_points1, dst_points1 = compute_SIFT(img1, img_template, 
                                                des_template, key_template, 
                                                detector, flann, 
                                                ratio_tresh=RATIO_TRESH)
        src_points2, dst_points2 = compute_SIFT(img2, img_template, 
                                                des_template, key_template, 
                                                detector, flann, 
                                                ratio_tresh=RATIO_TRESH)
        
        POINTS_THRESH = 4
        MAX_ITERS = 200
        
        if(len(dst_points1)>POINTS_THRESH):# If the numeber of good matches is good enough
            H1, mask1 = cv2.findHomography(src_points1, dst_points1, cv2.RANSAC, 
                                           reproj_thresh1, maxIters=MAX_ITERS)
            good1 = False
            if (H1 is not None):
                good1 = check_homography(H1, img1, scale_factor1)
            if (H1 is not None) and good1:
                rotated1 = cv2.warpPerspective(img1, H1, (img_template.shape[1], 
                                                          img_template.shape[0]))
            else:
                print('No H1')
        else:
            print('No matches 1')
            
            
        if(len(dst_points2)>POINTS_THRESH):
            H2, mask2 = cv2.findHomography(src_points2, dst_points2, cv2.RANSAC, 
                                           reproj_thresh2, maxIters=MAX_ITERS)
            good2 = False
            if (H2 is not None):
                good2 = check_homography(H2, img2, scale_factor2)
            if (H2 is not None) and good2:
                rotated2 = cv2.warpPerspective(img2, H2, (img_template.shape[1], 
                                                          img_template.shape[0]))
            else:
                print('No H2')
        else:
            print('No matches 2')
            
        curr = None
        
        if rotated1 is not None:
            img1_noskin = remove_skin_sleeve_hsv(rotated1)
            curr = img1_noskin
        if rotated2 is not None:
            img2_noskin = remove_skin_sleeve_hsv(rotated2)
            if curr is not None:
                gray1 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
                mask1 = cv2.threshold(gray1, 0, 255, cv2.THRESH_BINARY_INV)[1]
                mask1_inv = cv2.bitwise_not(mask1)
                curr1_bg = cv2.bitwise_and(curr, curr, mask=mask1_inv)
                img2_fg = cv2.bitwise_and(img2_noskin, img2_noskin, mask=mask1)
                curr = cv2.add(curr1_bg, img2_fg)
            else:
                curr = img2_noskin
        if curr is not None:
            gray2 = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
            mask2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY_INV)[1]
            mask2_inv = cv2.bitwise_not(mask2)
            curr2_bg = cv2.bitwise_and(curr, curr, mask=mask2_inv)
            prev_fg = cv2.bitwise_and(prev_frame, prev_frame, mask=mask2)
            curr = cv2.add(curr2_bg, prev_fg)
        else:
            curr = prev_frame
        
        result = curr
        #result = cv2.addWeighted(curr, 0.5, prev_frame, 0.5, 0)
        cv2.imwrite(output_path+"/"+img1_name, result)
        #prev_frame = curr
    
    
    
    
        