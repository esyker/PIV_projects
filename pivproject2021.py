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
def remove_skin_hsv(frame):
    min_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_HSV = np.array([33, 255, 255], dtype = "uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    noSkinHSV = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(skinRegionHSV))    
    return noSkinHSV

# Values are taken from: 'RGB-H-CbCr Skin Colour Model for Human Face Detection'
# (R > 95) AND (G > 40) AND (B > 20) AND (max{R, G, B} − min{R, G, B} > 15) AND (|R − G| > 15) AND (R > G) AND (R > B)
# (R > 220) AND (G > 210) AND (B > 170) AND (|R − G| ≤ 15) AND (R > B) AND (G > B)
def bgr_skin(b, g, r):
    """Rule for skin pixel segmentation based on the paper 'RGB-H-CbCr Skin Colour Model for Human Face Detection'"""
    e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (
    abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))
    return e1 or e2

# Skin detector based on the BGR color space
def skin_rgb_mask(bgr_image):
    """Skin segmentation based on the RGB color space"""
    h = bgr_image.shape[0]
    w = bgr_image.shape[1]
    # We crete the result image with back background
    res = np.zeros((h, w, 1), dtype="uint8")
    # Only 'skin pixels' will be set to white (255) in the res image:
    for y in range(0, h):
        for x in range(0, w):
            (b, g, r) = bgr_image[y, x]
            if bgr_skin(b, g, r):
                res[y, x] = 1
    return res

def remove_skin_rgb(bgr_image):
    mask = skin_rgb_mask(bgr_image)
    noSkinRGB = cv2.bitwise_and(bgr_image, bgr_image, mask = cv2.bitwise_not(mask))
    return noSkinRGB


def compute_mse(points1,points2):
    err = np.subtract(points1, points2)
    squared_err = np.square(err)
    mse = squared_err.mean()
    return mse

def apply_mask(points,mask):
    filtered_points = []
    for i in range(len(points)):
        if(mask[i][0]==1):
            filtered_points.append(points[i])
    return np.array(filtered_points)

def estimate_good_homography(points1, mask1, points2, mask2, MSETRESH=1750000,GOOD_POINTS_TRESH=50):
    good_points1=apply_mask(points1,mask1)
    good_points2=apply_mask(points2, mask2)
    numb_good_points=len(good_points1)
    mse=compute_mse(good_points1,good_points2)
    if(mse>MSETRESH or numb_good_points<GOOD_POINTS_TRESH):
        return False
    else:
        return True

"""
def remove_skin2(frame):
    min_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_HSV = np.array([33, 255, 255], dtype = "uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    imageYCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)
    noSkinHSV = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(skinRegionHSV))    
    return noSkinHSV

#def get_mask(rgbImage,hsvImage, ycrcbImage):
#    for
"""



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
        #frame_filtered = cv2.medianBlur(frame, 7) # Add median filter to image
        frame_filtered = cv2.GaussianBlur(frame,(3,3),cv2.BORDER_DEFAULT)
        #Find dst_points and src_points using SIFT
        src_points, dst_points = compute_SIFT(frame_filtered, img_template, des_template, key_template, 
                                                      detector, flann, ratio_tresh= 0.82)
        print("src: ",src_points.shape," ",src_points.dtype)
        print("dst: ",dst_points.shape," ",dst_points.dtype)
        if(len(dst_points)>0):
            # If the numeber of good matches is good enough, compute the homography
            # Compute the homography with the Ransac method
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 40, maxIters=3000)
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
#Run with for example: 4 Dataset/GoogleGlass/template_glass.jpg Output_Images Dataset/GoogleGlass/nexus Dataset/GoogleGlass/glass
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
    for i in range(len(camera1_images)):
        img1_name = img2_name = camera1_images[i]
        print(img1_name)
        img1= cv2.imread(camera1_images_path+"/"+img1_name)
        img2 = cv2.imread(camera2_images_path+"/"+img2_name)
        #cv2.imshow('orginal1',img1)
        #cv2.imshow('original2',img2)
        img1_noskin = remove_skin_hsv(img1)
        img2_noskin = remove_skin_hsv(img2)
        #img1_noskin = remove_skin_rgb(img1)
        #img2_noskin = remove_skin_rgb(img2)
        #cv2.imshow('noskin1',img1_noskin)
        #cv2.imshow('noskin2',img2_noskin)
        src_points1, dst_points1 = compute_SIFT(img1_noskin, img_template, des_template, key_template, 
                                                      detector, flann, ratio_tresh= 0.82)
        src_points2, dst_points2 = compute_SIFT(img2_noskin, img_template, des_template, key_template, 
                                                      detector, flann, ratio_tresh= 0.82)
        if(len(dst_points1)>70):# If the numeber of good matches is good enough
            H1, mask1 = cv2.findHomography(src_points1, dst_points1, cv2.RANSAC, 40, maxIters=3000)
            if(H1 is not None):
                rotated1 = cv2.warpPerspective(img1_noskin, H1, (img_template.shape[1], img_template.shape[0]))
                cv2.imwrite(output_path+"/"+"1"+img1_name,rotated1)
                print('1 MSE:',estimate_good_homography(src_points1, mask1, dst_points1, mask1), " gmatches:",len(dst_points1))
        if(len(dst_points2)>70):
            H2, mask2 = cv2.findHomography(src_points2, dst_points2, cv2.RANSAC, 40, maxIters=3000)
            if(H2 is not None):
                rotated2 = cv2.warpPerspective(img2_noskin, H2, (img_template.shape[1], img_template.shape[0]))
                cv2.imwrite(output_path+"/"+"2"+img2_name,rotated2)
                print('2',estimate_good_homography(src_points2, mask2, dst_points2, mask2), " gmatches:",len(dst_points2),"\n")
            else:
                print('Could not find homography matrix')
        #blend the two images https://docs.opencv.org/4.x/d5/dc4/tutorial_adding_images.html
        result = cv2.addWeighted(rotated1, 0.5, img_template, 0.5, 0)
        cv2.imwrite(output_path+"/"+img2_name,result)
        