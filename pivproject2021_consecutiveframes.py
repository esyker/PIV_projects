import cv2
import numpy as np
import os
import argparse
import re

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

def compute_SIFT(image_1, image_2, des_2, key_2, detector, flann, MIN_MATCH_COUNT=7, ratio_tresh=0.82):
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
    print("GM:",len(good_matches))
    # To compute the function, we need a minimum of efficient matches
    if len(good_matches) > MIN_MATCH_COUNT:
        # Get the point from the match of the image and the template
        src_points = np.float32([key_1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        dst_points = np.float32([key_2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
        return src_points, dst_points, len(good_matches)
    else:#Not enough good matches
        return [], [], 0

parser = argparse.ArgumentParser()
parser.add_argument('task', type = int, choices= [1,2,3,4],
                    help="Task type")
parser.add_argument('path_to_template', type=str, help="Path to template.")
parser.add_argument('path_to_output_folder', type = str, help="Path to output folder")
parser.add_argument('arg1', type = str, help= "if task = 1,2,3 , arg1 is the path to the image input folder. "+
                    "if task=4,  arg1 is the path to images of camera 1 (all rgb images)")
opt = parser.parse_args()

task = opt.task
input_images_path = opt.arg1#'./Dataset/FewArucos-Viewpoint2.mp4'
template_path = opt.path_to_template#'Dataset/template2_fewArucos.png'
output_path = opt.path_to_output_folder

#Run with for example: 1 Dataset/template2_fewArucos.png Output_Images Input_Images_Small
if task==1:
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
elif task==2:
    img_template = cv2.imread(template_path)
    input_images = sorted(os.listdir(input_images_path), key=lambda f: int(re.sub('\D', '', f)))
    #SIFT Detector
    detector = cv2.xfeatures2d.SIFT_create()# SIFT detector
    key_template, des_template = detector.detectAndCompute(img_template, None)
    #FLANN Matcher
    FLANN_INDEX_KDTREE = 0
    index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_parameters = dict(checks = 70)
    flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
    max_good_matches = 0
    best_homography_matrix = None
    for i in range(len(input_images)):
        img_name = input_images[i]
        #if(i==0):
        #    img_name='frame1.png'
        #else:
        #    img_name = 'frame123.png'
        frame = cv2.imread(input_images_path+"/"+img_name, cv2.COLOR_BGR2GRAY)
        #Find dst_points and src_points using SIFT
        print("Image",i)
        src_points, dst_points,good_matches = compute_SIFT(frame, img_template, des_template, key_template, 
                                                      detector, flann)
        if(good_matches>=0.6*max_good_matches):
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 40, maxIters=3000)
            rotated = cv2.warpPerspective(frame, H, (img_template.shape[1], img_template.shape[0]))
            best_homography_matrix=H
            max_good_matches=good_matches
        else:
            pre_rotated = cv2.warpPerspective(frame, best_homography_matrix, 
                                              (img_template.shape[1], img_template.shape[0]))
            #cv2.imshow('pre_rotated',pre_rotated)
            src_points, dst_points,good_matches = compute_SIFT(pre_rotated, img_template, des_template, key_template, 
                                                          detector, flann)
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 40, maxIters=3000)
            rotated = cv2.warpPerspective(frame, H, (img_template.shape[1], img_template.shape[0]))         
        cv2.imwrite(output_path+"/"+img_name,rotated)
        #if(i==1):
        #    break

        



