import cv2
import numpy as np
import os
import argparse

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
    #h=h/h[2][2]#normalize matrix
    return h

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

img_template = cv2.imread(template_path)

rect_template = np.array([[[0,0],
                          [img_template.shape[1],0],
                          [img_template.shape[1],img_template.shape[0]],
                           [0,img_template.shape[0]]]],dtype="float32")

referenceArucos = getArucos(img_template)
referenceCorners=getReferenceCorners(referenceArucos)

max_numb_images =100

if task==1:
    input_images = os.listdir(input_images_path)
    for i in range(len(input_images)):
        img_name = input_images[i]
        frame = cv2.imread(input_images_path+"/"+img_name)
        arucos=getArucos(frame)
        if(len(arucos["ids"])>0):
            corners=getSourceCorners(arucos)
            destCorners= getDestCorners(arucos["ids"],referenceCorners)
            H= findHomography(corners, destCorners)
            #H= findHomography(destCorners, corners)
            #rect_frame = cv2.perspectiveTransform(rect_template, H,(frame.shape[1],frame.shape[0]))
        #M= cv2.getPerspectiveTransform(rect_frame,rect_template)
        #rotated = cv2.warpPerspective(frame,M, (img_template.shape[1],img_template.shape[0]))
        rotated = cv2.warpPerspective(frame,H, (img_template.shape[1],img_template.shape[0]))
        cv2.imwrite(output_path+"/"+img_name,rotated)
        



