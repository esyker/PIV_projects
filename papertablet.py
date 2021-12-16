#Some references
#https://docs.google.com/document/d/e/2PACX-1vRF494drpSOiZ6In-KKg98eQ6iS4XdBwDFywW4j9SNZKmcJJ3-jRE-_86MVq6GUFg/pub
#https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
#https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html#:~:text=10%20%2Did%3D23-,Marker%20Detection,-Given%20an%20image
#https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
#https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
import cv2
import numpy as np
from getCorners import run 
from getCorners import getArucos
import scipy.io
import time
import matplotlib.pyplot as plt

def findHomography(sourcePoints, destPoints):
    A=[]
    for i in range(16):
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
    
def getSourceCorners(arucos):
    sourceCorners = []
    numb_arucos = len(arucos["corners"])
    for i in range(numb_arucos):
        for j in range(arucos["corners"][i][0].shape[0]):
            sourceCorners.append(np.array(arucos["corners"][i][0][j]))
    return np.array(sourceCorners)

def getDestCorners(sourceIDs,referenceCorners):
    destCorners=[]
    numb_arucos=0
    for _id in sourceIDs:
        if numb_arucos==4:
            break
        #print(_id)
        numb_arucos+=1
        corner = referenceCorners[_id[0]]
        for point in corner: 
            destCorners.append(point)
    return np.array(destCorners)

def getReferenceCorners(referenceArucos):
    referenceCorners={}
    numbArucos=len(referenceArucos["corners"])
    for i in range(numbArucos):
        referenceCorners[referenceArucos["ids"][i][0]] = [corner for corner 
                                                          in referenceArucos["corners"][i][0]]
    return referenceCorners

template = cv2.cvtColor(cv2.imread('Dataset/template2_fewArucos.png'), cv2.COLOR_BGR2GRAY)
#template = cv2.imread('Dataset/template2_fewArucos.png')

rect_template = np.array([[[0,0],
                          [template.shape[1],0],
                          [template.shape[1],template.shape[0]],
                           [0,template.shape[0]]]],dtype="float32")
#run('Dataset/template1_manyArucos.png');
run('Dataset/template2_fewArucos.png')

referenceArucos = scipy.io.loadmat('cornersIds.mat')
referenceCorners=getReferenceCorners(referenceArucos)

input_video_path = './Dataset/FewArucos-Viewpoint2.mp4'

#Show video
cap = cv2.VideoCapture(input_video_path)
start_time = time.time()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        arucos=getArucos(frame)
        if(len(arucos["ids"])>0):
            corners=getSourceCorners(arucos)
            destCorners= getDestCorners(arucos["ids"],referenceCorners)
            print("Corners",corners)
            print("DestCorners",destCorners)
            #M,mask = cv2.findHomography(corners, destCorners)#, method=cv2.RANSAC,ransacReprojThreshold=5.0)
            H= findHomography(destCorners, corners)
            #M= findHomography(corners, destCorners)
            #rotated=cv2.warpPerspective(frame,M,(frame.shape[1],frame.shape[0]))
            #rotated = cv2.perspectiveTransform(frame,M,(frame.shape[1],frame.shape[0]))
            rect_frame = cv2.perspectiveTransform(rect_template, H,(gray.shape[1],gray.shape[0]))
            print("H",H)
        M= cv2.getPerspectiveTransform(rect_frame,rect_template)
        print("M",M)
        rotated = cv2.warpPerspective(frame,M, (template.shape[1],template.shape[0]))
        #print(M)
        #
        """
        frame=cv2.circle(frame, (int(corners[0][0]),int(corners[0][1])), 
                   radius=10, color=(0, 0, 255), thickness=10)#red
        rotated=cv2.circle(rotated, (int(destCorners[0][0]),int(destCorners[0][1])), 
                   radius=10, color=(255, 0, 0), thickness=10)#blue
        homography_point = np.matmul(M,np.array([corners[0][0],corners[0][1],1]))
        rotated=cv2.circle(rotated, (int(homography_point[0]),int(homography_point[1])), 
                   radius=10, color=(0, 255, 0), thickness=10)#green
        """
        print(rotated.shape)
        #concatenated = np.concatenate((frame,rotated),axis=1)
        #concatenated=cv2.resize(concatenated,(960,540))
        #cv2.imshow("original and homography",concatenated)
        im_out=cv2.resize(rotated,None,fx=0.2,fy=0.2)
        cv2.imshow("original and homo",im_out)
        k=cv2.waitKey(30) & 0xff
        #once you inter video will stop
        if k==27:
            break
    else:
        break

print("--- %s seconds ---" % (time.time() - start_time))

cap.release()
cv2.destroyAllWindows()


#check if homography is correct
plt.figure()
plt.imshow(frame)
original_point=np.array([1532,453,1])
plt.scatter(original_point[0], original_point[1], s=3, c='red', marker='o')
plt.figure()
plt.imshow(rotated)
rotated_point = np.matmul(M,original_point)
plt.scatter(rotated_point[0], rotated_point[1], s=3, c='red', marker='o')