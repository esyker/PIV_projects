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

def getSourceCorners(arucos):
    sourceCorners = []
    numb_arucos = len(arucos["corners"])
    for i in range(numb_arucos):
        for j in range(arucos["corners"][i][0].shape[0]):
            sourceCorners.append(np.array(arucos["corners"][i][0][j]))
    return np.array(sourceCorners)

def getDestCorners(sourceIDs,referenceCorners):
    destCorners=[]
    for _id in sourceIDs:
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


#run('Dataset/template1_manyArucos.png');
run('Dataset/template2_fewArucos.png')

referenceArucos = scipy.io.loadmat('cornersIds.mat')
referenceCorners=getReferenceCorners(referenceArucos)

input_video_path = './Dataset/FewArucos-Viewpoint1.mp4'

#Show video
cap = cv2.VideoCapture(input_video_path)
start_time = time.time()

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        arucos=getArucos(frame)
        corners=getSourceCorners(arucos)
        destCorners= getDestCorners(arucos["ids"],referenceCorners)
        M,mask = cv2.findHomography(corners, destCorners)
        rotated=cv2.warpPerspective(frame,M,(frame.shape[1],frame.shape[0]))
        #print(M)
        concatenated = np.concatenate((frame,rotated),axis=1)
        concatenated=cv2.resize(concatenated,(960,540))
        cv2.imshow("original and homography",concatenated)
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
original_point=np.array([1254,302,1])
plt.scatter(original_point[0], original_point[1], s=5, c='red', marker='o')
plt.figure()
plt.imshow(rotated)
rotated_point = np.matmul(M,original_point)
plt.scatter(rotated_point[0], rotated_point[1], s=5, c='red', marker='o')