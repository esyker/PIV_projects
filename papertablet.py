#Some references
#https://docs.google.com/document/d/e/2PACX-1vRF494drpSOiZ6In-KKg98eQ6iS4XdBwDFywW4j9SNZKmcJJ3-jRE-_86MVq6GUFg/pub
#https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
#https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html#:~:text=10%20%2Did%3D23-,Marker%20Detection,-Given%20an%20image
#https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
#https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
import cv2
import numpy as np
from getCorners import run 
from getCorners import get_arucos
import scipy.io


def getSortedCorners(arucos):
    _sorted = [None]*4
    for i in range(len(arucos["ids"])):
        _sorted[arucos["ids"][i][0]]=arucos["corners"][i]
    return _sorted

#run('Dataset/template1_manyArucos.png');
run('Dataset/template2_fewArucos.png');

reference_arucos = scipy.io.loadmat('cornersIds.mat')
reference_corners=getSortedCorners(reference_arucos)

input_video_path = './Dataset/FewArucos-Viewpoint1.mp4'

#Show video
cap = cv2.VideoCapture(input_video_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    arucos=get_arucos(frame)
    corners=getSortedCorners(arucos)
    print(corners)
    #print(corners["corners"])
    #print(frame, ret)
    if ret:
        frame_copy=frame.copy()
        concatenated = np.concatenate((frame,frame_copy),axis=1)
        concatenated=cv2.resize(concatenated,(960,540))
        cv2.imshow("original and homography",concatenated)
        k=cv2.waitKey(30) & 0xff
        #once you inter video will stop
        if k==27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
