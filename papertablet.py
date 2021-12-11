#Some references
#https://docs.google.com/document/d/e/2PACX-1vRF494drpSOiZ6In-KKg98eQ6iS4XdBwDFywW4j9SNZKmcJJ3-jRE-_86MVq6GUFg/pub
#https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html
#https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html#:~:text=10%20%2Did%3D23-,Marker%20Detection,-Given%20an%20image
#https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
#https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
import cv2
import numpy as np
from getCorners import run 
import scipy.io

run('Dataset/template1_manyArucos.png');
run('Dataset/template2_fewArucos.png');

mat = scipy.io.loadmat('cornersIds.mat')

input_video_path = './Dataset/FewArucos-Viewpoint1.mp4'

"""
# capture video
cap = cv2.VideoCapture(0)
#descripe a loop
#read video frame by frame
while True:
    ret,img = cap.read()
    cv2.imshow('Original Video',img)
    #flip for truning(fliping) frames of video
    img2=cv2.flip(img,-1)
    cv2.imshow('Flipped video',img2)
    k=cv2.waitKey(30) & 0xff
    #once you inter Esc capturing will stop
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
"""

#Show video
cap = cv2.VideoCapture(input_video_path)

while(cap.isOpened()):
    ret, frame = cap.read()
    print(frame, ret)
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
