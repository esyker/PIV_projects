# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 23:22:08 2022

@author: Utilizador
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
#input_images_path = 'Dataset/Gehry/images'
#img_name = 'rgb0077.jpg'
input_images_path = 'Dataset\GoogleGlass\glass'
img_name = 'rgb0931.jpg'

frame = cv2.imread(input_images_path+"/"+img_name)
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([frame],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()

cv2.imshow('original',frame)
"""
HSV
"""
min_HSV = np.array([0, 58, 30], dtype = "uint8")
max_HSV = np.array([33, 255, 255], dtype = "uint8")
# Get pointer to video frames from primary device
imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
skinRegionHSV = cv2.inRange(imageHSV, min_HSV, max_HSV)

skinHSV = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(skinRegionHSV))

cv2.imshow('HSV',skinHSV)
#cv2.imwrite("skin_detection.png", np.hstack([frame, skinHSV]))

"""
YCrCb
"""
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)

skinYCrCb = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(skinRegionYCrCb))

cv2.imshow('YCrCb',skinHSV)