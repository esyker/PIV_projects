import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def remove_background_gray(gray_frame):
    min_gray = 147
    max_gray = 178
    paperRegionGray = cv2.inRange(gray_frame, min_gray, max_gray)
    noBackGroundpaper = cv2.bitwise_and(gray_frame, gray_frame, mask = paperRegionGray)    
    return noBackGroundpaper

def gray_histogram(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='k')
    plt.figure()
    plt.show()
    
def rgb_histogram(img):
    for i, col in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color = col)
        plt.xlim([0, 256])
        plt.figure()
        plt.show()

img1 = cv2.imread('./Input_Images/frame1031.png')          # queryImage
img2 = cv2.imread('Dataset/template2_fewArucos.png') # trainImage
img3 = cv2.imread('Dataset/GoogleGlass/glass/rgb0987.jpg')#sleeve image
img10 = cv2.imread('Output_Images/rgb0987.jpg')

#gray_histogram(img1)
#rgb_histogram(img1)

#gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#removedBackground = remove_background_gray(gray_image)

#cv2.imshow('nobackground',removedBackground)

#REMOVE SKIN AND SLEEVE
# Attention: OpenCV uses BGR color ordering per default whereas
# Matplotlib assumes RGB color ordering!

#plt.figure(1)
#plt.imshow(cv2.cvtColor(img3, cv2.COLOR_BGR2HSV))
#plt.show()


def remove_skin_sleeve_hsv(frame):
    min_skin_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_skin_HSV = np.array([33, 255, 255], dtype = "uint8")
    min_sleeve_HSV = np.array([10,0,0],dtype="uint8")
    max_sleeve_HSV = np.array([120,255,80],dtype="uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_skin_HSV, max_skin_HSV)
    sleeveRegionHSV = cv2.inRange(imageHSV, min_sleeve_HSV, max_sleeve_HSV)
    noSkinHSV = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(cv2.bitwise_or(skinRegionHSV,sleeveRegionHSV)))    
    return noSkinHSV

img4 = remove_skin_sleeve_hsv(img10)
plt.figure(1)
plt.imshow(img4)
plt.show()
