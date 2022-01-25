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

gray_histogram(img1)
rgb_histogram(img1)

gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

removedBackground = remove_background_gray(gray_image)

cv2.imshow('nobackground',removedBackground)