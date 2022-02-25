import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
img5 = mpimg.imread('Dataset/TwoCameras/ulisboa1/photo/rgb_0001.jpg')
img6 = cv2.imread('Dataset/TwoCameras/ulisboa1/photo/rgb_0001.jpg')

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


def remove_skin_yrcbr(frame):
    min_YCrCb = np.array([0,133,77],np.uint8)
    max_YCrCb = np.array([235,173,127],np.uint8)
    imageYCrCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    skinRegionYCrCb = cv2.inRange(imageYCrCb,min_YCrCb,max_YCrCb)
    skinYCrCb = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(skinRegionYCrCb))
    return skinYCrCb

def remove_skin_sleeve_hsv(frame):
    min_skin_HSV = np.array([0, 58, 30], dtype = "uint8")
    max_skin_HSV = np.array([33, 255, 255], dtype = "uint8")
    min_sleeve_HSV = np.array([10,0,0],dtype="uint8")
    max_sleeve_HSV = np.array([120,255,80],dtype="uint8")
    imageHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_skin_HSV, max_skin_HSV)
    closedNoSkin = cv2.morphologyEx(cv2.bitwise_not(skinRegionHSV), cv2.MORPH_OPEN, np.ones((9,9),np.uint8))
    sleeveRegionHSV = cv2.inRange(imageHSV, min_sleeve_HSV, max_sleeve_HSV)
    noSleeveNoSkinHSV = cv2.bitwise_and(closedNoSkin, closedNoSkin)#, mask = cv2.bitwise_not(sleeveRegionHSV))    
    return noSleeveNoSkinHSV

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
    res = np.zeros((h, w, 1), dtype=np.uint8)
    # Only 'skin pixels' will be set to white (255) in the res image:
    for y in range(0, h):
        for x in range(0, w):
            (b, g, r) = bgr_image[y, x]
            if bgr_skin(b, g, r):
                res[y, x] = 1
    return res

def remove_skin_rgb(bgr_image):
    mask = skin_rgb_mask(bgr_image)
    noSkinRGB = cv2.bitwise_and(bgr_image, bgr_image, mask = mask)
    return noSkinRGB, mask

def new_filter(frame):
    blur = cv2.GaussianBlur(frame,(7,7),1,1)
    min_skin_HSV = np.array([5, 38, 51], dtype = "uint8")
    max_skin_HSV = np.array([17, 250, 242], dtype = "uint8")
    imageHSV = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    skinRegionHSV = cv2.inRange(imageHSV, min_skin_HSV, max_skin_HSV)
    min_gray = 60 
    max_gray = 255
    imageGray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    skinRegionGray = cv2.inRange(imageGray, min_gray, max_gray)
    eroded = cv2.morphologyEx(skinRegionGray, cv2.MORPH_ERODE, np.ones((3,3),np.uint8))
    opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((9,9),np.uint8))
    final = cv2.bitwise_and(frame, frame, mask = cv2.bitwise_not(cv2.bitwise_or(closed,skinRegionGray,skinRegionHSV)))
    median = cv2.medianBlur(final, 15)
    return median
    

img4 = new_filter(img6)
plt.figure(1)
plt.imshow(img4)
plt.show()
#plt.figure(2)
#plt.imshow(cv2.cvtColor(img5,cv2.COLOR_BGR2HSV))
#plt.show()