import cv2
import time 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.io

image1_path = "Dataset/teste/rgb_0003.jpg"
image2_path = "Dataset/teste/rgb_0010.jpg"

calibration_path = "Dataset/calib_asus.mat"

image1 = cv2.imread(image1_path, 0)
image2 = cv2.imread(image2_path, 0)

calib = scipy.io.loadmat(calibration_path)

#print(calib)

#plt.subplot(2,1,1), plt.imshow(image1),
#plt.subplot(2,1,2), plt.imshow(image2), plt.show()

minDisparity = 0
numDisparities = 64
blockSize = 8
disp12MaxDiff = 1
uniquenessRatio = 10
speckleWindowSize = 10
speckleRange = 8

stereo = cv2.StereoSGBM_create(minDisparity = minDisparity,
                            numDisparities = numDisparities, 
                            blockSize = blockSize,
                            disp12MaxDiff = disp12MaxDiff,
                            uniquenessRatio = uniquenessRatio,
                            speckleWindowSize = speckleWindowSize,
                            speckleRange = speckleRange)

stereo_1 = cv2.StereoBM_create(numDisparities=48, blockSize=25)                       

disparity = stereo.compute(image1, image2)
print(disparity)
disp = cv2.normalize(disparity, 0,255,cv2.NORM_MINMAX)

plt.imshow(disp), plt.show()

h,w = image1.shape[:2]

focal_length = 1

Q = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
    [0,0,0,1]])