import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


img1 = cv2.imread('./Input_Images/frame1031.png',0)          # queryImage
img2 = cv2.imread('Dataset/template2_fewArucos.png',0) # trainImage

print(img1.shape)
print(img2.shape)

sift = cv2.xfeatures2d.SIFT_create()


kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#print(des1)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1,des2,k=2)

#print(matches)

good = []
for m,n in matches:
    if m.distance < 0.825*n.distance:
        good.append(m)

src_pts = np.array([ kp1[m.queryIdx].pt for m in good ])
dst_pts = np.array([ kp2[m.trainIdx].pt for m in good ])
#print(src_pts)

plt.figure()
plt.scatter(src_pts[:, 0], src_pts[:, 1])
plt.show()


M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

h,w = img1.shape
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) #PAS COMPRIS
print(pts.shape)
dst = cv2.perspectiveTransform(pts,M)

img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

draw_params = dict(matchColor = (255,0,255), # draw matches in green pink
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)


img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()


rotated = cv2.warpPerspective(img1,M, (img2.shape[1],img2.shape[0]))

#resize
frame = cv2.resize(img1,None,fx=0.2,fy=0.2)
original = cv2.resize(img2,None,fx=0.2,fy=0.2)
rotated = cv2.resize(rotated,None,fx=0.2,fy=0.2)

previous_valid_rotated = np.copy(original)
valid_rotated = np.copy(original)

# print(frame)

#------------------------------------------------- blur + mask section ---------------------------------------------------

pixel_value = 150


#original
rotated_bool = np.copy(rotated)
rotated_bool[rotated_bool <= pixel_value] = 0
rotated_bool[rotated_bool > pixel_value] = 255
     


#rotated
original_bool= np.copy(previous_valid_rotated)
original_bool[original_bool <= pixel_value] = 0
original_bool[original_bool > pixel_value] = 255
        


blur_rotated = cv2.GaussianBlur(rotated_bool,(29,29),0)
print("blur_rotated is ", blur_rotated)
blur_original = cv2.GaussianBlur(original_bool,(29,29),0)
print("blur_original is ", blur_original)



#------------------------------------------------- compute MSE + output definition -------------------------------------------------------

err = np.subtract(blur_rotated, blur_original)
print("err is : ", err)
squared_err = np.square(err)
mse = squared_err.mean()
print("MSE is :", mse)

if (mse < 24):
    previous_valid_rotated = rotated
    valid_rotated = rotated
    
else: 
    valid_rotated = previous_valid_rotated

#--------------------------------------------------- show images -----------------------------------------------------------
cv2.imshow("frame",frame)
cv2.imshow("original",original)
cv2.imshow("homography",rotated)
cv2.imshow("blur rotated", blur_rotated)
cv2.imshow("blur original", blur_original)
cv2.imshow("previous_valid_rotated", previous_valid_rotated)
cv2.imshow("valid_rotated", valid_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()




def compute_mse(img1,img2):
    err = np.subtract(img1, img2)
    squared_err = np.square(err)
    mse = squared_err.mean()
    return mse


