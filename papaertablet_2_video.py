import cv2
import numpy as np
import time
import matplotlib.pyplot as plt


input_video_path = './Dataset/FewArucos-Viewpoint2.mp4'#'./Dataset/ManyArucos.mp4'#
template_path = 'Dataset/template2_fewArucos.png'#'Dataset/template1_manyArucos.png'


img_template = cv2.imread(template_path,0) 
print(img_template.shape)

sift = cv2.xfeatures2d.SIFT_create()

kp2, des2 = sift.detectAndCompute(img_template,None)

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 70)

flann = cv2.FlannBasedMatcher(index_params, search_params)

#Show video
cap = cv2.VideoCapture(input_video_path)
start_time = time.time()

while(cap.isOpened()):
    ret, frame_colored = cap.read()
    frame = cv2.cvtColor(frame_colored, cv2.COLOR_BGR2GRAY)
    if ret:
        kp1, des1 = sift.detectAndCompute(frame,None)
        matches = flann.knnMatch(des1,des2,k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.82*n.distance:
                good.append(m)
        
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        print(frame.shape)
        h,w = frame.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        img2 = cv2.polylines(img_template,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                            singlePointColor = None,
                            matchesMask = matchesMask, # draw only inliers
                            flags = 2)
        
        img3 = cv2.drawMatches(frame,kp1,img_template,kp2,good,None,**draw_params)
        
        plt.imshow(img3, 'gray'),plt.show()
        
        rotated = cv2.warpPerspective(frame,M, (img_template.shape[1],img_template.shape[0]))
        #resize
        frame = cv2.resize(frame,None,fx=0.2,fy=0.2)
        rotated = cv2.resize(rotated,None,fx=0.2,fy=0.2)
        cv2.imshow("original",frame)
        cv2.imshow("homography",rotated)
        
        k=cv2.waitKey(1) & 0xff
        #once enter key is pressed video will stop
        if k==27:
            break
    else:
        break

print("--- %s seconds ---" % (time.time() - start_time))

cap.release()
cv2.destroyAllWindows()











# h,w = img1.shape
# pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
# dst = cv2.perspectiveTransform(pts,M)

# img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
#                    singlePointColor = None,
#                    matchesMask = matchesMask, # draw only inliers
#                    flags = 2)

# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()
# # print(src_pts)
# # print(src_pts.shape)
# # print(dst_pts.shape)

# # H = findHomography(src_pts,dst_pts)

# # print(H)

# rotated = cv2.warpPerspective(img1,M, (img2.shape[1],img2.shape[0]))
# #resize
# frame = cv2.resize(img1,None,fx=0.2,fy=0.2)
# rotated = cv2.resize(rotated,None,fx=0.2,fy=0.2)

# # print(frame)

# cv2.imshow("original",frame)
# cv2.imshow("homography",rotated)
# cv2.waitKey(0)
# cv2.destroyAllWindows()






