img_template = cv2.imread(template_path)
camera1_images_path = sys.argv[4]
camera2_images_path = sys.argv[5]
camera1_images = os.listdir(camera1_images_path)
camera2_images = os.listdir(camera2_images_path)
detector = cv2.xfeatures2d.SIFT_create()# SIFT detector
key_template, des_template = detector.detectAndCompute(img_template, None)
#FLANN Matcher
FLANN_INDEX_KDTREE = 0
index_parameters = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_parameters = dict(checks = 70)
flann = cv2.FlannBasedMatcher(index_parameters, search_parameters)
for i in range(len(camera1_images)):
    img1_name = img2_name = camera1_images[i]
    print(img1_name)
    img1= cv2.imread(camera1_images_path+"/"+img1_name)
    img2 = cv2.imread(camera2_images_path+"/"+img2_name)
    img1_noskin = remove_skin_hsv(img1)
    img2_noskin = remove_skin_hsv(img2)
    src_points1, dst_points1 = compute_SIFT(img1_noskin, img_template, des_template, key_template, 
                                                  detector, flann, ratio_tresh= 0.82)
    src_points2, dst_points2 = compute_SIFT(img2_noskin, img_template, des_template, key_template, 
                                                  detector, flann, ratio_tresh= 0.82)
    is_good1=False
    if(len(dst_points1)>0):# If there are matches in the points
        H1, mask1 = cv2.findHomography(src_points1, dst_points1, cv2.RANSAC, 40, maxIters=3000)
        if(H1 is not None):
            rotated1 = cv2.warpPerspective(img1_noskin, H1, (img_template.shape[1], img_template.shape[0]))
            #cv2.imwrite(output_path+"/"+"1"+img1_name,rotated1)
            is_good1=check_homography(H1, img1)
    is_good2=False
    if(len(dst_points2)>0):#if there are matches in the points
        H2, mask2 = cv2.findHomography(src_points2, dst_points2, cv2.RANSAC, 40, maxIters=3000)
        if(H2 is not None):
            rotated2 = cv2.warpPerspective(img2_noskin, H2, (img_template.shape[1], img_template.shape[0]))
            #cv2.imwrite(output_path+"/"+"2"+img2_name,rotated2)
            is_good2=check_homography(H2, img2)
    result = None
    if(is_good1 and is_good2):
        print("1 and 2")
        result = cv2.addWeighted(rotated1, 0.5, rotated2, 0.5, 0)
    elif(is_good1):
        print("1")
        result = rotated1
    elif(is_good2):
        print("2")
        result = rotated2
    if(result is not None):
        cv2.imwrite(output_path+"/"+img1_name,result)