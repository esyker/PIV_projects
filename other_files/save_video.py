import cv2
import os

video_path = 'Dataset/ManyArucos.mp4'
output_folder = 'ManyArucos'

os.mkdir(output_folder)

# Opens the Video file
cap= cv2.VideoCapture(video_path)
i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite(output_folder+'/frame'+str(i)+'.jpg',frame)
    i+=1
 
cap.release()
cv2.destroyAllWindows()