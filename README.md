# Image Processing and Vision Project

Open pivproject2021.py to check the code.

The project is divided into 4 tasks:

* Task 1 – Receiving a folder with a set of images like the one of figure 1(left), generate the corresponding corrected images (that should be saved to the output folder) as in figure 1(right) where:
The paper template is known (there is an image file with the original image of the template).
The template has a set of Aruco Markers (link here) that can be detected with available software and from which pose can be extracted. In other words, the 4 corners of each marker and their pose (Rotation and Translation) relative to the camera can be obtained from a library. Task 1 amounts to compute the transformation that maps the viewed paper to the template and render it.
Precision is a key issue here, since it influences the quality of the rendering. Hand removal or beautification of the output is not a priority, the focus is on the correctness of the perspective transformation (geometry first!).

* Task 2 – The same goal as Task 1 but  without using the Aruco markers.

* Task 3 – The same as Task 2 with rgb and depth cameras.   This Task has been canceled !

* Task 4 -  Two rgb cameras

## Some of the results

![image](https://user-images.githubusercontent.com/50277636/155801863-4bcdbdc1-257b-4fbc-a003-ec3b153c424d.png)

![image](https://user-images.githubusercontent.com/50277636/155801887-54be24b1-60d0-46be-87b6-9efdc6a115a0.png)

![image](https://user-images.githubusercontent.com/50277636/155801919-e4824663-b16b-4cac-bb27-bba3432820cf.png)

![image](https://user-images.githubusercontent.com/50277636/155801959-547da023-dcdc-4eb3-b625-18b726b664ae.png)

![image](https://user-images.githubusercontent.com/50277636/155802058-7aa9e786-610e-4d5a-ba75-71429ad6a9d7.png)

![image](https://user-images.githubusercontent.com/50277636/155801737-772ed9dd-25b7-4cb4-9784-25b4dbb9d996.png)

![image](https://user-images.githubusercontent.com/50277636/155801806-5b5edf05-6aaf-4f40-ad5a-4358b18ebb0c.png)

## Video of the working project
Check a description video of the project results' here: https://youtu.be/d2I5Krl1kHM

