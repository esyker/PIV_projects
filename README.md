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


Check a description video of the project here: 

