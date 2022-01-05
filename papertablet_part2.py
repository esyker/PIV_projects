import cv2
import numpy as np
from numpy.core.fromnumeric import shape, size
from numpy.lib.shape_base import expand_dims, tile
from getCorners import run 
from getCorners import getArucos
import scipy.io
import time
import matplotlib.pyplot as plt
orb = cv2.ORB_create()

calib_asus = scipy.io.loadmat('Dataset/calib_asus.mat')
depth2_0 = scipy.io.loadmat('Dataset/imgs2depth/depth2cams_0000.mat')
depth2_1 = scipy.io.loadmat('Dataset/imgs2depth/depth2cams_0001.mat')
rgb2_0 = cv2.imread('Dataset/imgs2depth/rgb2cams_0000.jpg',cv2.COLOR_BGR2GRAY)
rgb2_1 = cv2.imread('Dataset/imgs2depth/rgb2cams_0001.jpg',cv2.COLOR_BGR2GRAY)

def get_xyz_asus (image_vect, image_original_size, K):
    """
    Return the xyz coordinates
    input :
    output :
        | fsx   0   Cx |   | Kx  0   Cx |
    K = |  0   fsy  Cy | = | 0   Ky  Cy | 
        |  0    0   1  |   | 0   0   1  |

    
                            | x' |     | x |
    We solved the equation  | y' | = K | y | where [x' y' 1]' is the original image.
                            | 1  |     | 1 |

    """

    fsx = K[0,0]
    Cx  = K[0,2]
    fsy = K[1,1]
    Cy  = K[1,2]

    image_size = [0,0]

    if image_size == [0,0]:
        image_size = image_original_size

        a = np.arange(1,image_size[1])  #verfifier les dimensions
        u = np.tile(a,(image_size[0],1))
        u = np.reshape(u,-1) - Cx             #reshape a matrix into a vector

        b = np.arange(1,image_size[0])
        v = np.tile(b, (image_size[1],1))
        v = np.reshape(v,-1) - Cy
        
        print(len(u))
        xyz = np.zeros[len(u),3]
    
    xyz[:,2] = np.double(image_vect)*0.001      # Convertion in meters
    xyz[:,0] = (xyz[:,2]/fsx) * u
    xyz[:,1] = (xyz[:,2]/fsy) * v

    return(xyz)

def get_rgbd(xyz,rgb,R,T,K_rgb):
    """
    """

    fsx = K_rgb[0,0]
    Cx  = K_rgb[0,2]
    fsy = K_rgb[1,1]
    Cy  = K_rgb[1,2] 

    xyz_rgb = R*np.transpose(xyz)
    xyz_rgb = xyz_rgb + T

    x = xyz_rgb[0,:]
    y = xyz_rgb[1,:]
    z = xyz_rgb[2,:]

    u = (fsx*x)/z + Cx
    v = (fsy*y)/z + Cy

    rgb_size = shape(rgb)
    n_pixels = np.size(rgb[:,:,0])

    v[v>rgb_size[0]] = 1
    v[v<1] = 1
    u[u > rgb_size[1]] = 1
    u[u < 1] = 1

    rgb_inds = plt.sub2ind(rgb_size, v, u)

    rgbd = np.zeros[n_pixels,3]
    rgb_aux = np.tile(rgb,(size(xyz),3))

    c = np.arange(1,n_pixels)
    rgbd[np.transpose(c),:] = rgb_aux[rgb_inds,:]

    rgbd[xyz[:,1] == 0 & xyz[:,2]== 0 & xyz[:,3] == 3] = 0

    #continuer cette fonction mais surtout la comprendre

    return(np.tile(rgbd,rgb_size))


def main ():

    # Get xyz from image 1: 
    size_image1 = shape(depth2_0['depth_array'])
    size_image1 = np.array(size_image1)
    image1 = np.reshape(depth2_0['depth_array'],-1)
    K = calib_asus['Depth_cam'][0][0][0]
    xyz_image1 = get_xyz_asus(image1, size_image1,K)

    # Get the virtual image aligned wuth depth:
    R_d_to__rgb = calib_asus['R_d_to_rgb']
    T_d_to_rgb = calib_asus['T_d_to_rgb']
    K_rgb = calib_asus['RGB_cam'][0][0][0]
    rgbd = get_rgbd(xyz_image1, rgb2_0, R_d_to__rgb, T_d_to_rgb, K_rgb)

    #Detect corresponding points:
    kp1, des1 = orb.detectAndCompute(rgb2_0, None)
    kp2, des2 = orb.detectAndCompute(rgb2_1, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1,des2)
    # So
    # rt them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance) 
    # Draw first 10 matches.
    img3 = cv2.drawMatches(rgb2_0,kp1,rgb2_1,kp2,matches[:10],None, flags=2)
    plt.imshow(img3),plt.show()

#print(calib_asus['Depth_cam'][0][0][0][0,0])
main()