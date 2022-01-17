import cv2
import numpy as np
from numpy.core.fromnumeric import shape, size, transpose
from numpy import double, reshape, ubyte, uint8
import time
import matplotlib.pyplot as plt
from numpy.lib.npyio import load
import scipy.io
import math

def main():
    # import camera calibration
    calib_path = 'Dataset/calib_asus.mat'
    calib_asus = scipy.io.loadmat(calib_path)

    # Read the calibration file od the camera
    Depth_cam = calib_asus['Depth_cam'][0][0][0]
    RGB_cam = calib_asus['RGB_cam'][0][0][0]
    R_d_to_rgb = calib_asus['R_d_to_rgb']
    T_d_to_rgb = calib_asus['T_d_to_rgb']

    # import rgb
    rgb_path = ('Dataset/teste/rgb_0000.jpg')
    rgb = plt.imread(rgb_path)

    # Import all depth path
    depth_vector = []
    for i in range(17):
        if i < 10:
            depth_path = 'Dataset/teste/depth_000'
        else:
            depth_path = 'Dataset/teste/depth_00'
        depth_data = scipy.io.loadmat(depth_path + str(i) + '.mat')
        depth_vector.append(depth_data)

    #  Import all depth path
    rgb_vector = []
    for i in range(17):
        if i < 10:
            rgb_path = 'Dataset/teste/rgb_000'
        else:
            rgb_path = 'Dataset/teste/rgb_00'
        rgb_data = cv2.imread(rgb_path + str(i) + '.jpg')
        rgb_vector.append(rgb_data)

    nb_data = len(rgb_vector)

    # for i in range(nb_data-1):
    #     for j in range(i+1, nb_data):
    #         depth1, depth2 = depth_vector[i]['depth_array'], depth_vector[j]['depth_array']
    #         img1, img2 = rgb_vector[i], rgb_vector[j]

    #         xyz1 = get_xyz(depth1, Depth_cam)
    #         rgbd1 = get_rgbd(xyz1,depth1, R_d_to_rgb, T_d_to_rgb, RGB_cam)

    #         xyz2 = get_xyz(depth2, Depth_cam)
    #         rgbd2 = get_rgbd(xyz2,depth2, R_d_to_rgb, T_d_to_rgb, RGB_cam)

    depth1, depth2 = depth_vector[0]['depth_array'], depth_vector[15]['depth_array']
    img1, img2 = rgb_vector[0], rgb_vector[15]

    xyz1 = get_xyz(depth1, Depth_cam)
    rgbd1 = get_rgbd(xyz1,depth1, R_d_to_rgb, T_d_to_rgb, RGB_cam)

    xyz2 = get_xyz(depth2, Depth_cam)
    rgbd2 = get_rgbd(xyz2,depth2, R_d_to_rgb, T_d_to_rgb, RGB_cam)

    plt.imshow(rgbd1, rgb_vector[0]), plt.show()
    plt.imshow(rgbd2, rgb_vector[15]), plt.show()


def get_rgbd(xyz, rgb, R, T, K_rgb):
    """
    """
    Kx = K_rgb[0,0]
    Cx = K_rgb[0,2]
    Ky = K_rgb[1,1]
    Cy = K_rgb[1,2]

    xyz_rgb = np.dot(R,np.transpose(xyz))
    xyz_rgb = np.array([xyz_rgb[0,:] + T[0], xyz_rgb[1,:] + T[1], xyz_rgb[2,:] + T[2]])

    x = xyz_rgb[0,:]
    y = xyz_rgb[1,:]
    z = xyz_rgb[2,:]

    u = np.round(Kx * x/z + Cx)
    v = np.round(Ky * y/z + Cy)

    rgb_size = np.shape(rgb)
    n_pixels = np.size(rgb)

    v[v > rgb_size[0]]= 1
    v[v < 1] = 1
    u[u > rgb_size[1]]= 1
    u[u < 1] = 1

    rgb_inds = sub2ind(rgb_size, v, u)

    rgbd = np.zeros((n_pixels,3))
    print(n_pixels)
    rgb_aux = np.reshape(rgb,n_pixels, 3)

    rgbd[n_pixels,:] = rgb_aux[rgb_inds,:]

    rgbd[xyz[:,0] == 0 & xyz[:,1] == 0 & xyz[:,2] == 0] = 0

    rgbd = uint8(np.reshape(rgbd, rgb_size))

    return(rgbd)

def get_xyz(depth_im, K):
    """
    """
    im_size = np.shape(depth_im)
    im_vec = np.reshape(depth_im, -1)

    Kx = K[0,0]
    Cx = K[0,2]
    Ky = K[1,1]
    Cy = K[1,2]  

    A = np.arange(1,im_size[1]+1)
    u = np.tile(A, (im_size[0],1))  
    u = np.reshape(u,-1) - Cx

    B = np.transpose(np.arange(1,im_size[0]+1))
    v = np.tile(B, (im_size[1],1))
    v = np.reshape(v,-1) - Cy

    xyz = np.zeros((len(u),3))
    xyz[:,2] = double(im_vec)*0.001
    xyz[:,0] = (xyz[:,2]/Kx) * u
    xyz[:,1] = (xyz[:,2]/Ky) * v

    return xyz

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols

main()