# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 12:04:57 2022

@author: Utilizador
"""
import cv2
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def get_rgb_mask(img, debug=False):
    assert isinstance(img, np.ndarray), 'image must be a numpy array'
    assert img.ndim == 3, 'skin detection can only work on color images'
    lower_thresh = np.array([45, 52, 108], dtype=np.uint8)
    upper_thresh = np.array([255, 255, 255], dtype=np.uint8)
    mask_a = cv2.inRange(img, lower_thresh, upper_thresh)
    mask_b = 255 * ((img[:, :, 2] - img[:, :, 1]) / 20)
    mask_c = 255 * ((np.max(img, axis=2) - np.min(img, axis=2)) / 20)
    # msk_rgb = cv2.bitwise_and(mask_c, cv2.bitwise_and(mask_a, mask_b))
    mask_d = np.bitwise_and(np.uint64(mask_a), np.uint64(mask_b))
    msk_rgb = np.bitwise_and(np.uint64(mask_c), np.uint64(mask_d))
    msk_rgb[msk_rgb < 128] = 0
    msk_rgb[msk_rgb >= 128] = 1
    return msk_rgb.astype(float)

def closing(mask):
    assert isinstance(mask, np.ndarray), 'mask must be a numpy array'
    assert mask.ndim == 2, 'mask must be a greyscale image'
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask

def grab_cut_mask(img_col, mask):
    assert isinstance(img_col, np.ndarray), 'image must be a numpy array'
    assert isinstance(mask, np.ndarray), 'mask must be a numpy array'
    assert img_col.ndim == 3, 'skin detection can only work on color images'
    assert mask.ndim == 2, 'mask must be 2D'

    kernel = np.ones((50, 50), np.float32) / (50 * 50)
    dst = cv2.filter2D(mask, -1, kernel)
    dst[dst != 0] = 255
    free = np.array(cv2.bitwise_not(dst), dtype=np.uint8)

    grab_mask = np.zeros(mask.shape, dtype=np.uint8)
    grab_mask[:, :] = 2
    grab_mask[mask == 255] = 1
    grab_mask[free == 255] = 0

    if np.unique(grab_mask).tolist() == [0, 1]:
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if img_col.size != 0:
            mask, bgdModel, fgdModel = cv2.grabCut(img_col, grab_mask, None, bgdModel, fgdModel, 5,
                                                   cv2.GC_INIT_WITH_MASK)
            mask = np.where((mask == 2) | (mask == 0), 0, 1).astype(np.uint8)

    return mask

input_images_path = 'Dataset/Gehry/images'
img_name = 'rgb0077.jpg'
frame = cv2.imread(input_images_path+"/"+img_name)
cv2.imshow('original',frame)
mask = get_rgb_mask(frame)
mask = mask.astype(np.uint8)
mask = closing(mask)
res = grab_cut_mask(frame, mask)
cv2.imshow('just skin', res)

"""
B, G, R =  [frame[...,BGR] for BGR in range(3)]# [...] is the same as [:,:]
RGB_MAX = np.maximum.reduce([R,G,B])
RGB_MIN = np.minimum.reduce([R,G,B])
rule1= np.logical_and.reduce([R>95,G>40,B>20,RGB_MAX-RGB_MIN>15,abs(R-G)>15,R>G,R>B])
rule2=np.logical_and.reduce([R>220,G>210,B>170,abs(R-G)<=15,R>B,G>B])
RGB_Rule=np.logical_or(rule1,rule2)
img_bw = RGB_Rule.astype(np.uint8)
"""