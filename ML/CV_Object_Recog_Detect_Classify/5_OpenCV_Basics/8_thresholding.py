#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image and show
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
cv.imshow('img',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# Simple thresholding
threshold, thresh = cv.threshold(gray,100,255,cv.THRESH_BINARY)
cv.imshow('threshold',thresh)
threshold_inv, thresh_inv = cv.threshold(gray,100,255,cv.THRESH_BINARY_INV)
cv.imshow('threshold inverse',thresh_inv)

# Adaptive thresholding
adaptive_threshold = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,0)
cv.imshow('adaptive inverse',adaptive_threshold)

cv.waitKey(0)
# %%
