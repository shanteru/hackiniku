#%% 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image and show. Read as BGR
img = cv.imread('wanqianliulian/9d7b949a95.jpg')

cv.imshow('bgr',img)
plt.imshow(img) # plt uses RGB format, so we will see the inverse image
plt.show()

# grayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# hsv
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
cv.imshow('hsv',hsv)

# LAB
lab = cv.cvtColor(img,cv.COLOR_BGR2LAB)
cv.imshow('lab',lab)

# RGB
rgb = cv.cvtColor(img,cv.COLOR_BGR2RGB)
cv.imshow('rgb',rgb) # cv uses bgr, we will see the inverse on cv
plt.imshow(rgb) # plt uses RGB format, so plt will show the correct image
plt.show()

cv.waitKey(0)

# %%
