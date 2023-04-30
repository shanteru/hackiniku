#%% 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# read the image and show. Read as BGR
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
cv.imshow('img',img)

# smooth and blur noise in images
# method 1: averaging blur
avg = cv.blur(img,(7,7))
cv.imshow('avg',avg)

# gaussian blur
gaus_blur = cv.GaussianBlur(img,(7,7),0)
cv.imshow('gauss',gaus_blur)

# median blur ==> used for reducing salt and pepper noise
med_blur = cv.medianBlur(img,7)
cv.imshow('median',med_blur)

# bilateral blur
bilat_blur = cv.bilateralFilter(img,50,50,50)
cv.imshow('bilat',bilat_blur)

cv.waitKey(0)
# %%
