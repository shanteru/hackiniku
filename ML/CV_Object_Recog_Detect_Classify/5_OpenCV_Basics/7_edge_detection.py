#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image and show
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
cv.imshow('img',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# edge detection
# laplacing
lap = cv.Laplacian(gray,cv.CV_64F)
lap = np.uint8(np.absolute(lap))
cv.imshow('lap',lap)

# sobel gradient magnitude
sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
sobely = cv.Sobel(gray,cv.CV_64F,0,1)
cv.imshow('sobelx',sobelx)
cv.imshow('sobely',sobely)

# combine sobel
combine_sobel = cv.bitwise_or(sobelx,sobely)
cv.imshow('combine sobel',combine_sobel)

# canny
canny = cv.Canny(gray,150,175)
cv.imshow('canny',canny)
cv.waitKey(0)
# %%
