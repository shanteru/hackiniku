#%%
import cv2 as cv
import numpy as np

#%%
# read the image and show
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
blank_canny = np.zeros(img.shape,dtype = 'uint8')
blank_thresh = np.zeros(img.shape,dtype = 'uint8')
# cv.imshow('blank',blank)
# cv.waitKey(0)

# %%
# standard procedure for finding contours of images
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)
blur = cv.GaussianBlur(gray,(5,5),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)
canny_blur = cv.Canny(blur,125,175)
cv.imshow('canny',canny_blur)

# look at the edge and returns the contour (python list of contours)
contours_canny,hierarchies = cv.findContours(canny_blur,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE) 
# RETR_LIST arg gives all the contours
# RETR_EXTERNAL arg give the external contours
# cv.CHAIN_APPROX_NONE args describes how to approximate the contours
print(f"{len(contours_canny)} contours found")
cv.waitKey(0)

# %%
# method 2 of finding contours usign thesholding

ret,threshold = cv.threshold(gray,125,255,cv.THRESH_BINARY) 
# cv. threshold binarize image, if pixel value below 125, set to 0, if above 125, set to max (255)
cv.imshow('threshold',threshold)

# look at the edge and returns the contour (python list of contours)
contours_thresh,hierarchies = cv.findContours(threshold,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE) 
print(f"{len(contours_thresh)} contours found")

cv.drawContours(blank_thresh,contours_thresh,-1,(0,0,255),thickness=1)
cv.imshow('contours-thresh',blank_thresh)

cv.drawContours(blank_canny,contours_canny,-1,(0,255,0),thickness=1)
cv.imshow('contours-canny',blank_canny)

cv.waitKey(0)
# %%
