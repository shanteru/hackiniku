#%% 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image and show. Read as BGR
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
cv.imshow('img',img)

b,g,r = cv.split(img) #split the color channels and display as separate grayscale
cv.imshow('b',b)
cv.imshow('g',g)
cv.imshow('r',r)

# merge the split channels
merged = cv.merge([g,b,r])
cv.imshow('merged',merged)

# display splot channels in the color channel instead of grayscale
blank = np.zeros(img.shape[:2],dtype='uint8')
b2,g2,r2 = cv.merge([b,blank,blank]),cv.merge([blank,g,blank]),cv.merge([blank,blank,r])
cv.imshow('b2',b2)
cv.imshow('g2',g2)
cv.imshow('r2',r2)

cv.waitKey(0)
# %%
