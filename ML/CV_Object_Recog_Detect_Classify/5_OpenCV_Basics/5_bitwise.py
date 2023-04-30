#%% 
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# read the image and show. Read as BGR
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
cv.imshow('img',img)

blank = np.zeros([400,400],dtype='uint8')
rectangle = cv.rectangle(blank.copy(),(30,30),(370,370),255,-1)
circle = cv.circle(blank.copy(),(200,200),200,255,-1)

cv.imshow('blank',blank)
cv.imshow('rectangle',rectangle)
cv.imshow('circle',circle)

# bitwise AND ==> intersecting
bitwise_and = cv.bitwise_and(rectangle,circle)
cv.imshow('bitwise and',bitwise_and)

# bitwise OR ==> intersecting and non intersecting
bitwise_or = cv.bitwise_or(rectangle,circle)
cv.imshow('bitwise or',bitwise_or)

# bitwise XOR ==> non intersecting
bitwise_xor = cv.bitwise_xor(rectangle,circle)
cv.imshow('bitwise xor',bitwise_xor)

# bitwise not ==> invert
bitwise_not = cv.bitwise_not(circle)
cv.imshow('bitwise not',bitwise_not)

cv.waitKey(0)

# %%
# masking
# can be used to remove unwanted sections of images
blank2 = np.zeros(img.shape[:2],dtype= 'uint8')
mask = cv.circle(blank2,(img.shape[1]//2,img.shape[0]//2),300,255,-1)
masked_image = cv.bitwise_and(img,img,mask=mask)
cv.imshow('mask',mask)
cv.imshow('masked image',masked_image)
cv.waitKey(0)
# %%
