#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# read the image and show
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
cv.imshow('img',img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

blank2 = np.zeros(img.shape[:2],dtype= 'uint8')
mask = cv.circle(blank2,(img.shape[1]//2,img.shape[0]//2),300,255,-1)

gray_hist = cv.calcHist([gray],[0],None,[256],[0,256])
gray_hist_mask = cv.calcHist([img],[0],mask,[256],[0,256])

plt.figure()
plt.title('Grayscale histogram')
plt.xlabel('Bins')
plt.ylabel(' # of pixels')
plt.plot(gray_hist)
plt.xlim((0,256))

plt.figure()
plt.title('Grayscale histogram (masked)')
plt.xlabel('Bins')
plt.ylabel(' # of pixels')
plt.plot(gray_hist_mask)
plt.xlim((0,256))
plt.show()
cv.waitKey(0)

#%%
# color histogram
plt.figure()
color_channel = ('b','g','r')

for i,color in enumerate (color_channel):
    
    color_hist = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(color_hist,color=color)
    plt.xlim((0,256))
plt.legend(color_channel)
plt.show()

cv.waitKey(0)
# %%
