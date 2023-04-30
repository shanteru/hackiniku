#%%
import cv2 as cv
import numpy as np
"""
Functions 
"""
# resize and rescale images and videos for large frames
def rescale_frame(frame,scale = 0.5):
    """
    resize and rescale images and videos for large frames
    works for videos, live videos and images
    """
    width = int(frame.shape[1]*scale)
    height = int(frame.shape[0]*scale)

    dimension = (width,height)

    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)

def change_res (width,height):
    """
    only works for live videos
    """
    capture.set(3,width)
    capture.set(3,height)

##################################################################################
#%%
# read the image and show
img = cv.imread('wanqianliulian/9d7b949a95.jpg')
# cv.imshow('hoho',img)
# cv.waitKey(0)

# %%
# read videos
capture = cv.VideoCapture('wanqianliulian/metagross-adventure-pokemon-moewalls.com.mp4')
# capture = cv.VideoCapture(#int) # reference webcam with the integer for multiple cameras

while True:
    isTrue, frame = capture.read() # reads the video frame by frame, return bool that indicates success and the frame
    cv.imshow('video', rescale_frame(frame))
    if cv.waitKey(3) & 0xFF==ord('d'): # wait for 20 seconds before take in GPIO input and letter d is pressed
        break
capture.release() # release capture pointer
cv.destroyAllWindows() # destroy all windows

# %%
# drawing on images

# draw on a blank images
test = np.ones((500,500,3),dtype='uint8') # height, width, number of color channels
# test[200:300,400:500] = 255,255,0 # paint the image with a color on the specified pixels
# cv.imshow('hoho',img)
# cv.imshow('test',test)

# draw a rectangle
cv.rectangle(test,(0,0),(250,250),(0,255,0), thickness=2) # to get filled rectangle, thickness= -1
cv.circle(test,(250,250),40,(0,0,255),thickness=1)
cv.line(test,(0,0),(250,250),(255,255,255),thickness = 1)
cv.putText(test,"Hello",(255,255),cv.FONT_ITALIC,1.0,(0,255,255),1)
cv.imshow('test2',test)
cv.waitKey(0)

# %%
# important operation
# convert image to ggrayscale
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray',gray)

# blur the image to filter out noise
blur = cv.GaussianBlur(img,(3,5),cv.BORDER_DEFAULT)
cv.imshow('blur',blur)

# edge cascade/ edge detector
canny = cv.Canny(img,125,175)
cv.imshow('canny',canny)

# normally blur and canny edge to get cleaner edges
canny_blur = cv.Canny(blur,125,175)
cv.imshow('canny-blur',canny_blur)

# dilating image edges
dilate = cv.dilate(canny_blur,(11,11),iterations = 5)
cv.imshow('dilate',dilate)

# reverse dilations
eroded = cv.erode(dilate,(11,11),iterations = 5)
cv.imshow('eroded',eroded)

# resize 
resize = cv.resize(img,(2500,2500),interpolation=cv.INTER_CUBIC)
cv.imshow('resize',resize)

# crop
cropped = img[:500,:1000]
cv.imshow("cropped",cropped)

# flip 
flip = cv.flip(img,-1)
cv.imshow('flip',flip)
cv.waitKey(0)

# %%
# important transformation
def translate (img,x,y):
    """
    -x = left
    -y = up
    x = right
    y = down
    """
    transMat = np.float32([[1,0,x],[0,1,y]])
    dimension = (img.shape[1],img.shape[0])
    return cv.warpAffine(img,transMat,dimension)

def rotate (img,angle,rotation_point=None):
    (height,width) = img.shape[:2]
    if rotation_point is None:
        rotation_point = (width//2,height//2)
    rotMat = cv.getRotationMatrix2D(rotation_point,angle,1.0)
    dimension = (width,height)
    return cv.warpAffine(img,rotMat,dimension)


img_translate = translate(img,-40,-50)
cv.imshow('translate',img_translate)
img_rotate = rotate(img,90)
cv.imshow('rotate',img_rotate)
cv.waitKey(0)
# %%
