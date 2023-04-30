#%% 
import cv2 as cv

# setup camera
cam = cv.VideoCapture(0)

cv.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    frame_save = frame.copy()
    if not ret:
        print("failed to grab frame")
        break

    # do face recognition here
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    # set the detected faces rectangle
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=1)
    print(f'no.of.faces found:{len(faces_rect)}')

    # draw bounding box for faces
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), thickness=2)
    cv.imshow("test",frame)

    k = cv.waitKey(1)
    # ESC pressed
    if k%256 == 27:
        print("Escape hit, closing...")
        break
    # SPACE pressed
    elif k%256 == 32:
        img_name = "faces\\jq\\jq_{}.png".format(img_counter)
        cv.imwrite(img_name, frame_save)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv.destroyAllWindows()
# %%
