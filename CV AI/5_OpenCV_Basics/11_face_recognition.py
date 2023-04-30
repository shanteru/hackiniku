#%% 
import cv2 as cv
import os 
import numpy as np

# %%
people = []
# get list of directories in the faces folder
DIR = r'C:\Users\User\OneDrive - Monash University\Hackathon Stuff\python AI\5_OpenCV_Basics\faces'
direc = os.listdir(DIR)
for i in direc:
    people.append(i)

# %%
# face recognition training pipeline

# get the facial features and append labels
def create_train(people,showFaces = False):
    features = []
    labels = []
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    i = 0
    for person in people:
        person_dir = os.path.join(DIR,person)
        # set the label for the current person
        label = people.index(person)

        for img in os.listdir(person_dir):
            img_path = os.path.join(person_dir,img)
            cur_img_array = cv.imread(img_path)
            gray = cv.cvtColor(cur_img_array,cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=2)
            for (x,y,w,h) in faces_rect:
                faces_features = gray[y:y+h,x:x+w]
                features.append(faces_features)
                labels.append(label)
                if showFaces:
                    cv.imshow(f'{i}:{label}',faces_features)
                    
                i = i+1
    cv.waitKey(0)
    return np.array(features,dtype=object),np.array(labels)

features,labels = create_train(people,False)
print(f'number of faces: {len(features)},{len(labels)}')

#%%
# train the face recognizer with the facial features and label
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(features,labels)
np.save('models\\features.npy',features)
np.save('models\\labels.npy',labels)
face_recognizer.save('models\\face_trained.yml')

###############################################################################
# %%
# face recognition deployment pipeline
# get classifiers
people = []
# get list of directories in the faces folder
DIR = r'C:\Users\User\OneDrive - Monash University\Hackathon Stuff\python AI\5_OpenCV_Basics\faces'
direc = os.listdir(DIR)
for i in direc:
    people.append(i)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('models\\face_trained.yml')

# setup camera for real time image capture, processing and classfication
cam = cv.VideoCapture(0)
cv.namedWindow("test")

while True:
    ret, frame = cam.read()
    frame_save = frame.copy()
    if not ret:
        print("failed to grab frame")
        break

    # do face detection here
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    haar_cascade = cv.CascadeClassifier('haar_face.xml')

    # set the detected faces rectangle
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=1)

    for (x,y,w,h) in faces_rect:
         # draw bounding box for detected face
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0), thickness=2)
        faces_features_pred = gray[y:y+h,x:x+w]

        # predict whose face
        face_label,confidence = face_recognizer.predict(faces_features_pred)
        if confidence > 50:
            cv.putText(frame,f'Name: {people[face_label]} {confidence:.2f}',(x,y),cv.FONT_ITALIC,0.5,(255,255,255),)
        else:
            cv.putText(frame,f'Siapa ni',(x,y),cv.FONT_ITALIC,0.5,(255,255,255),)
    
    cv.imshow("test",frame)

    k = cv.waitKey(1)
    # ESC pressed
    if k%256 == 27:
        print("Escape hit, closing...")
        break

cam.release()

cv.destroyAllWindows()

# %%
