
import cv2 as cv
import cvzone
from ultralytics import YOLO
import math
import torch
cv.__version__

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
cur_device = torch.device(0 if torch.cuda.is_available() else 'cpu')
print(cur_device)

# # run Static YOLO
# # load and run the pre-trained model
# model = YOLO('6_Object_Detection\\yolo-weights\\yolov8n.pt')
# results = model("6_Object_Detection\\Images\\2.png",show=True)
# cv.waitKey(0)

# run Dynamic YOLO

# webcam setup or video 
# cam = cv.VideoCapture("6_Object_Detection\\Videos\\bikes.mp4")
cam = cv.VideoCapture(0)
# cam.set(3,1280) # set height
# cam.set(4,720) # set width
model = YOLO('6_Object_Detection\\yolo-weights\\yolov8l.pt')

while True:
    success, img = cam.read()

    # Video ended
    if not success:
        print("Video ended or capture failed, closing...")
        break

    results = model(img,stream=True,device = cur_device)
    
    # enter the results object
    for i,result in enumerate(results):
        boxes = result.boxes
        print(f"list: {i,len(result),len(boxes)}")
        # get the bounding boxes for that iteration
        

        # iterate through each object found
        for box in boxes:
            # # get bounding box coord method 1
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            # # draw the box method 1
            # cv.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            
            # get bounding box coord method 2
            w,h = x2-x1,y2-y1
            bbox = x1,y1,w,h

            # get confidence level and print
            conf = math.ceil(box.conf[0]*100)/100
            if conf > 0.5:
                # get class id and name
                cls = int(box.cls[0])
                cls_name = classNames[cls]
                cvzone.putTextRect(img,f"confidence:{float(conf)},class: {cls_name}",(max(0,x1),max(30,y1)),scale = 2)
                # print(f"confidence:{float(conf)},class: {cls_name}")
                cvzone.cornerRect(img,bbox)

    cv.imshow("Image",img)

    k = cv.waitKey(1)
    if k%256 == 27:
        print("Escape hit, closing...")
        break
    # keep running until escape key is pressed
    
# release camera and destroy window
cam.release()
cv.destroyAllWindows()
