#import the needed modules
import cv2
import numpy as np
from threading import *
from ultralytics import YOLO

#initialize all variables
def init_variables():
    global model, train_new_yolo, face_count,training, epochs, dataset_collected, cap
    #get Yolov8 face model
    model = YOLO('yolov8_pretrained/yolov8n-face-2.pt')
    #initialize threading
    train_new_yolo = Thread(target=train_it)
    #initalize face count
    face_count = 0
    #initializing variable training
    training = False
    #initialize epochs
    epochs = 5
    #initialize camera
    cap = cv2.VideoCapture(0)

#training function
def train_it():
    global model
    global training
    model.train(data='train/custom_data.yaml', epochs=epochs)
    training = False

#rercord frames and labels + start training process function
def record_faces(frame, result):
    global face_count, training, model
    #if training is not started:
    if not training:
        #then, see if the number datas are collected enough):
        if face_count < 300:
            #if number of data is not enough, add more
            face_count += 1
        else:
            #if number of data is enough, start training thread
            train_new_yolo.start()
            #set training to true
            training = True
            #set face_count to 0
            face_count = 0
        #get the bboxes of the result
        boxes = result[0].boxes
        #see if the index of data recorded is in the interval between 0 and 99:
        if face_count >= 0 and face_count < 100:
            #if the index of data is in the interval between 0 and 99
            #record the frame captured in real-time into train/dataset/images/val/
            cv2.imwrite(f'train/dataset/images/val/{face_count}.png', frame)
            #record the labels to put on the frame captured in real-time into train/dataset/images/val/
            with open(f'train/dataset/labels/val/{face_count}.txt', 'w') as file:
                for bboxes in boxes:
                    for xywh in bboxes.xywh:
                        xywh = np.array(xywh)
                        #write in format of yolo_txt (normalized x and y (center of bbox) and w and h (width and height of the bbox)
                        file.write(f'0 {xywh[0]/frame.shape[1]} {xywh[1]/frame.shape[0]} {xywh[2]/frame.shape[1]} {xywh[3]/frame.shape[0]}\n')
                file.close()
        else:
            #else
            #record the frame captured in real-time into train/dataset/images/val/
            cv2.imwrite(f'train/dataset/images/train/{face_count}.png', frame)
            #record the labels to put on the frame captured in real-time into train/dataset/images/val/
            with open(f'train/dataset/labels/train/{face_count}.txt', 'w') as file:
                for bboxes in boxes:
                    for xywh in bboxes.xywh:
                        xywh = np.array(xywh)
                        #write in format of yolo_txt (normalized x and y (center of bbox) and w and h (width and height of the bbox)
                        file.write(f'0 {xywh[0]/frame.shape[1]} {xywh[1]/frame.shape[0]} {xywh[2]/frame.shape[1]} {xywh[3]/frame.shape[0]}\n')
                file.close()
#call init_variables function to initialize variables
init_variables()
while True:
    #capture frames in real-time
    ret, frame = cap.read()
    #get results from predicting frame.
    results = model.predict(frame)
    #call function record_faces to record the frame and labels on it.
    record_faces(frame, results)
    #plot the bboxes
    annotated = results[0].plot()
    #show to annotated frame
    cv2.imshow('returned', annotated)
    #quit sensor
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()