# import the needed modules
import cv2
import numpy as np
import os
from threading import Thread
from ultralytics import YOLO


# initialize all variables
def init_variables():
    # get Yolov8 face model
    model = YOLO('train/yolov8n-face.pt')



    # initalize face count
    face_count = 0

    # initializing variable training
    training = False

    # initialize epochs
    epochs = 5

    # initialize camera
    cap = cv2.VideoCapture(0)
    
    MAX_FACE_COUNT = 1800
    
    VALIDATION_SPLIT_COUNT = 600

    cap = cv2.VideoCapture(0)

    if len(os.listdir('train/dataset/images/train')) > 0 and len(os.listdir('train/dataset/images/val')) > 0:

        MAX_METRICS = model.val(data='train/custom_data.yaml').box.map
    else:
        MAX_METRICS = 0

    return model, face_count, training, epochs, cap, MAX_FACE_COUNT, VALIDATION_SPLIT_COUNT, MAX_METRICS



# rercord frames and labels + start training process function
def record_faces():
    global face_count, training, model
    
    # training function
    def train_it():
        global model, training
        def do():
            global model, training
            m = model
                
            m.train(data='train/custom_data.yaml', epochs=epochs)

            metrics = m.val()

            if float(metrics.box.map) > float(MAX_METRICS):
                MAX_METRICS = metrics.box.map
                model = m


            training = False
        Thread(target=do).start()
    while True:
        try:
            # if training is not started:
            if not training:
                # get the bboxes of the result
                boxes = result[0].boxes
                if bool(boxes.numpy()):
                    # then, see if the number datas are collected enough):
                    if face_count < 300:
                        # if number of data is not enough, add more
                        face_count += 1
                    else:
                        # if number of data is enough, start training thread
                        train_it()

                        # set training to true
                        training = True

                        # set face_count to 0
                        face_count = 0

                    # see if the index of data recorded is in the interval between 0 and 99:
                    if face_count >= 0 and face_count < 100:
                        # if the index of data is in the interval between 0 and 99
                        filepath = f'train/dataset/images/val/{face_count}.png'
                        labelspath = f'train/dataset/labels/val/{face_count}.txt'
                    else:
                        # else
                        filepath = f'train/dataset/images/train/{face_count}.png'
                        labelspath = f'train/dataset/labels/train/{face_count}.txt'

                    cv2.imwrite(filepath, frame)

                    with open(labelspath, 'w') as file:
                        for bboxes in boxes:
                            for xywh in bboxes.xywh:
                                xywh = xywh * np.array(
                                    [1 / frame.shape[1], 1 / frame.shape[0], 1 / frame.shape[1], 1 / frame.shape[0]])
                                # write in format of yolo_txt (normalized x and y (center of bbox) and w and h (width and height of the bbox)
                                file.write(f'0 {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')
                        file.close()
        except:
            pass
Thread(target=record_faces).start()

# call init_variables function to initialize variables
model, face_count, training, epochs, cap=init_variables()
while True:

    # capture frames in real-time
    ret, frame = cap.read()

    # get results from predicting frame.
    results = model.predict(frame, show=False)

    # plot the bboxes
    annotated = results[0].plot()

    # show to annotated frame
    cv2.imshow('returned', annotated)

    # quit sensor
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()


