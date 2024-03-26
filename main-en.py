# import the needed modules
import cv2
from threading import Thread
import numpy as np
import os
from skimage.util import random_noise
from ultralytics import YOLO
import shutil
from PIL import Image
from scipy.stats import qmc


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

    MAX_FACE_COUNT = 60

    VALIDATION_SPLIT_COUNT = 20

    MAX_DIFF_FACES = 100

    OFFSET = 10

    return model, face_count, training, epochs, cap, MAX_FACE_COUNT, VALIDATION_SPLIT_COUNT, MAX_METRICS, MAX_DIFF_FACES, OFFSET

def create_background():

    img = np.ones((frame.shape[0], frame.shape[1], 3), dtype=np.uint8) * 255
    noise_img = random_noise(img,mode="gaussian",var=0.7)

    noise_img = np.array(255 * noise_img, dtype='uint8')

    return noise_img

def init_dirs():
    shutil.rmtree('train/dataset')
    os.makedirs('train/dataset/images/train')
    os.makedirs('train/dataset/images/val')

    os.makedirs('train/dataset/labels/train')
    os.makedirs('train/dataset/labels/val')


# rercord frames and labels + start training process function
def record_faces(frame, result):
    global face_count, training, model
    
    # training function
    def train_it():
        global model, training
        def do():
            global model, training
            m = model
                
            m.train(data='train/custom_data.yaml', epochs=epochs,pretrained=True, device=0)

            metrics = m.val()

            if float(metrics.box.map) > float(MAX_METRICS):
                MAX_METRICS = metrics.box.map
                model = m


            training = False
        Thread(target=do).start()
        
    # if training is not started:
    if not training:
        # get the bboxes of the result
        boxes = result[0].boxes
        if bool(boxes.numpy()):
            # then, see if the number datas are collected enough):
            if face_count < MAX_FACE_COUNT:
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
            if face_count >= 0 and face_count < VALIDATION_SPLIT_COUNT:
                # if the index of data is in the interval between 0 and 99
                filepath = f'train/dataset/images/val/{face_count}.png'
                labelspath = f'train/dataset/labels/val/{face_count}.txt'
            else:
                # else
                filepath = f'train/dataset/images/train/{face_count}.png'
                labelspath = f'train/dataset/labels/train/{face_count}.txt'


            with open(labelspath+'.txt', 'w') as file:
                for bboxes in boxes:
                    for xywh in bboxes.xywh:
                        xywh = xywh * np.array(
                            [1 / frame.shape[1], 1 / frame.shape[0], 1 / frame.shape[1], 1 / frame.shape[0]])
                        # write in format of yolo_txt (normalized x and y (center of bbox) and w and h (width and height of the bbox)
                        file.write(f'0 {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')
                file.close()
            with open(labelspath+'_'+str(cnt)+'.txt', 'w') as file:
                    for xywh in bboxes.xywh:
                        croped = frame[int(xywh[1] - (xywh[3] / 2)):int(xywh[1] + xywh[3] / 2),
                                         int(xywh[0] - (xywh[2] / 2)):int(xywh[0] + xywh[2] / 2)]
                        background = to_pil(array_created)
                        resized = frame.shape[0] / (MAX_DIFF_FACES / OFFSET)
                        croped_pil = to_pil(croped).resize(
                                    (int((croped.shape[1] * resized) / croped.shape[0]), int(resized)))
                        if croped.shape[1] > croped.shape[0]:
                            radius = croped_pil.size[0] / frame.shape[1]
                        else:
                            radius = croped_pil.size[1] / frame.shape[1]
                        rng = np.random.default_rng()
                        engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
                        samples = engine.random(MAX_DIFF_FACES)

                        for x in samples:
                            if x[0] - radius >= 0 and x[0] + radius <= 1 and x[1] + radius >= 0 and x[
                                        1] + radius <= 1:
                                background.paste(croped_pil,
                                                         (int(x[0] * frame.shape[1]), int(x[1] * frame.shape[0])))
                                xywh1 = [(x[0] * frame.shape[1] + croped_pil.size[0] / 2) / frame.shape[1],
                                                 (x[1] * frame.shape[0] + croped_pil.size[0] / 2) / frame.shape[0],
                                                 (croped_pil.size[0]) / frame.shape[1],
                                                 (croped_pil.size[1]) / frame.shape[0]]
                                file.write(f'0 {xywh1[0]} {xywh1[1]} {xywh1[2]} {xywh1[3]}\n')
                    background.save(filepath+'_'+str(cnt)+'.png')
                    file.close()
                cnt += 1


# call init_variables function to initialize variables
model, face_count, training, epochs, cap, MAX_FACE_COUNT, VALIDATION_SPLIT_COUNT, MAX_METRICS, MAX_DIFF_FACES,OFFSET  = init_variables()
init_dirs()
while True:

    # capture frames in real-time
    ret, frame = cap.read()

    # get results from predicting frame.
    results = model.predict(frame, show=False)

    # call function record_faces to record the frame and labels on it.
    record_faces(frame, results)

    array_created = create_background()

    # plot the bboxes
    annotated = results[0].plot()

    # show to annotated frame
    cv2.imshow('returned', annotated)

    # quit sensor
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
