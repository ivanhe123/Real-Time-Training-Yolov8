#导入需要的库
import cv2
import numpy as np
from threading import *
from ultralytics import YOLO

#初始化变量函数（会让程序更简介）
def init_variables():
    global model, train_new_yolo, face_count,training, epochs, dataset_collected, cap
    #导入人脸检测模型
    model = YOLO('yolov8_pretrained/yolov8n-face-2.pt')
    #初始化‘同步运行’训练模块
    train_new_yolo = Thread(target=train_it)
    #初始化人脸录取数变量
    face_count = 0
    #初始化是否在训练中变量
    training = False
    #初始化epochs
    epochs = 5
    #初始化摄像头
    cap = cv2.VideoCapture(0)

#训练函数
def train_it():
    global model
    global training
    model.train(data='train/custom_data.yaml', epochs=epochs)
    training = False

def record_faces(frame, result):
    global face_count, training, model
    #检测是否在训练：1. 不在训练：记录 ’问题‘ 和 ‘回答’ 2. 在训练：什么都不做
    if not training:
        #检测记录的 ’回答‘ 和 ’问题’ 是否足够：
        if face_count < 300:
            #1. 不足够（<300）继续
            face_count += 1
        else:
            #2. 足够 （>= 300），开始用已经记录的新的 '问题' 与 '答案‘ 训练型模型
            train_new_yolo.start()
            #将在训练设为 True（在训练）
            training = True
            #初始化记录的 ’回答‘ 和 ’问题’ 数
            face_count = 0
        #获取 ’回答‘ 中的bbox
        boxes = result[0].boxes
        #记录的 ’答案‘ 和 ’问题’ 数是不是在 0-99 之间：
        if face_count >= 0 and face_count < 100:
            #1. 是在 0-99 之间：
            #把 ’问题‘ 记录在 train/dataset/images/val/ 文件夹中
            cv2.imwrite(f'train/dataset/images/val/{face_count}.png', frame)
            #把 ‘回答’ 记录在 train/dataset/labels/val/ 文件夹中
            with open(f'train/dataset/labels/val/{face_count}.txt', 'w') as file:
                for bboxes in boxes:
                    for xywh in bboxes.xywh:
                        xywh = np.array(xywh)
                        file.write(f'0 {xywh[0]/frame.shape[1]} {xywh[1]/frame.shape[0]} {xywh[2]/frame.shape[1]} {xywh[3]/frame.shape[0]}\n')
                        cnt += 1
                file.close()
        else:
            # 把 ’问题‘ 记录在 train/dataset/images/train/ 文件夹中
            cv2.imwrite(f'train/dataset/images/train/{face_count}.png', frame)
            # 把 ‘回答’ 记录在 yolov8_pretrained/dataset/labels/train/ 文件夹中
            with open(f'train/dataset/labels/train/{face_count}.txt', 'w') as file:
                for bboxes in boxes:
                    for xywh in bboxes.xywh:
                        xywh = np.array(xywh)
                        file.write(f'0 {xywh[0]/frame.shape[1]} {xywh[1]/frame.shape[0]} {xywh[2]/frame.shape[1]} {xywh[3]/frame.shape[0]}\n')
                        cnt += 1
                file.close()
init_variables()
while True:
    #读取实时图像
    ret, frame = cap.read()
    #获取检测的答案
    results = model.predict(frame)
    #记录‘问题’(frame) 和‘回答’ (results)
    record_faces(frame, results)
    #把 ’回答‘ 画出来
    annotated = results[0].plot()
    #显示画出来的效果
    cv2.imshow('returned', annotated)
    #检测是否要退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()