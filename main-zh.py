# 导入需要的库
import cv2
from threading import Thread
import numpy as np
import os
from ultralytics import YOLO


# 初始化变量
def init_variables():


    # 导入人脸检测模型
    model = YOLO("train/yolov8n-face.pt")

    # 初始化同步运行训练模块
    #train_new_yolo = Thread(target=train_it)

    # 初始化人脸录取数变量
    face_count = 0

    # 初始化是否在训练中变量
    training = False

    # 初始化 epochs
    epochs = 5

    # 初始化摄像头
    cap = cv2.VideoCapture(0)

    if len(os.listdir('train/dataset/images/train')) > 0 and len(os.listdir('train/dataset/images/val')) > 0:

        MAX_METRICS = model.val(data='train/custom_data.yaml').box.map
    else:
        MAX_METRICS = 0

    MAX_FACE_COUNT = 1800

    VALIDATION_SPLIT_COUNT = 600

    return model, face_count, training, epochs, cap, MAX_FACE_COUNT, VALIDATION_SPLIT_COUNT, MAX_METRICS



# 记录人脸函数
def record_faces(frame, result):
    global face_count, training, model, MAX_METRICS
    # 训练函数
    def train_it():
        global model, training, MAX_METRICS
        def do():
            global model, training, MAX_METRICS

            m = model
            
            # 训练模型
            m.train(data='train/custom_data.yaml', epochs=epochs)

            metrics = m.val()

            if float(metrics.box.map) > float(MAX_METRICS):
                MAX_METRICS = metrics.box.map
                model = m

            # 将在训练设为 False（不在训练）
            training = False
        Thread(target=do).start()
    # 检测是否在训练
    if not training:
        boxes = result[0].boxes
        if bool(boxes.numpy()):
            #检测是否需要继续记录人脸
            
            if face_count < MAX_FACE_COUNT:
                face_count += 1
            else:
                # 开始训练新模型
                train_it()
                
                # 将在训练设为 True（在训练）
                training = True

                # 重置人脸录取次数
                face_count = 0

            # 获取人脸位置信息
            boxes = result[0].boxes

            # 保存人脸图像和位置信息到文件
            if face_count >= 0 and face_count < VALIDATION_SPLIT_COUNT:
                # 记录到验证集
                filepath = f'train/dataset/images/val/{face_count}.png'
                labelspath = f'train/dataset/labels/val/{face_count}.txt'
            else:
                # 记录到训练集
                filepath = f'train/dataset/images/train/{face_count}.png'
                labelspath = f'train/dataset/labels/train/{face_count}.txt'

            # 保存人脸图像
            cv2.imwrite(filepath, frame)

            # 保存位置信息到文件
            with open(labelspath, 'w') as file:
                for bboxes in boxes:
                    for xywh in bboxes.xywh:
                        xywh = xywh * np.array(
                            [1 / frame.shape[1], 1 / frame.shape[0], 1 / frame.shape[1], 1 / frame.shape[0]])
                        file.write(f'0 {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}\n')
                file.close()
model,face_count, training, epochs, cap, MAX_FACE_COUNT, VALIDATION_SPLIT_COUNT, MAX_METRICS = init_variables()

while True:
    # 读取实时图像
    ret, frame = cap.read()

    # 获取检测的结果
    results = model.predict(frame, show=False)

    # 记录人脸
    record_faces(frame, results)

    # 在图像上绘制检测结果
    annotated = results[0].plot()

    # 显示画出来的效果
    cv2.imshow('returned', annotated)

    # 检测是否要退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()

