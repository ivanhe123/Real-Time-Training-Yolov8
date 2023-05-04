# 导入需要的库
import cv2
from threading import Thread
import numpy as np
from ultralytics import YOLO


# 初始化变量
def init_variables():


    # 导入人脸检测模型
    model = YOLO('train/yolov8n-face.pt')

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

    return model, face_count, training, epochs, cap



# 记录人脸函数
def record_faces(frame, result):
    global face_count, training, model
    # 训练函数
    def train_it():
        global model, training
        def do():
            global model, training

            # 训练模型
            model.train(data='train/custom_data.yaml', epochs=epochs)

            # 将在训练设为 False（不在训练）
            training = False
        Thread(target=do).start()
    # 检测是否在训练
    if not training
        boxes = result[0].boxes
        if bool(boxes.numpy()):
            #检测是否需要继续记录人脸
            
            if face_count < 300:
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
            if face_count >= 0 and face_count < 100:
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
model,face_count, training, epochs, cap = init_variables()

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

