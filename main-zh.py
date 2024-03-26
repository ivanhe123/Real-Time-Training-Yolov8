# 导入需要的库
import cv2
from threading import Thread
import numpy as np
import os
from skimage.util import random_noise
from ultralytics import YOLO
import shutil
from PIL import Image
from scipy.stats import qmc
# 初始化变量


def init_variables():
    # 导入人脸检测模型
    model = YOLO("train/yolov8n-face.pt")

    # 初始化同步运行训练模块
    # train_new_yolo = Thread(target=train_it)

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
# 记录人脸函数
def record_faces(frame, result):
    global face_count, training, model, MAX_METRICS
    def to_pil(image):
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(color_coverted)
        return pil_image

    # 训练函数
    def train_it():
        global model, training, MAX_METRICS

        def do():
            global model, training, MAX_METRICS
            m = model

            # 训练模型
            m.train(data='train/custom_data.yaml', epochs=epochs,pretrained=True,device=0)

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
            # 检测是否需要继续记录人脸

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
            cnt = 0
            for bboxes in boxes:
                if face_count >= 0 and face_count < VALIDATION_SPLIT_COUNT:
                    # 记录到验证集
                    filepath = f'train/dataset/images/val/{face_count}'
                    labelspath = f'train/dataset/labels/val/{face_count}'
                

                else:
                    # 记录到训练集
                    filepath = f'train/dataset/images/train/{face_count}'
                    labelspath = f'train/dataset/labels/train/{face_count}'
                with open(labelspath+'.txt', 'w') as file:
                    for xywh in bboxes.xywh:
                        cv2.imwrite(filepath+'.png', frame)
                        xywh2 = xywh * np.array(
                                                [1 / frame.shape[1], 1 / frame.shape[0], 1 / frame.shape[1], 1 / frame.shape[0]])
                        file.write(f'0 {xywh2[0]} {xywh2[1]} {xywh2[2]} {xywh2[3]}\n')
                    file.close()
                # 保存人脸图像和位置信息到文件
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



model, face_count, training, epochs, cap, MAX_FACE_COUNT, VALIDATION_SPLIT_COUNT, MAX_METRICS, MAX_DIFF_FACES,OFFSET  = init_variables()
init_dirs()


while True:

    # 读取实时图像
    ret, frame = cap.read()
    # 获取检测的结果
    results = model.predict(frame, show=False)

    array_created = create_background()

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
