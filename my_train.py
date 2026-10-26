from ultralytics import YOLO
import datetime
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
torch.use_deterministic_algorithms(False)


if __name__ == '__main__':
    # # model = YOLO(r'F:/PZ/ultralytics/cfg/models/11/yolo11n.yaml')
    model = YOLO(r'/media/ddc/新加卷/hys/qmy/PZ_8/runs/train/91512/weights/last.pt')

    model.train(data=r"/media/ddc/新加卷/hys/qmy/PZ_8/ultralytics/cfg/datasets/pz6.yaml",#数据集划分
                    ch = 9, # yz32.yaml文件路径
                    # 如果大家任务是其它的'ultralytics/cfg/default.yaml'找到这里修改task可以改成detect, segment, classify, pose
                    cache=False,
                    imgsz=512,
                    epochs=200,
                    single_cls=False,  # 是否是单类别检测
                    batch=8,
                    # save_period=5,
                    patience=20, 
                    workers=8,
                    device='2',
                    optimizer='SGD',  # using SGD 优化器
                    resume= True, # 续训的话这里填写True, 
                    augment=False,
                    project='runs/train',
                    name='915',
                    pretrained=False,
                    deterministic=False,)
                    
    