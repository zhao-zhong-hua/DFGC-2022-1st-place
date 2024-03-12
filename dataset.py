"""
Some codes borrowed from https://github.com/jphdotam/DFDC/blob/master/cnn3d/training/datasets_video.py
Extract images from videos in Celeb-DF v2

Author: HanChen
Date: 13.10.2020
"""

import cv2
import math
import json
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import os


class binary_Rebalanced_Dataloader(object):
    # 该方法用于初始化类的实例。接收一系列参数，包括视频文件路径列表 video_names、标签列表 video_labels、阶段（'train'、'valid'、'test'） phase、类别数量 num_class 和数据变换函数 transform。
    # 在初始化过程中，对 phase 进行了断言，确保其值在 ['train', 'valid', 'test'] 中。
    # 设定了默认的视频文件路径 default_video_name 和默认标签 default_label。
    def __init__(self, video_names=[], video_labels=[], phase='train', num_class=2, transform=None):
        assert phase in ['train', 'valid', 'test']
        self.video_names = video_names
        self.video_labels = video_labels
        self.phase = phase
        self.num_classes = num_class
        self.transform = transform
        #出错时的默认路径和图像
        self.default_video_name = '/data/linyz/Celeb-DF-v2/face_crop_png/Celeb-real/id53_0008.mp4'
        self.default_label = 0

    # 该方法用于获取数据集中指定索引 index 的样本
    def __getitem__(self, index):
        try:
            # 尝试从数据集中获取指定索引的视频路径video_name和标签 label。
            video_name = self.video_names[index]
            label = self.video_labels[index]

            #使用 os.listdir(video_name) 获取视频文件夹中的所有图像文件名列表。使用 random.sample 随机选择一个图像文件名。
            # 将选定的图像文件名赋值给 image_name。
            image_name = random.sample(os.listdir(video_name), 1)[0]

            # 构建完整的图像文件路径，将视频文件夹路径和图像文件名连接在一起。
            image_path = os.path.join(video_name, image_name)

        #如果出现异常
        except:
            # 打印出出现异常时的视频文件路径，以便进行调试。
            print(video_name)
            # 将标签设为默认值self.default_label
            label = self.default_label

            # 使用默认视频文件夹路径 self.default_video_name，随机选择一个图像文件名。
            image_name = random.sample(os.listdir(self.default_video_name), 1)[0]

            # 构建默认情况下的图像文件路径。
            image_path = os.path.join(self.default_video_name, image_name)

        # 使用 OpenCV 读取图像，将其从 BGR 色彩空间转换为 RGB
        image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        # 应用给定的数据变换函数 self.transform 到图像。该函数通常包含一系列图像处理操作，例如缩放、裁剪、归一化等。
        image = self.transform(image=image)["image"]
        # 返回处理后的图像和标签作为一个样本。这是数据加载器提供给模型训练的一个样本。
        return image, label

    def __len__(self):
        return len(self.video_names)


