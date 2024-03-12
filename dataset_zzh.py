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
            video_name = self.video_names[index]
            label = self.video_labels[index]

            # 获取视频文件夹中的所有图像文件名
            all_image_names = os.listdir(video_name)

            # 选择指定数量的随机图像文件名
            num_images_to_select = 2  # 根据需要调整此数值
            selected_image_names = random.sample(all_image_names, num_images_to_select)

            # 创建列表以存储图像和它们的路径
            image = []
            # image_paths = []

            for image_name in selected_image_names:
                # 构建完整的图像文件路径
                image_path = os.path.join(video_name, image_name)
                # image_paths.append(image_path)

                # 读取和处理图像
                image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                #对每个图像进行转换处理
                image = self.transform(image=image)["image"]
                # images.append(image)
                # 将图像列表和标签作为一个样本返回
                return image, label


        except Exception as e:
            # 处理异常（您可能希望记录或打印异常以进行调试）
            print(f"处理索引为{index}的视频时发生错误：{str(e)}")

            # 将标签设置为默认值
            label = self.default_label

            # 从默认视频文件夹中选择指定数量的随机图像文件名
            num_images_to_select = 2  # 根据需要调整此数值
            selected_image_names = random.sample(os.listdir(self.default_video_name), num_images_to_select)

            # 创建列表以存储图像和它们的路径
            image = []
            # image_paths = []

            for image_name in selected_image_names:
                # 构建完整的图像文件路径
                image_path = os.path.join(self.default_video_name, image_name)
                image_paths.append(image_path)

                # 读取和处理图像
                image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                image = self.transform(image=image)["image"]
                # images.append(image)

                # 将图像列表和标签作为一个样本返回
                return image, label


    def __len__(self):
        return len(self.video_names)


