"""
Author: HanChen
Date: 21.06.2022
"""
# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
from torch.autograd import Variable
import argparse

import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
from logger import create_logger

from network.models import get_swin_transformers
from transforms import build_transforms
from metrics import get_metrics
from dataset import binary_Rebalanced_Dataloader

import os



######################################################################
# Save model
def save_network(network, save_filename):
    torch.save(network.cpu().state_dict(), save_filename)
    if torch.cuda.is_available():
        network.cuda()


def load_network(network, save_filename):
    network.load_state_dict(torch.load(save_filename))
    return network

def parse_args():
    parser = argparse.ArgumentParser(description='Training network')

    #DFDC数据集的根路径
    parser.add_argument('--root_path_dfdc', default='/data/linyz/DFDC/face_crop_png',
                        type=str, help='path to DFDC dataset')
    #保存训练结果的路径。
    parser.add_argument('--save_path', type=str, default='./save_result')

    # 使用的模型的名称
    parser.add_argument('--model_name', type=str, default='swin_large_patch4_window12_384_in22k')

    #用于训练的GPU的ID
    parser.add_argument('--gpu_id', type=int, default=6)

    #分类任务中的类别数量。
    parser.add_argument('--num_class', type=int, default=2)

    # 分类任务中每个类别的名称。
    parser.add_argument('--class_name', type=list, 
                        default=['real', 'fake'])

    #训练的轮数。
    parser.add_argument('--num_epochs', type=int, default=60)

    #在多少次迭代后调整学习率。
    parser.add_argument('--adjust_lr_iteration', type=int, default=30000)

    #神经网络中的dropout，防止过拟合
    parser.add_argument('--droprate', type=float, default=0.2)

    # 优化器的初始学习率。
    parser.add_argument('--base_lr', type=float, default=0.00005)

    #每个训练批次的样本数量。
    parser.add_argument('--batch_size', type=int, default=24)

    #输入图像的分辨率。
    parser.add_argument('--resolution', type=int, default=384)

    # 每个验证批次的样本数量。
    parser.add_argument('--val_batch_size', type=int, default=128)
    args = parser.parse_args()
    return args

#从指定目录中的文本文件加载视频文件路径和相应的标签，
# txt_path='./txt'：默认文本文件所在的目录路径，默认为当前目录下的 txt 目录，logger=None：一个用于记录日志的对象，默认为 None。
def load_txt(txt_path='./txt', logger=None):
    # 获取指定目录下的所有文本文件名
    txt_names = os.listdir(txt_path)
    # 创建两个空列表，用于存储加载的视频路径和标签。
    tmp_videos, tmp_labels = [], []
    # 遍历每个文本文件。
    for txt_name in txt_names:
        # 打开当前文本文件。
        with open(os.path.join(txt_path, txt_name), 'r') as f:
            # 读取文本文件中的每一行。
            videos_names = f.readlines()
            # 遍历每一行。
            for i in videos_names:
                # 检查是否包含'landmarks'，如果是则跳过当前行
                if i.find('landmarks') != -1:
                    continue
                # 检查当前视频路径对应的目录是否为空，如果是则跳过当前行
                if len(os.listdir(i.strip().split()[0])) == 0:
                    continue
                # 将视频路径和标签添加到临时列表中
                tmp_videos.append(i.strip().split()[0])
                tmp_labels.append(int(i.strip().split()[1]))

        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        # 打印当前加载的标签信息
        print(timeStr, len(tmp_labels), sum(tmp_labels), sum(tmp_labels)/len(tmp_labels))
    # 返回加载的视频路径列表和标签列表
    return tmp_videos, tmp_labels


def main():
    #解析命令行参数
    args = parse_args()

    #创建日志记录器
    logger = create_logger(output_dir='%s/report' % args.save_path, name=f"{args.model_name}")
    logger.info('Start Training %s' % args.model_name)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    logger.info(timeStr)  

    #根据输入图像，构建数据转换函数
    transform_train, transform_test = build_transforms(args.resolution, args.resolution, 
                        max_pixel_value=255.0, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225])

    #加载训练数据集
    #存储训练数据集的视频文件路径和对应的标签
    train_videos, train_labels = [], []

    #迭代构建访问训练数据集的子文件夹，idx 从0到49循环遍历
    for idx in tqdm(range(0, 50)):

        #构建当前子文件夹的名称，格式为 dfdc_train_part_0, dfdc_train_part_1, ..., dfdc_train_part_49。
        sub_name = 'dfdc_train_part_%d' % idx

        #构建当前子文件夹的完整路径，args.root_path_dfdc 是存储数据集的根目录。
        video_sub_path = os.path.join(args.root_path_dfdc, sub_name)

        #: 打开当前子文件夹下的 metadata.json 文件，该文件包含了每个视频文件的元数据信息。
        with open(os.path.join(video_sub_path, 'metadata.json')) as metadata_json:
            #将 metadata.json 文件中的内容加载到 metadata 变量中，以便后续处理。
            metadata = json.load(metadata_json)

        #对 metadata 中的每一项进行迭代处理。
        for key, value in metadata.items():
            #检查当前视频文件的标签是否为“FAKE”。如果标签是“FAKE”，将标签设为1，否则设为0。
            if value['label'] == 'FAKE': # FAKE or REAL
                label = 1
            else:
                label = 0

            #构建当前视频文件的完整路径。
            inputPath = os.path.join(args.root_path_dfdc, sub_name, key)

            #检查当前视频文件夹是否为空。如果为空，则跳过该视频文件。
            if len(os.listdir(inputPath)) == 0:
                continue
            #将当前视频文件的路径和标签分别添加到 train_videos 和 train_labels 列表中。通过这段代码，训练数据集中每个视频文件夹的路径和对应的标签被收集和存储起来，以便后续在模型训练过程中使用。
            train_videos.append(inputPath)
            train_labels.append(label)


    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    # 这行代码打印了当前时间、训练数据集中的样本数、标签为1的样本数（即"FAKE"样本数）、以及标签为1的样本在整个数据集中的比例（即“FAKE”的比例）。
    print(timeStr, len(train_labels), sum(train_labels), sum(train_labels)/len(train_labels))
    # 这行代码调用了 load_txt 函数来加载额外的训练数据。他从 .txt 文件夹中加载包含视频路径和标签的txt文件。
    tmp_videos, tmp_labels = load_txt(txt_path='./txt')
    # 这两行代码将新加载的训练数据（视频路径和对应标签）添加到原始训练数据集的末尾。
    train_videos += tmp_videos
    train_labels += tmp_labels
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    # 这行代码再次打印了当前时间、更新后的训练数据集中的样本数、标签为1的样本数（即"FAKE"样本数）、以及标签为1的样本在整个数据集中的比例（即“FAKE”的比例）。
    print(timeStr, len(train_labels), sum(train_labels), sum(train_labels)/len(train_labels))
    #通过上述部分代码，我们可以在加载额外训练数据之前和之后，观察训练数据集的变化情况，包括样本数量和标签分布等信息。

    # 构建训练数据加载器，使用 binary_Rebalanced_Dataloader 类构建训练数据加载器，其中包括视频文件路径、标签、训练阶段、类别数量和数据转换。
    train_dataset = binary_Rebalanced_Dataloader(video_names=train_videos, video_labels=train_labels, phase='train', 
                                                num_class=args.num_class, transform=transform_train)
    timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
    #输出训练数据集的大小
    print(timeStr, 'All Train videos Number: %d' % (len(train_dataset)))

    #创建神经网络模型：使用 get_swin_transformers 函数创建 Swin Transformer 模型，指定模型名称和类别数量，并将模型移到 GPU 上
    model = get_swin_transformers(model_name=args.model_name, num_classes=args.num_class).cuda()

    #定义优化器、学习率调度器和损失函数：使用 AdamW 优化器、ExponentialLR 学习率调度器和交叉熵损失函数。
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.base_lr)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # 构建训练数据加载器，使用 PyTorch 的 DataLoader 类构建训练数据加载器，指定批次大小、是否舍弃最后一个不完整的批次、是否打乱数据等。
    # numworks指定了用于数据加载的线程数量。通过多线程加载数据可以加快数据加载速度，提高训练效率。
    # spin_memory=True: 如果为True，数据加载器会将数据存储在固定内存中，以提高GPU的数据传输效率。通常在使用GPU训练时，将此选项设置为True可以加快训练速度。
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True,
                                               shuffle=True, num_workers=6, pin_memory=True)


    # 这行代码定义了一个包含损失函数名称的列表 loss_name，这个列表中只包含一个元素 'BCE'，即二元交叉熵损失函数。
    loss_name = ['BCE']
    # 这行代码定义了一个变量iteration，用于记录当前训练的迭代次数，初始化为0。
    iteration = 0
    # 这行代码定义了一个字典 running_loss，其中的键是 loss_name 中的损失函数名称，值初始化为0。这样做的目的是为每个损失函数创建一个累积损失值的计数器，在训练过程中每个迭代都会更新对应损失函数的累积损失值。
    running_loss = {loss: 0 for loss in loss_name}
    # 以上代码创建了一个用于跟踪模型训练过程中不同损失函数的累积损失值的字典 running_loss，
    # 并初始化了迭代次数变量 iteration。在模型训练的每个迭代中，将根据实际的损失值更新 running_loss 中对应的损失值。

    # 训练模型：
    # 迭代训练模型，计算损失并更新模型参数。
    # 在每个 epoch 中，输出日志信息，包括当前 epoch、迭代次数、损失等。
    # 定期保存模型，例如每隔 20 个 epoch 保存一次。
    for epoch in range(args.num_epochs):
        timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
        logger.info(timeStr+'Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        logger.info(timeStr+'-' * 10)

        model.train(True)  # Set model to training mode
        # Iterate over data (including images and labels).
        # 迭代数据（包括图像和标签）。
        for index, (images, labels) in enumerate(train_loader):
            iteration += 1
            # wrap them in Variable

            # 将数据包装为Variable类型
            # 在早期的PyTorch版本中，Variable类型用于包装张量（Tensor），以便进行自动求导和计算梯度。
            # 然而，从PyTorch 0.4版本开始，Variable被弃用，而张量（Tensor）直接支持自动求导功能。
            # 如果代码是基于较新版本的PyTorch，请直接将张量传递给模型进行计算即可，无需使用Variable进行包装。
            images = Variable(images.cuda().detach())
            labels = Variable(labels.cuda().detach())


            # zero the parameter gradients
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播，计算输出
            outputs = model(images)
            # Calculate loss
            # 计算损失
            loss = criterion(outputs, labels)

            # update the parameters
            # 更新参数
            # 通过反向传播计算梯度
            loss.backward()
            # 根据计算的梯度更新模型的参数。
            optimizer.step()

            running_loss['BCE'] += loss.item()
            # break
            # 如果迭代次数达到100的倍数，打印出当前epoch、迭代次数、总迭代次数和平均损失值。
            if iteration % 100 == 0:
                timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
                logger.info(timeStr+'Epoch: {:g}, Itera: {:g}, Step: {:g}, BCE: {:g} '.
                      format(epoch, index, len(train_loader), *[running_loss[name] / 100 for name in loss_name]))
                running_loss = {loss: 0 for loss in loss_name}
            # 如果迭代次数达到args.adjust_lr_iteration的倍数，调整学习率。
            if iteration % args.adjust_lr_iteration == 0:
                scheduler.step()

        # 每经过20个epochs，保存一次模型。
        if epoch % 20 == 0:
            timeStr = time.strftime('[%Y-%m-%d %H:%M:%S]',time.localtime(time.time()))
            logger.info(timeStr + '  Save  Model  ')
            save_network(model, '%s/models/%s_%d.pth' % (args.save_path, args.model_name, epoch))

    # 保存最终训练好的模型。
    save_network(model, '%s/models/%s.pth' % (args.save_path, args.model_name))


if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists('%s/models' % args.save_path):
        os.makedirs('%s/models' % args.save_path)
    if not os.path.exists('%s/report' % args.save_path):
        os.makedirs('%s/report' % args.save_path)

    main()
