"""
Author: Benny
Date: Nov 2019
"""

import os
import sys
import torch
import numpy as np

import datetime
import logging
import provider
import importlib
import shutil
import argparse
import torch.nn.functional as F

from pathlib import Path
from tqdm import tqdm
from data_utils.ToothQualityDataLoader import ToothQualityDataLoader, farthest_point_sample, pc_normalize
import trimesh
from stl import mesh

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')

    # 这里没有进行多卡训练，需要我来设置
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    # batch size
    parser.add_argument('--batch_size', type=int, default=24, help='batch size in training')

    # 选择PointNet++，然后这里的点云没有法向量信息
    parser.add_argument('--model', default='pointnet2_cls_ssg', help='model name [default: pointnet_cls]')

    # 这里只有二分类
    parser.add_argument('--num_category', default=2, type=int,  help='classes')

    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')

    # 设置采样点的个数
    parser.add_argument('--num_point', type=int, default=5000, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')

    # 是否使用法向量
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')

    # 是否要离线增强数据，否
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')

    # 是否使用平均采样，否
    parser.add_argument('--c', action='store_true', default=False, help='use uniform sampiling')


    args = parser.parse_args()
    # 使用的模型
    args.model = 'pointnet2_cls_msg'
    # 使用法向量
    args.use_normals = True
    # 采样的点云数
    args.num_point = 8000
    # 采样方法，默认使用FPS
    args.use_uniform_sample = False
    #
    args.batch_size = 48

    return args


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def main(input_file):
    args = parse_args()

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    # classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()

    try:
        checkpoint = torch.load('./checkpoints/best_model.pth')
        classifier.load_state_dict(checkpoint['model_state_dict'])
    except:
        print('No existing model...')
        exit(-1)

    '''prediction'''
    origin_mesh = trimesh.load_mesh(input_file)
    # 降采样到10000
    downsampled_mesh = origin_mesh.simplify_quadratic_decimation(10000)
    # 降采样文件输出
    downsampled_mesh.export('./downsampled.stl')


    # 把降采样文件转换成txt文本文件
    tooth = mesh.Mesh.from_file('./downsampled.stl')
    # 面片法向量
    normals = tooth.normals
    centroids = tooth.centroids

    with open('./downsampled.txt', 'w', encoding='ascii') as f:
        for centroid, normal in zip(centroids, normals):
            f.write(f'{centroid[0]},{centroid[1]},{centroid[2]},{normal[0]},{normal[1]},{normal[2]}\n')
            f.flush()

    point_set = np.loadtxt('./downsampled.txt', delimiter=',').astype(np.float32)

    # TODO: 这里的逻辑是不是有问题
    if args.use_uniform_sample:
        point_set = farthest_point_sample(point_set, args.num_point)
    else:
        # 随机打乱
        np.random.shuffle(point_set)
        point_set = point_set[0:args.num_point, :]


    point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
    if not args.use_normals:
        point_set = point_set[:, 0:3]

    # 这里需要组合成一个batch
    points = point_set[None, :, :]
    points = torch.Tensor(points)
    points = points.transpose(2, 1)

    if not args.use_cpu:
        points = points.cuda()

    with torch.no_grad():
        classifier.eval()
        vote_pool = torch.zeros(1, num_class).cuda()
        # 投票
        vote_num = 5
        for _ in range(vote_num):
            pred, _ = classifier(points)
            vote_pool += pred
        pred = vote_pool / vote_num

    # 计算钻牙正常的概率
    score = F.softmax(pred.data, dim=1)[0][0]
    pred_choice = pred.data.max(1)[1].cpu().item()

    print(f'预测的钻牙类别是"{"好牙" if pred_choice == 0 else "坏牙"}", 置信度是{score if pred_choice == 0 else 1 - score}')


if __name__ == '__main__':
    # 这里是需要输入的文件
    input_file = '/media/why/77B8B456EE73FE06/users/xsf_ubuntu/Dataset/Tooth_quality/TP003.stl'
    main(input_file)
