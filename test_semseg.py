"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data_utils.TrimLineDataloader import TrimLineDataloader
import torch
import sys
import importlib
from tqdm import tqdm
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 分割的类别，我们只需要两类：牙龈线，其他区域
classes = ['others', 'trim_line']

# {others: 0, trim_line: 1}
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label

# {0: others, 1: trim_line}
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in testing [default: 32]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=16000, help='point number [default: 4096]')
    return parser.parse_args()


def visualize_ply(cell_labels, cells, points, file_name):
    """
    把面片输出到ply文件中
    """
    header = f"ply\n" \
             f"format ascii 1.0\n" \
             f"comment VCGLIB generated\n" \
             f"element vertex {len(points)}\n" \
             f"property float x\n" \
             f"property float y\n" \
             f"property float z\n" \
             f"property uchar red\n" \
             f"property uchar green\n" \
             f"property uchar blue\n" \
             f"property uchar alpha\n" \
             f"element face {len(cell_labels)}\n" \
             f"property list uchar int vertex_indices\n" \
             f"property uchar red\n" \
             f"property uchar green\n" \
             f"property uchar blue\n" \
             f"property uchar alpha\n" \
             f"end_header\n"
    point_info = ""
    point_labels = np.zeros(len(points))
    cell_info = ""
    for cell_label, cell in zip(cell_labels, cells):
        if cell_label == 1:
            cell_info += f"3 {cell[0]} {cell[1]} {cell[2]} 255 0 0 255\n"
            point_labels[cell[0]] = 1
            point_labels[cell[1]] = 1
            point_labels[cell[2]] = 1
        else:
            cell_info += f"3 {cell[0]} {cell[1]} {cell[2]} 255 255 255 255\n"

    for point, label in zip(points, point_labels):
        if label == 1:
            point_info += f"{point[0]} {point[1]} {point[2]} 255 0 0 255\n"
        else:
            point_info += f"{point[0]} {point[1]} {point[2]} 255 255 255 255\n"

    # 写出到文件中
    with open(file_name, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(point_info)
        f.write(cell_info)
    return


def main(args):
    # 训练数据文件夹
    root = "/media/why/新加卷/xsf/Dataset/visualize_ply_expand_3_selective_downsample_20000_npy"
    # 可视化文件夹
    visualize_dir = "/media/why/新加卷/xsf/Dataset/实验5/pred_result"
    # 训练结果文件夹
    experiment_dir = "/media/why/新加卷/xsf/Pointnet_Pointnet2_pytorch/log/sem_seg/实验5"

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    NUM_CLASSES = 2
    BATCH_SIZE = args.batch_size
    NUM_POINT = args.num_point

    TEST_DATASET = TrimLineDataloader(data_root=root, num_point=NUM_POINT, transform=None, is_train=False)

    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(NUM_CLASSES, channel=15).cuda()
    # 这里加载的是最好的model
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    with torch.no_grad():
        for tooth_file in os.listdir(root):
            points, labels, info = TEST_DATASET.parse_npy(os.path.join(root, tooth_file))

            points = torch.Tensor(points)
            points = points.float().cuda()
            # TODO: 这里将点云维度重构成了 [1, num channel, num points]
            points = points.unsqueeze(dim=0).transpose(2, 1)

            seg_pred, _ = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()

            # 把预测结果写入到ply文件中进行可视化
            visualize_ply(pred_choice, info['faces'], info['vertices'], os.path.join(visualize_dir, tooth_file[:-4] + '.ply'))


if __name__ == '__main__':
    args = parse_args()
    main(args)