"""
这个脚本用于可视化FPS的效果，专属于PointNet++的处理方式
看一下采样结果对训练的影响
"""
import os
from models import pointnet2_utils
from data_utils.TrimLineDataloader import TrimLineDataloader
import torch
import numpy as np


def output_ply(cell_true_labels, ball_query_index, cells, points, file_name):
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
             f"element face {len(cells)}\n" \
             f"property list uchar int vertex_indices\n" \
             f"property uchar red\n" \
             f"property uchar green\n" \
             f"property uchar blue\n" \
             f"property uchar alpha\n" \
             f"end_header\n"
    point_info = ""
    cell_info = ""
    for cell_id, cell in enumerate(cells):
        if cell_id in cell_true_labels:
            cell_info += f"3 {cell[0]} {cell[1]} {cell[2]} 255 0 0 255\n"
        elif cell_id in ball_query_index:
            cell_info += f"3 {cell[0]} {cell[1]} {cell[2]} 0 255 0 255\n"
        else:
            cell_info += f"3 {cell[0]} {cell[1]} {cell[2]} 255 255 255 255\n"

    for point in points:
        point_info += f"{point[0]} {point[1]} {point[2]} 255 255 255 255\n"

    # 写出到文件中
    with open(file_name, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(point_info)
        f.write(cell_info)
    return


# 进行一次fps并进行绘图
def fps_once(config, xyz, points, select_index, info, target_path):
    xyz_new, points_new, _, fps_index, ball_query_index = pointnet2_utils.sample_and_group(npoint=config['npoint'],
                                                                                           radius=config['radius'],
                                                                                           nsample=config['nsample'],
                                                                                           xyz=xyz,
                                                                                           points=points,
                                                                                           returnfps=True)
    points_new = points_new.permute(0, 3, 2, 1)  # [B, C+D, nsample, npoint]
    points_new = torch.max(points_new, 2)[0]  # [B, C+D, npoints]
    points_new = points_new.permute(0, 2, 1)
    # 输出FPS选点、以及球查询的可视化图
    ball_query_index = select_index[ball_query_index.reshape(-1).cpu().numpy()]
    select_index = select_index[fps_index.squeeze(0).cpu().numpy()]

    output_ply(select_index, ball_query_index, info['faces'], info['vertices'],
               os.path.join(target_path, info['file_name'][:-4] + f"_{config['npoint']}.ply"))
    return xyz_new, points_new, select_index


# 输出4次FPS+ball query的效果图
def visualize(points, labels, select_index, info, target_path):
    points = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    xyz = points[:, :, :3]
    select_index = np.array(select_index)

    config_1 = {
        "npoint": 2000,
        "radius": 0.1,
        "nsample": 32
    }
    xyz_1, points_1, select_index = fps_once(config_1, xyz, points, select_index, info, target_path)

    config_2 = {
        "npoint": 500,
        "radius": 0.2,
        "nsample": 16
    }
    xyz_2, points_2, select_index = fps_once(config_2, xyz_1, points_1, select_index, info, target_path)

    config_3 = {
        "npoint": 125,
        "radius": 0.4,
        "nsample": 8
    }
    xyz_3, points_3, select_index = fps_once(config_3, xyz_2, points_2, select_index, info, target_path)

    config_4 = {
        "npoint": 25,
        "radius": 0.8,
        "nsample": 4
    }
    xyz_4, points_4, select_index = fps_once(config_4, xyz_3, points_3, select_index, info, target_path)


if __name__ == '__main__':
    file_path = 'D:\\Dataset\\OralScan_trim_line\\实验五\\visualize_ply_expand_3_selective_downsample_20000_npy'
    target_path = "D:\\Dataset\\OralScan_trim_line\\viualize_pointnet_fps"

    dataset = TrimLineDataloader(data_root=file_path, total_point=20000, num_point=16000, transform=None,
                                 is_train=False, return_info=True)

    points, labels, select_index, info = dataset[0]
    visualize(points, labels, select_index, info, target_path)

    # for (points, labels, select_index, info) in dataset:
    #    visualize(points, labels, select_index, info, target_path)
