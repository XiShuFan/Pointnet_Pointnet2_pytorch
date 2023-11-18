"""
这个脚本用于可视化FPS的效果，专属于PointNet++的处理方式
看一下采样结果对训练的影响
"""
import os
from models import pointnet2_utils
from data_utils.TrimLineDataloader import TrimLineDataloader
import torch
import numpy as np


def output_ply(cell_true_labels, ball_query_index, cells, points, file_name, only_fps):
    """
    把面片输出到ply文件中
    only_fps: 是否只输出fps选择的面片
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
             f"element face {len(cell_true_labels) if only_fps else len(cells)}\n" \
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
            if only_fps:
                continue
            cell_info += f"3 {cell[0]} {cell[1]} {cell[2]} 0 255 0 255\n"
        else:
            if only_fps:
                continue
            cell_info += f"3 {cell[0]} {cell[1]} {cell[2]} 255 255 255 255\n"

    for point in points:
        point_info += f"{point[0]} {point[1]} {point[2]} 255 255 255 255\n"

    # 写出到文件中
    with open(file_name, 'w', encoding='ascii') as f:
        f.write(header)
        f.write(point_info)
        f.write(cell_info)
    return


def output_edge(xyzs, labels, select_index, radius, nsample, info, target_path):
    """
    我们的目标是可视化点云的边缘，通过球查询来判断每个点是否是边缘
    """
    n = select_index.shape[0]

    xyz_labels = labels[select_index]
    # 使用球查询，来判断当前点的邻居是否是同一类别，如果不同类别的邻居数量超过40%，就认为是边缘点
    sqrdists = pointnet2_utils.square_distance(xyzs, xyzs)

    # 按照距离升序排序
    sqrdists, sqrdists_idx = sqrdists.sort(dim=-1)
    sqrdists = sqrdists[:, :, :nsample]
    sqrdists_idx = sqrdists_idx[:, :, :nsample]
    sqrdists = sqrdists.squeeze(dim=0).cpu().numpy()
    sqrdists_idx = sqrdists_idx.squeeze(dim=0).cpu().numpy()
    query_self = np.arange(n).reshape(n, 1).repeat(nsample, axis=1)
    mask = sqrdists > radius ** 2
    sqrdists_idx[mask] = query_self[mask]

    # 现在每个点的邻居已经知道了，我们遍历每个点，进行统计
    pointcloud_info = ""
    points = xyzs.cpu().squeeze(dim=0).numpy()
    for point, neighbors in zip(points, sqrdists_idx):
        neighbor_labels = xyz_labels[neighbors]
        diff_labels_count = (neighbor_labels != neighbor_labels[0]).sum()
        if diff_labels_count >= 0.4 * nsample:
            # TODO: 认为当前点是边界
            pointcloud_info += f'{point[0]} {point[1]} {point[2]} 255 0 0 255\n'
        else:
            pointcloud_info += f'{point[0]} {point[1]} {point[2]} 255 255 255 255\n'

    # 输出到文件中
    header = f"ply\n" \
             f"format ascii 1.0\n" \
             f"comment VCGLIB generated\n" \
             f"element vertex {n}\n" \
             f"property float x\n" \
             f"property float y\n" \
             f"property float z\n" \
             f"property uchar red\n" \
             f"property uchar green\n" \
             f"property uchar blue\n" \
             f"property uchar alpha\n" \
             f"end_header\n"
    with open(os.path.join(target_path, info['file_name'][:-4] + f'_{n}_edge.ply'), 'w', encoding='ascii') as f:
        f.write(header)
        f.write(pointcloud_info)
    return


# 进行一次fps并进行绘图
def fps_once(config, xyz, points, select_index, info, target_path, only_fps):
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
               os.path.join(target_path, info['file_name'][:-4] + f"_{config['npoint']}.ply"), only_fps)
    return xyz_new, points_new, select_index


# 输出4次FPS+ball query的效果图
def visualize(points, labels, select_index, info, target_path, only_fps=False):
    points = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    xyz = points[:, :, :3]
    select_index = np.array(select_index)

    # TODO: ground truth的边缘点云
    output_edge(xyz, labels, select_index, 0.1, 32, info, target_path)

    config_1 = {
        "npoint": 8000,
        "radius": 0.1,
        "nsample": 32
    }
    xyz_1, points_1, select_index = fps_once(config_1, xyz, points, select_index, info, target_path, only_fps)

    # TODO: 一次FPS的边缘点云
    output_edge(xyz_1, labels, select_index, config_1['radius'], config_1['nsample'], info, target_path)

    config_2 = {
        "npoint": 6000,
        "radius": 0.2,
        "nsample": 16
    }
    xyz_2, points_2, select_index = fps_once(config_2, xyz_1, points_1, select_index, info, target_path, only_fps)

    # TODO: 二次FPS的边缘点云
    output_edge(xyz_2, labels, select_index, config_2['radius'], config_2['nsample'], info, target_path)

    config_3 = {
        "npoint": 4000,
        "radius": 0.4,
        "nsample": 8
    }
    xyz_3, points_3, select_index = fps_once(config_3, xyz_2, points_2, select_index, info, target_path, only_fps)

    # TODO: 三次FPS的边缘点云
    output_edge(xyz_3, labels, select_index, config_3['radius'], config_3['nsample'], info, target_path)

    config_4 = {
        "npoint": 2000,
        "radius": 0.6,
        "nsample": 4
    }
    xyz_4, points_4, select_index = fps_once(config_4, xyz_3, points_3, select_index, info, target_path, only_fps)

    # TODO: 四次FPS的边缘点云
    output_edge(xyz_4, labels, select_index, config_4['radius'], config_4['nsample'], info, target_path)


if __name__ == '__main__':
    file_path = 'D:\\Dataset\\OralScan\\dataset_labelled_cell_color_downsampled_10000_npy'
    target_path = "D:\\Dataset\\OralScan\\visualize_fps_edge_detect"

    dataset = TrimLineDataloader(data_root=file_path, total_point=10000, num_point=10000, transform=None,
                                 is_train=False, return_info=True)

    points, labels, select_index, info = dataset[66]
    visualize(points, labels, select_index, info, target_path, only_fps=False)

    # for (points, labels, select_index, info) in dataset:
    #    visualize(points, labels, select_index, info, target_path)
