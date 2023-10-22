"""
这个脚本的作用是将ply文件的信息输出到npy文件中
"""
import os

import numpy as np
import vedo
import vtk

def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    为了得到足够多的训练数据，对vtp数据进行随机旋转、平移、缩放
    get transformation matrix (4*4)
    return: vtkMatrix4x4
    '''
    Trans = vtk.vtkTransform()

    ry_flag = np.random.randint(0, 2)  # if 0, no rotate
    rx_flag = np.random.randint(0, 2)  # if 0, no rotate
    rz_flag = np.random.randint(0, 2)  # if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0, 2)  # if 0, no translate
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    scale_flag = np.random.randint(0, 2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])

    matrix = Trans.GetMatrix()

    return matrix


def data_augment(source_path, augment_path):
    # 输出数据增强之后的路径
    if not os.path.exists(augment_path):
        os.mkdir(augment_path)

    sample_list = os.listdir(source_path)

    # 对每一个数据，做随机刚性变换
    num_augmentations = 10

    for sample in sample_list:
        if 'INTER' in sample:
            continue
        print(sample)
        for aug in range(num_augmentations):
            output_file_name = sample.split('.')[0] + f'_{aug}.ply'
            vtk_matrix = GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                                                    translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                                                    scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2])
            mesh = vedo.load(os.path.join(source_path, sample))

            mesh.apply_transform(vtk_matrix)
            vedo.io.write(mesh, os.path.join(augment_path, output_file_name), binary=False)

def ply_to_npy(augment_path, target_path, is_precision=False):
    """

    Args:
        augment_path:
        target_path:
        is_precision: 是否是精度分割测试

    Returns:

    """
    min_ncells = 1e9
    max_ncells = 0

    # TODO: 统计一下精度分割测试的面片数量
    precision_cells = []

    for file in os.listdir(augment_path):
        if 'INTER' in file:
            continue
        print(file)
        mesh = vedo.load(os.path.join(augment_path, file))

        info = {
            'face_num': mesh.ncells,
            'vertices': mesh.points(),
            'faces': np.array(mesh.cells()),
            'vertex_norms': mesh.normals(cells=False, recompute=True),
            'face_norms': mesh.normals(cells=True, recompute=True),
            'vertex_colors': mesh.pointcolors,
            'face_colors': mesh.cellcolors
        }

        min_ncells = min(mesh.ncells, min_ncells)
        max_ncells = max(mesh.ncells, max_ncells)

        # 注意还要添加一下面片的label
        if is_precision:
            # TODO: 黄色是需要分割的label
            labels = np.all(mesh.cellcolors == np.array([255, 255, 0, 255]), axis=1, keepdims=False)
            precision_cells.append(labels.sum())
        else:
            # TODO: 转换成颜色不是白色的为label
            labels = np.any(mesh.cellcolors != np.array([255, 255, 255, 255]), axis=1, keepdims=False)

        info['labels'] = labels

        np.save(os.path.join(target_path, os.path.basename(file)[:-4] + ".npy"), info)

    print(f'min_ncells {min_ncells}, max_ncells {max_ncells}')

    precision_cells.sort()
    print(f'精度分割面片数量：{precision_cells}')


def main():
    source_path = "D:\\Dataset\\OralScan_trim_line\\visualize_ply_expand_0_outof_6_selective_downsample_20000"
    augment_path = "D:\\Dataset\\OralScan_trim_line\\visualize_ply_expand_0_outof_6_selective_downsample_20000_aug"
    target_path = "D:\\Dataset\\OralScan_trim_line\\visualize_ply_expand_0_outof_6_selective_downsample_20000_aug_npy"

    # data_augment(source_path, augment_path)
    ply_to_npy(augment_path, target_path, is_precision=True)


if __name__ == '__main__':
    # 包括数据增强以及
    # 将数据转换成npy格式
    main()