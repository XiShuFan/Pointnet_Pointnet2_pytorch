"""
这个脚本的作用是将ply文件的信息输出到npy文件中
"""
import os

import numpy as np
import vedo

source_path = "D:\\Dataset\\OralScan_trim_line\\visualize_ply_expand_3_selective_downsample_40000"
target_path = "D:\\Dataset\\OralScan_trim_line\\visualize_ply_expand_3_selective_downsample_40000_npy"

min_ncells = 1e9
max_ncells = 0

for file in os.listdir(source_path):
    if 'INTER' in file:
        continue
    print(file)
    mesh = vedo.load(os.path.join(source_path, file))

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

    # 注意还要添加一下面片的label，颜色为红色的是label
    # TODO: 转换成颜色不是白色的为label
    labels = np.any(mesh.cellcolors != np.array([255, 255, 255, 255]), axis=1, keepdims=False)

    info['labels'] = labels

    np.save(os.path.join(target_path, os.path.basename(file)[:-4] + ".npy"), info)

print(f'min_ncells {min_ncells}, max_ncells {max_ncells}')