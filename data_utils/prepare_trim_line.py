"""
这个脚本的作用是将ply文件的信息输出到npy文件中
"""
import os

import numpy as np
import vedo

source_path = "D:\\Dataset\\OralScan_trim_line\\visualize_ply_expand_3_selective_downsample_20000"
target_path = "D:\\Dataset\\OralScan_trim_line\\visualize_ply_expand_3_selective_downsample_20000_npy"

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

    # 注意还要添加一下面片的label，颜色为红色的是label
    # TODO: 转换成颜色不是白色的为label
    labels = np.any(mesh.cellcolors != np.array([255, 255, 255, 255]), axis=1, keepdims=False)

    info['labels'] = labels

    np.save(os.path.join(target_path, os.path.basename(file)[:-4] + ".npy"), info)
