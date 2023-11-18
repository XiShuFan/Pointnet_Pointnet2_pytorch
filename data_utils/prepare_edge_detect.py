"""
这个脚本用来准备牙齿多类别边缘检测数据集
"""

import os

import numpy as np
import vedo
import vtk
from color_label_utils import color2label

def ply_to_npy(source_path, target_path):

    for folder in os.listdir(source_path):
        for file in os.listdir(os.path.join(source_path, folder)):
            if 'labeled' not in file:
                continue
            print(file)
            mesh = vedo.load(os.path.join(source_path, folder, file))

            info = {
                'face_num': mesh.ncells,
                'vertices': mesh.points(),
                'faces': np.array(mesh.cells()),
                'vertex_norms': mesh.normals(cells=False, recompute=True),
                'face_norms': mesh.normals(cells=True, recompute=True),
                'vertex_colors': mesh.pointcolors,
                'face_colors': mesh.cellcolors
            }

            # 把label解析出来
            labels = np.zeros(mesh.ncells)
            for id, color in enumerate(mesh.cellcolors):
                color = (color[0], color[1], color[2])
                if color in color2label:
                    labels[id] = color2label[color][2]

            info['labels'] = labels

            np.save(os.path.join(target_path, folder + "_" + os.path.basename(file)[:-4] + ".npy"), info)


def main():
    source_path = "D:\\Dataset\\OralScan\\dataset_labelled_cell_color_downsampled_10000"
    target_path = "D:\\Dataset\\OralScan\\dataset_labelled_cell_color_downsampled_10000_npy"

    ply_to_npy(source_path, target_path)


if __name__ == '__main__':
    # 将数据转换成npy格式
    main()