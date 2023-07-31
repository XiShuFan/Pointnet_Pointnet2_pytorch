import os
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset


class TrimLineDataloader(Dataset):
    def __init__(self, data_root, num_point=150000, transform=None):
        super().__init__()
        self.data_root = data_root
        self.num_point = num_point
        self.transform = transform

        # 得到文件夹下所有的文件
        self.file_list = os.listdir(self.data_root)

        # 标签的权重值，避免不平衡的标签数量
        labelweights = np.zeros(2)

        # 统计面片数量的最大最小值
        self.min_faces = 1e10
        self.max_faces = 0

        for file in tqdm(self.file_list, total=len(self.file_list)):
            tooth_path = os.path.join(self.data_root, file)
            # 这里记录了面片信息和顶点信息，注意需要normalize之后才能输入到网络中
            info = np.load(tooth_path, allow_pickle=True)
            info = dict(info)
            # 得到面片类别
            labels = np.array(info['labels'])

            # 统计标签不平衡数目
            labelweights[0] += (labels == 0).sum()
            labelweights[1] += (labels == 1).sum()

            # 统计面片个数最大最小值
            self.min_faces = min(self.min_faces, info['face_num'])
            self.max_faces = max(self.max_faces, info['face_num'])

        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        self.labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)
        print('标签权重', self.labelweights)
        print('面片数量范围', self.min_faces, self.max_faces)

    def __getitem__(self, idx):
        tooth_path = os.path.join(self.data_root, self.file_list[idx])
        info = np.load(tooth_path, allow_pickle=True)
        info = dict(info)

        # 拿出口扫的信息
        face_num = info['face_num']
        vertices = info['vertices']
        faces = info['faces']
        vertex_norms = info['vertex_norms']
        face_norms = info['face_norms']
        vertex_colors = info['vertex_colors']
        # TODO: 注意这里面片上并没有颜色，我们需要从顶点颜色上计算
        face_colors = info['face_colors']
        labels = np.array(info['labels'], dtype=int)

        # 计算得到顶点的范围，为了归一化
        coord_min, coord_max = np.amin(vertices, axis=0)[:3], np.amax(vertices, axis=0)[:3]

        # TODO: 这里不方便做下采样，直接取所有的面片，不够的用0填充
        pad_num = self.num_point - info['face_num']

        # 顶点坐标归一化
        vertices[:, 0] = (vertices[:, 0] - coord_min[0]) / (coord_max[0] - coord_min[0])
        vertices[:, 1] = (vertices[:, 1] - coord_min[1]) / (coord_max[1] - coord_min[1])
        vertices[:, 2] = (vertices[:, 2] - coord_min[2]) / (coord_max[2] - coord_min[2])

        # 得到面片中心点坐标
        face_centers = [np.mean(vertices[face], axis=0) for face in faces]
        face_centers = np.array(face_centers)

        # 得到面片的法向量，注意归一化
        # 计算法向量的长度
        face_norms_len = np.linalg.norm(face_norms, axis=1)
        face_norms = face_norms / face_norms_len[:, np.newaxis]

        # 得到面片的颜色信息，注意归一化
        face_colors = [np.mean(vertex_colors[face], axis=0) for face in faces]
        face_colors = np.array(face_colors)[:, :3]
        face_colors /= 255

        # TODO: 使用的信息包括：面片中心点坐标、三个顶点的坐标、面片法向量、面片颜色rbg
        # 把面片中心放在第一位是为了方便计算邻居
        current_points = np.zeros((self.num_point, 18))
        current_points[:face_num, 0:3] = face_centers
        current_points[:face_num, 3:6] = vertices[faces[:, 0]]
        current_points[:face_num, 6:9] = vertices[faces[:, 1]]
        current_points[:face_num, 9:12] = vertices[faces[:, 2]]
        current_points[:face_num, 12:15] = face_norms
        current_points[:face_num, 15:18] = face_colors

        # 标签值也一样
        current_labels = np.zeros((self.num_point))
        current_labels[:face_num] = labels

        # 随机打乱顺序
        permute = np.random.permutation(self.num_point)
        current_labels = current_labels[permute]
        current_points = current_points[permute]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)
        return current_points, current_labels

    def __len__(self):
        return len(self.file_list)


if __name__ == '__main__':
    dataset = TrimLineDataloader(data_root='D:\\Dataset\\OralScan_trim_line\\train')

    print(len(dataset))

    print(dataset[0])