# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cv2
import math
import numpy as np
import torch
import transforms3d
import random
import open3d as o3d


class Compose(object):
    def __init__(self, transforms):
        
        self.transforms = transforms

    def __call__(self, ptcloud, gtcloud):
        for t in self.transforms:
            ptcloud, gtcloud = t(ptcloud, gtcloud)
        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class ToTensor(object):
    """
    将numpy转化为Tensor
    
    Returns:
    """
    def __init__(self):
        pass

    def __call__(self, ptcloud, gtcloud):

        # ptcloud
        shape = ptcloud.shape
        if len(shape) == 3:  # RGB/Depth Images
            ptcloud = ptcloud.transpose(2, 0, 1)
        ptcloud = torch.from_numpy(ptcloud.copy()).float()

        # gtcloud
        shape = gtcloud.shape
        if len(shape) == 3:  # RGB/Depth Images
            gtcloud = gtcloud.transpose(2, 0, 1)
        gtcloud = torch.from_numpy(gtcloud.copy()).float()

        # Ref: https://discuss.pytorch.org/t/torch-from-numpy-not-support-negative-strides/3663/2
        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    

class Identity(object):
    
    def __init__(self) -> None:
        pass
    
    def __call__(self, ptcloud, gtcloud):
        return ptcloud, gtcloud
    

class ZeroCenter(object):
    """
    将中心坐标放入(0, 0, 0)
    """
    def __init__(self) -> None:
        pass

    def __call__(self, ptcloud, gtcloud):

        ptcloud = ptcloud - np.mean(ptcloud, axis=1, keepdims=True)
        gtcloud = gtcloud - np.mean(gtcloud, axis=1, keepdims=True)
        
        return ptcloud, gtcloud


class Normalize(object):
    """
    归一化
    
    Params:
        mean: 
        std: 
    Returns:
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, ptcloud, gtcloud):

        # ptcloud
        ptcloud = ptcloud.astype(np.float32)
        ptcloud /= self.std
        ptcloud -= self.mean

        # gtcloud
        gtcloud = gtcloud.astype(np.float32)
        gtcloud /= self.std
        gtcloud -= self.mean

        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class RandomSamplePoints(object):
    """
    随机采样点云
    
    Params:
        n_points:
    Returns:
    """
    def __init__(self, n_points):
        self.n_points = n_points

    def __call__(self, ptcloud, gtcloud):
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[: self.n_points]]

        while ptcloud.shape[0] < self.n_points:
            # zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            # ptcloud = np.concatenate([ptcloud, zeros])
            copy = ptcloud[:(self.n_points - ptcloud.shape[0]) % ptcloud.shape[0], :]
            ptcloud = np.concatenate([ptcloud, copy])

        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_points={self.n_points})"


class RandomClipPoints(object):
    """
    随机裁剪点云(实际可以理解为概率丢弃,并不具有几何意义)
    
    Params:
        sigma:
        clip:
    Returns:
    """
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, ptcloud, gtcloud):

        ptcloud += np.clip(
            self.sigma * np.random.randn(*ptcloud.shape), -self.clip, self.clip
        ).astype(np.float32)

        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sigma={self.sigma}, clip={self.clip})"


class RandomRotatePoints(object):
    """
    随机旋转点云
    
    Params:
        axis:
    Returns:
    """
    def __init__(self, axis):
        
        self.axis = axis
        
        self.rnd_value = np.random.uniform(0, 1)

    def __call__(self, ptcloud, gtcloud):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        angle = 2 * math.pi * self.rnd_value
        
        # 沿指定轴和角度进行旋转
        if self.axis == 'x':
            trfm_mat = np.dot(transforms3d.axangles.axangle2mat(
                [1, 0, 0], angle), trfm_mat)
        elif self.axis == 'y':
            trfm_mat = np.dot(transforms3d.axangles.axangle2mat(
                [0, 1, 0], angle), trfm_mat)
        elif self.axis == 'z':
            trfm_mat = np.dot(transforms3d.axangles.axangle2mat(
                [0, 0, 1], angle), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

        gtcloud[:, :3] = np.dot(gtcloud[:, :3], trfm_mat.T)
        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis})"


class ScalePoints(object):
    """
    缩放点云
    
    Params:
        scale:
    Returns:
    """

    def __init__(self, scale):

        self.scale = scale

    def __call__(self, ptcloud, gtcloud):
        # 不做操作
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        # 生成缩放矩阵
        trfm_mat = np.dot(transforms3d.zooms.zfdir2mat(self.scale), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)
        gtcloud[:, :3] = np.dot(gtcloud[:, :3], trfm_mat.T)
        return ptcloud, gtcloud

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale})"


class RandomScalePoints(object):
    """
    随机缩放点云(本实验不能使用)
    
    Params:
        scale:
        rnd_value:
    Returns:
    """
    def __init__(self, scale, rnd_value):
        
        self.scale = scale
        self.rnd_value = rnd_value

    def __call__(self, ptcloud, gtcloud):
        # 不做操作
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        scale = np.random.uniform(
            1.0 / self.scale * self.rnd_value, self.scale * self.rnd_value)
        # 生成缩放矩阵
        trfm_mat = np.dot(transforms3d.zooms.zfdir2mat(scale), trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

        gtcloud[:, :3] = np.dot(gtcloud[:, :3], trfm_mat.T)

        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(scale={self.scale}, rnd_value={self.rnd_value})"


class RandomMirrorPoints(object):
    """
    对点云做随机镜像
    
    Returns:
    """
    def __init__(self):
        self.rnd_value = np.random.uniform(0, 1)

    def __call__(self, ptcloud, gtcloud):
        # 不做操作
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        
        # 沿x轴做镜像
        trfm_mat_x = np.dot(
            transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)  
        
        # 沿z轴做镜像
        trfm_mat_z = np.dot(
            transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
        
        if self.rnd_value <= 0.25:
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)
        elif self.rnd_value <= 0.5:  # lgtm [py/redundant-comparison]
            trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        elif self.rnd_value <= 0.75:
            trfm_mat = np.dot(trfm_mat_z, trfm_mat)

        ptcloud[:, :3] = np.dot(ptcloud[:, :3], trfm_mat.T)

        gtcloud[:, :3] = np.dot(gtcloud[:, :3], trfm_mat.T)

        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class RandomAxisMirrorPoints(object):
    """
    以一定概率沿指定轴对点云做镜像初始化
    
    Params:
        axis:
        p:
    Return:
    """
    
    def __init__(self, axis='x', p=0.5):
        
        self.axis = axis
        self.p = p
        
    def __call__(self, ptcloud, gtcloud):
        
        rnd_value = np.random.uniform(0, 1)
        if rnd_value > self.p:
            return ptcloud, gtcloud
        
        trfm_mat = transforms3d.zooms.zfdir2mat(1)

        if self.axis == 'x':
            # 沿x轴做镜像
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        elif self.axis == 'y':
            # 沿y轴做镜像
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), trfm_mat)
        elif self.axis == 'z':
            # 沿z轴做镜像
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)

        trfm_mat = np.dot(trfm_mat_x, trfm_mat)

        ptcloud = np.dot(ptcloud[:, :3], trfm_mat.T)

        gtcloud = np.dot(gtcloud[:, :3], trfm_mat.T)
        
        return ptcloud, gtcloud

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis}, p={self.p})"


class MirrorPoints(object):
    """
    对点云做镜像初始化
    
    Params:
        axis:
    Returns:
    """

    def __init__(self, axis='x'):
        
        self.axis = axis

    def __call__(self, ptcloud, gtcloud):
        trfm_mat = transforms3d.zooms.zfdir2mat(1)
        
        if self.axis == 'x':
            # 沿x轴做镜像
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
        elif self.axis == 'y':
            # 沿y轴做镜像
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [0, 1, 0]), trfm_mat)
        elif self.axis == 'z':
            # 沿z轴做镜像
            trfm_mat_x = np.dot(
                transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)
            
        trfm_mat = np.dot(trfm_mat_x, trfm_mat)

        ptcloud1 = np.dot(ptcloud[:, :3], trfm_mat.T)
        ptcloud2 = np.concatenate((ptcloud, ptcloud1), axis=0)

        gtcloud1 = np.dot(gtcloud[:, :3], trfm_mat.T)
        gtcloud2 = np.concatenate((gtcloud, gtcloud1), axis=0)

        return ptcloud2, gtcloud2
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(axis={self.axis})"


# class NormalizeObjectPose(object):
#     """
#     归一化目标姿势
    
#     Params:
#         input_keys:
#         ptcloud:
#         bbox:
#     Returns:
#     """
#     def __init__(self, input_keys, ptcloud, bbox):
#         input_keys = input_keys
#         self.ptcloud_key = ptcloud
#         self.bbox_key = bbox

#     def __call__(self, data):
#         ptcloud = data[self.ptcloud_key]
#         bbox = data[self.bbox_key]

#         # Calculate center, rotation and scale
#         # References:
#         # - https://github.com/wentaoyuan/pcn/blob/master/test_kitti.py#L40-L52
#         center = (bbox.min(0) + bbox.max(0)) / 2
#         bbox -= center
#         yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
#         rotation = np.array(
#             [[np.cos(yaw), -np.sin(yaw), 0],
#              [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
#         )
#         bbox = np.dot(bbox, rotation)
#         scale = bbox[3, 0] - bbox[0, 0]
#         bbox /= scale
#         ptcloud = np.dot(ptcloud - center, rotation) / scale
#         ptcloud = np.dot(ptcloud, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

#         data[self.ptcloud_key] = ptcloud
#         return data


class RandomCrop(object):
    """
    给定中心点和半径，对点云进行随机裁剪
    
    Params:
        min_radius: 可选的最小半径
        max_radius: 可选的最大半径
    Returns:
    """
    def __init__(self, min_radius, max_radius, p=0.0):
        
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.p = p
        
    def __call__(self, ptcloud, gtcloud):
        
        """
        ptcloud: N * 3
        gtcloud: N * 3
        return: 随机裁减的点云
        """
        
        rnd_value = np.random.uniform(0, 1)
        if rnd_value > self.p:
            return ptcloud, gtcloud
        
        # print(ptcloud.shape)
        N, _ = ptcloud.shape
        # 在所有点中随机挑选一个点
        point_ind = random.randint(0, N - 1)
        
        # print(self.min_radius, self.max_radius)
        
        # 在给定范围内随机取半径
        radius = random.uniform(self.min_radius, self.max_radius)
        # print(radius)
        
        centor_point = ptcloud[point_ind][np.newaxis, :]
        
        # 计算所有点与所选点的距离
        dists = np.sqrt(((ptcloud - centor_point) ** 2).sum(1))
        # print(dists)
        
        # 选择距离大于半径的点
        ind = np.where(dists > radius)[0]
        
        pcd_crop = ptcloud[ind]
        
        return pcd_crop, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_radius={self.min_radius}, max_radius={self.max_radius}, p={self.p})"


def farthest_point_sample(pts: np.array, num: int) -> np.array:
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    # pc1 = np.expand_dims(pts, axis=0)  # 1, N, 3
    pc1 = pts[None, :]
    batchsize, npts, dim = pc1.shape
    # centroids 是当前点集中最远点的坐标集合, shape (B, N) -> (1, 300)  300时要采样的点数
    centroids = np.zeros((batchsize, num), dtype=int)
    # 距离的shape是 (B, ndataset) ndataset是输入点的数量
    distance = np.ones((batchsize, npts)) * 1e10
    # 初始化的最远点, 随机选取id, 如果batchsize不是1, 那就选取 batchsize个,
    # farthest_id = np.random.randint(0, npts, (batchsize,), dtype=int)
    # batch_indices=[0,1,...,batchsize-1]

    barycenter = np.sum((pc1), 1)  # 计算重心坐标 及 距离重心最远的点
    barycenter = barycenter / pc1.shape[1]
    barycenter = barycenter.reshape(batchsize, 1, 3)

    dist = np.sum((pc1 - barycenter) ** 2, -1)
    # print(dist.shape)
    # print(np.argmax(dist, 1))
    farthest_id = np.argmax(dist, 1)[0]

    batch_index = np.arange(batchsize)
    for i in range(num):
        # 更新第i个最远点的id, 这里时所有的batch都同时更新, farthest的维度和 centroids[:, i]的维度相同
        centroids[:, i] = farthest_id
        # 取出这个最远点的xyz坐标, 按维度分别取 batch\\点的id\\点的坐标, 然后view变换维度
        centro_pt = pc1[batch_index, farthest_id, :].reshape(batchsize, 1, 3)
        # 计算点集中的所有点到这个最远点的欧式距离
        # 等价于torch.sum((xyz - centroid) ** 2, 2)
        dist = np.sum((pc1 - centro_pt) ** 2, -1)
        # 更新distances，记录样本中每个点距离所有已出现的采样点的最小距离
        mask = dist < distance
        # 从更新后的distances矩阵中找出距离最远的点，作为最远点用于下一轮迭代
        distance[mask] = dist[mask]
        farthest_id = np.argmax(distance[batch_index])
    # 返回采样点的id
    return centroids.squeeze(0)


class FPSRandomCrop(object):
    """
    最远点采样和随机裁剪结合
    
    Params:
        min_radius: 可选的最小半径
        max_radius: 可选的最大半径
        FPS_num: 使用FPS算法选择的中心点候选数
    Returns:
    """

    def __init__(self, min_radius, max_radius, FPS_num=1200, p=0.0):
        
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.FPS_num = FPS_num
        self.p = p

    def __call__(self, ptcloud, gtcloud):
        """
        ptcloud: N * 3
        gtcloud: N * 3
        return: 随机裁减的点云
        """
        
        rnd_value = np.random.uniform(0, 1)
        if rnd_value > self.p:
            return ptcloud, gtcloud
        
        FPS_point = farthest_point_sample(ptcloud, self.FPS_num)
        
        # 从FPS采样点中挑选其中一个点
        point_ind = random.randint(0, len(FPS_point))
        
        # 在给定范围内随机取半径
        radius = random.uniform(self.min_radius, self.max_radius)
        
        centor_point = ptcloud[FPS_point[point_ind]][np.newaxis, :]

        # 计算所有点与所选点的距离
        dists = np.sqrt(((ptcloud - centor_point) ** 2).sum(1))

        # 选择距离大于半径的点
        ind = np.where(dists > radius)[0]

        pcd_crop = ptcloud[ind]
        
        return pcd_crop, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min_radius={self.min_radius}, max_radius={self.max_radius}, FPS_num={self.FPS_num}, p={self.p})"


class RandomNoisePoints(object):
    """
    加随机噪声
    
    Params:
        noise_ratio:
    Returns:
    """
    def __init__(self, noise_ratio=None, p=0.1):
        
        self.noise_ratio = noise_ratio
        self.p = p

    def __call__(self, ptcloud, gtcloud):
        
        rnd_value = np.random.uniform(0, 1)
        if rnd_value > self.p:
            return ptcloud, gtcloud
        
        if self.noise_ratio is None:
            return ptcloud
        
        N, _ = ptcloud.shape
        
        noise_nums = int(N * self.noise_ratio) 
        
        noise_points = np.random.uniform(-1, 1,size=(noise_nums, 3))  
        
        noise_points[:, 0] = noise_points[:, 0] / \
            (max(ptcloud[:, 0]) - min(ptcloud[:, 0]))
        
        noise_points[:, 1] = noise_points[:, 1] / \
            (max(ptcloud[:, 1]) - min(ptcloud[:, 1]))
            
        noise_points[:, 2] = noise_points[:, 2] / \
            (max(ptcloud[:, 2]) - min(ptcloud[:, 2]))
        
        ptcloud = np.concatenate([ptcloud, noise_points], axis=0)
        
        return ptcloud, gtcloud
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(noise_ratio={self.noise_ratio}, p={self.p})"


if __name__ == '__main__':
    trfm_mat = transforms3d.zooms.zfdir2mat(0.5, [1, 0, 0])
    # print(trfm_mat)
    # print(RandomCrop(1, 2, 1))
    crop = FPSRandomCrop(100, 200, 20, 1)
    a = np.random.randn(100, 3)
    b = np.random.randn(100, 3)
    crop(a, b)
    # print(crop(a, b)[0].shape)
