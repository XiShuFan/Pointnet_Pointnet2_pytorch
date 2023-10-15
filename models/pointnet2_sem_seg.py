import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation


class get_model(nn.Module):
    def __init__(self, num_classes, channel):
        super(get_model, self).__init__()
        # TODO: 这里group all的操作是将所有点聚合成一个点，可以试试
        self.sa1 = PointNetSetAbstraction(2000, 0.1, 16, channel + 3, [16, 16, 32], False)
        self.sa2 = PointNetSetAbstraction(500, 0.2, 8, 32 + 3, [32, 32, 64], False)
        self.sa3 = PointNetSetAbstraction(125, 0.4, 8, 64 + 3, [64, 64, 128], False)
        self.sa4 = PointNetSetAbstraction(25, 0.8, 8, 128 + 3, [128, 128, 256], False)
        self.fp4 = PointNetFeaturePropagation(384, [256, 256])
        self.fp3 = PointNetFeaturePropagation(320, [256, 256])
        self.fp2 = PointNetFeaturePropagation(288, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 64])
        self.conv1 = nn.Conv1d(64, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(32, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        # TODO: 这里没明白为什么需要 log_softmax，算出来都是负值
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


if __name__ == '__main__':
    import torch

    model = get_model(13, 15)
    xyz = torch.rand(6, 15, 2048)
    (model(xyz))
