from visdom import Visdom
import numpy as np


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
    def plot(self, var_name, split_name, title_name, x, y):
        opts = dict(
            legend=[split_name],
            title=title_name,
            xlabel='训练轮数',
            ylabel=var_name,
            xtickstep=10,
            ytickstep=0.05,
            width=800,
            height=800
        )

        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=opts)
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')


if __name__ == '__main__':
    file_path = "log/sem_seg/实验11/logs/pointnet2_sem_seg.txt"
    train_acc = []
    val_acc = []
    trim_line_iou = []
    other_iou = []
    mean_iou = []
    with open(file_path, "r") as file:
        for line in file:
            # 在这里对每一行进行解析
            line = line.strip()
            if 'Training accuracy: ' in line:
                search_str = 'Training accuracy: '
                index = line.find(search_str)

                if index != -1:
                    result = line[index + len(search_str):].strip()
                    train_acc.append(float(result))
            if 'eval point accuracy: ' in line:
                search_str = 'eval point accuracy: '
                index = line.find(search_str)

                if index != -1:
                    result = line[index + len(search_str):].strip()
                    val_acc.append(float(result))
            if 'eval mIoU: ' in line:
                search_str = 'eval mIoU: '
                index = line.find(search_str)

                if index != -1:
                    result = line[index + len(search_str):].strip()
                    mean_iou.append(float(result))
            if 'class others' in line:
                search_str = 'IoU: '
                index = line.find(search_str)

                if index != -1:
                    result = line[index + len(search_str):].strip()
                    other_iou.append(float(result))
            if 'class trim_line' in line:
                search_str = 'IoU: '
                index = line.find(search_str)

                if index != -1:
                    result = line[index + len(search_str):].strip()
                    trim_line_iou.append(float(result))

    print(len(train_acc), len(val_acc), len(other_iou), len(trim_line_iou), len(mean_iou))
    # TODO: 画出中文的结果
    plotter = VisdomLinePlotter(env_name='PointNet2')
    for epoch in range(len(train_acc)):
        plotter.plot('acc', '训练', '准确率', epoch, train_acc[epoch])
        if epoch % 5 == 0:
            plotter.plot('acc', '测试', '准确率', epoch, val_acc[epoch // 5])
            plotter.plot('iou', '龈缘线', '交并比', epoch, trim_line_iou[epoch // 5])
            plotter.plot('iou', '其他区域', '交并比', epoch, other_iou[epoch // 5])
            plotter.plot('iou', '总模型', '交并比', epoch, mean_iou[epoch // 5])
