"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from data_utils.TrimLineDataloader import TrimLineDataloader
from util import VisdomLinePlotter

# 绘图用
plotter = VisdomLinePlotter(env_name='PointNet2')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

# 分割的类别，我们只需要两类：牙龈线，其他区域
classes = ['others', 'trim_line']

# {others: 0, trim_line: 1}
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label

# {0: others, 1: trim_line}
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def parse_args():
    parser = argparse.ArgumentParser('Model')

    # 使用pointnet2的语义分割模型试试
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet_sem_seg]')

    # 这里如果不进行下采样的话，batch size可能得设置的很小，之后看看
    parser.add_argument('--batch_size', type=int, default=12, help='Batch Size during training [default: 16]')

    parser.add_argument('--test_batch_size', type=int, default=8, help='Batch Size during test')

    # 训练轮数
    parser.add_argument('--epoch', default=400, type=int, help='Epoch to run [default: 32]')
    # 学习率不动c
    parser.add_argument('--learning_rate', default=4e-4, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='1', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    # 日志存放目录
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')

    # 使用SGD优化器才会用到动量衰减
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')

    # TODO: 训练数据的面片数量
    parser.add_argument('--total_point', type=int, default=20000, help='Total points of data')

    # TODO: 采样点云数量，这个得统计一下
    parser.add_argument('--npoint', type=int, default=16000, help='Point Number [default: 4096]')

    # 学习率衰减
    parser.add_argument('--step_size', type=int, default=30, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Decay rate for lr decay [default: 0.7]')

    return parser.parse_args()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # 数据目录，要换成我们的牙龈线分割数据目录
    root = '/media/why/新加卷/xsf/Dataset/实验11/visualize_ply_expand_0_outof_9_selective_downsample_20000_v2_aug_npy'
    # 两类
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    TOTAL_NUM_POINT = args.total_point
    TEST_BATCH_SIZE = args.test_batch_size

    print("start loading training data ...")

    TRAIN_DATASET = TrimLineDataloader(data_root=root, total_point=TOTAL_NUM_POINT, num_point=NUM_POINT, transform=None, is_train=True, return_info=False)
    print("start loading test data ...")
    TEST_DATASET = TrimLineDataloader(data_root=root, total_point=TOTAL_NUM_POINT, num_point=NUM_POINT, transform=None, is_train=False, return_info=False)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    # TODO: 这里注意特征维度
    classifier = MODEL.get_model(NUM_CLASSES, channel=15).cuda()
    criterion = MODEL.get_loss().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    # TODO: 限制一下学习率最小值，否则后续训练不动了
    # TODO: 学习率太大了也会很波动，画图看看，权衡一下
    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            # TODO: 这里有一个数据增强，所有坐标信息和法向量都得做才行
            # TODO: 我们手动离线增强了，所以不需要继续增强，先排除这个变量
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points[:, :, 3:6] = provider.rotate_point_cloud_z(points[:, :, 3:6])
            points[:, :, 6:9] = provider.rotate_point_cloud_z(points[:, :, 6:9])
            points[:, :, 9:12] = provider.rotate_point_cloud_z(points[:, :, 9:12])
            points[:, :, 12:15] = provider.rotate_point_cloud_z(points[:, :, 12:15])

            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            # TODO: 这里将点云维度重构成了 [batch size, num channel, num points]
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        # TODO: 画出训练过程
        loss_sum = loss_sum.cpu().detach()
        plotter.plot('loss', 'train', 'Loss', epoch, loss_sum / num_batches)
        plotter.plot('accuracy', 'train', 'Acc', epoch, total_correct / float(total_seen))
        plotter.plot('learning_rate', 'train', 'Lr', epoch, lr)

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        # TODO: 控制验证的频率
        if epoch % 5 == 0:
            '''Evaluate on chopped scenes'''
            with torch.no_grad():
                num_batches = len(testDataLoader)
                total_correct = 0
                total_seen = 0
                loss_sum = 0
                labelweights = np.zeros(NUM_CLASSES)
                total_seen_class = [0 for _ in range(NUM_CLASSES)]
                total_correct_class = [0 for _ in range(NUM_CLASSES)]
                total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
                classifier = classifier.eval()

                log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
                for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                    points = points.data.numpy()
                    points = torch.Tensor(points)
                    points, target = points.float().cuda(), target.long().cuda()
                    points = points.transpose(2, 1)

                    seg_pred, trans_feat = classifier(points)
                    pred_val = seg_pred.contiguous().cpu().data.numpy()
                    seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                    batch_label = target.cpu().data.numpy()
                    target = target.view(-1, 1)[:, 0]
                    loss = criterion(seg_pred, target, trans_feat, weights)
                    loss_sum += loss
                    pred_val = np.argmax(pred_val, 2)
                    correct = np.sum((pred_val == batch_label))
                    total_correct += correct
                    total_seen += (TEST_BATCH_SIZE * TOTAL_NUM_POINT)
                    tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                    labelweights += tmp

                    for l in range(NUM_CLASSES):
                        total_seen_class[l] += np.sum((batch_label == l))
                        total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                        total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

                labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
                mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
                log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
                log_string('eval mIoU: %f' % (mIoU))
                log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
                log_string('eval point class accuracy: %f' % (
                    np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

                # TODO: 画出预测过程的结果
                loss_sum = loss_sum.cpu().detach()
                plotter.plot('loss', 'val', 'Loss', epoch, loss_sum / float(num_batches))
                plotter.plot('accuracy', 'val', 'Acc', epoch, total_correct / float(total_seen))
                plotter.plot('iou', 'mIoU', 'IoU', epoch, mIoU)

                iou_per_class_str = '------- IoU --------\n'
                for l in range(NUM_CLASSES):
                    iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                        seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                        total_correct_class[l] / float(total_iou_deno_class[l]))
                    # TODO: 画出每个类别的IoU
                    plotter.plot('iou', f'{seg_label_to_cat[l]}', 'IoU', epoch, total_correct_class[l] / float(total_iou_deno_class[l]))

                log_string(iou_per_class_str)


                if mIoU >= best_iou:
                    best_iou = mIoU
                    logger.info('Save model...')
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': mIoU,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                    log_string('Saving model....')
                log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
