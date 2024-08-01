import datetime
import os
import time
import argparse
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import FLcore.models
from FLcore.aggregation import FedAvg
from FLcore.client import Client
from FLcore.server import Server
from FLcore.utils import Bar, accuracy
from FLcore.meter import AverageMeter
from FLcore.dataloader import pmnist as pmd

# 随机数种子
_seed_ = 2022
# 设置Python 的随机数生成器的种子。这将确保随机数生成器生成的随机序列是可预测的
random.seed(_seed_)
# 设置NumPy的随机数生成器的种子。这将确保在使用NumPy进行随机操作时得到可重复的结果
np.random.seed(_seed_)
# 设置PyTorch的随机数生成器的种子。这将确保在使用PyTorch进行随机操作时得到可重复的结果
torch.random.manual_seed(_seed_)
# 设置所有可用的CUDA设备的随机数生成器的种子。这将确保在使用CUDA加速时得到可重复的结果
torch.cuda.manual_seed_all(_seed_)
# 将CuDNN的随机性设置为确定性模式。这将确保在使用CuDNN加速时得到可
torch.backends.cudnn.deterministic = True
# 禁用CuDNN的自动寻找最佳卷积算法。这将确保在使用CuDNN加速时得到可重复的结果。
torch.backends.cudnn.benchmark = False
# 设置PyTorch进行CPU多线程并行计算时所占用的线程数，用来限制PyTorch所占用的CPU数目
torch.set_num_threads(4)


def main(args):
    # 使用CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # 使用pmnist的dataloader
    data, taskcla, inputsize = pmd.get(data_dir=args.data_dir, seed=_seed_)

    xtrain, ytrain, xtest, ytest = {}, {}, {}, {}

    for task_id, ncla in taskcla:
        xtrain[task_id] = data[task_id]['train']['x']
        ytrain[task_id] = data[task_id]['train']['y']
        xtest[task_id] = data[task_id]['test']['x']
        ytest[task_id] = data[task_id]['test']['y']

    acc_matrix = np.zeros((10, 10))

    # 脉冲神经网络设置
    # 参照文章3.1中的公式1
    snn_setting = {}
    snn_setting['timesteps'] = args.timesteps
    snn_setting['train_Vth'] = True if args.train_Vth == 1 else False
    snn_setting['Vth'] = args.Vth
    snn_setting['tau'] = args.tau
    snn_setting['delta_t'] = args.delta_t
    snn_setting['alpha'] = args.alpha
    snn_setting['Vth_bound'] = args.Vth_bound
    snn_setting['rate_stat'] = True if args.rate_stat == 1 else False

    root_dir = "logs/1"
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    task_count = 0  # 已经完成的task的计数器
    task_list = []

    hlop_out_num = [80, 200, 100]
    hlop_out_num_inc = [70, 70, 70]

    # 使用脉冲进行hlop更新
    hlop_with_wfr = True
    if args.not_hlop_with_wfr:
        hlop_with_wfr = False

    # 生成服务器
    server = Server()
    # 服务器配置测试数据
    server.configure_testset_info(xtest, ytest)
    server.configure_data_save_path(root_dir)

    # 生成客户端
    clients = []
    for i in range(3):
        client = Client(i)
        client.configure_trainset_info(xtrain, ytrain)
        client.configure_data_save_path(root_dir, args)
        clients.append(client)

    for task_id, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(task_id, data[task_id]['name']))
        print('*' * 100)
        task_list.append(task_id)

        # 当任务id为0时候
        if task_count == 0:
            model = FLcore.models.spiking_MLP(snn_setting, num_classes=ncla, n_hidden=800,
                                              ss=args.sign_symmetric,
                                              fa=args.feedback_alignment, hlop_with_wfr=hlop_with_wfr,
                                              hlop_spiking=args.hlop_spiking,
                                              hlop_spiking_scale=args.hlop_spiking_scale,
                                              hlop_spiking_timesteps=args.hlop_spiking_timesteps)
            model.add_hlop_subspace(hlop_out_num)
            model = model.cuda()
            # 客户端配置网络模型
            for client in clients:
                client.configure_local_model(model)
            # 服务器配置网络模型
            server.configure_global_model(model)

        else:
            if task_count % 3 == 0:
                hlop_out_num_inc[0] -= 20
                hlop_out_num_inc[1] -= 20
                hlop_out_num_inc[2] -= 20
            # 客户端配置网络模型
            for client in clients:
                client.local_model.add_hlop_subspace(hlop_out_num_inc)
            server.global_model.add_hlop_subspace(hlop_out_num_inc)

        # 客户端训练
        for client in clients:
            client.train(task_id, ncla, args, False, False)
        # FedAvg算法聚合
        server.update_model_weight(FedAvg([client.commit_model_weight() for client in clients]))
        # 服务器测试
        server.evaluate(task_id, task_count, args, False, False)

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_count + 1]:
            _, acc_matrix[task_count, jj] = server.evaluate(ii, task_count, args, False, False)
            jj += 1
        print('Accuracies =')
        for i_a in range(task_count + 1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
            print()

        for client in clients:
            client.local_model.merge_hlop_subspace()
        server.global_model.merge_hlop_subspace()

        # 如果重放并且起码参与了一个任务
        if args.replay and task_count >= 1:
            # 客户端重放训练
            for client in clients:
                client.replay_train(task_id, ncla, args)
            # FedAvg算法聚合
            server.update_model_weight(FedAvg([client.commit_model_weight() for client in clients]))
            # 服务器测试
            server.evaluate(task_id, task_count, args, False, False)

            # 保存准确率
            jj = 0
            for ii in np.array(task_list)[0:task_count + 1]:
                _, acc_matrix[task_count, jj] = server.evaluate(ii, task_count, args, False, False)
                jj += 1
            print('Accuracies =')
            for i_a in range(task_count + 1):
                print('\t', end='')
                for j_a in range(acc_matrix.shape[1]):
                    print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                print()

        # 更新task的id
        task_count += 1

    # 实验结果部分 ----------------------------------------------------------------------------------------------------- #
    """
    有两个指标，一个平均准确率acc，一个后向迁移bwt
    """
    print('-' * 50)
    print('Task Order : {}'.format(np.array(task_list)))
    acc = acc_matrix[-1].mean()
    print('Final Avg Accuracy: {:5.2f}%'.format(acc * 100))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt * 100))
    # print('[Elapsed time = {:.1f} ms]'.format((time.time()-tstart)*1000))
    print('-' * 50)
    # Plots
    # array = acc_matrix
    # df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]],
    #                  columns = [i for i in ["T1","T2","T3","T4","T5","T6","T7","T8","T9","T10"]])
    # sn.set(font_scale=1.4)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify PMNIST')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-global_epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-local_epochs', default=10, type=int, metavar='N', help='number of local epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./dataset')
    # parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-lr_scheduler', default='StepLR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-warmup', default=0, type=int, help='warmup epochs for learning rate')
    parser.add_argument('-cnf', type=str)

    parser.add_argument('-hlop_start_epochs', default=0, type=int, help='the start epoch to update hlop')

    parser.add_argument('-sign_symmetric', action='store_true', help='sign symmetric')
    parser.add_argument('-feedback_alignment', action='store_true', help='feedback alignment')

    parser.add_argument('-baseline', action='store_true', help='baseline')

    parser.add_argument('-replay', action='store_true', help='replay few-shot previous tasks')
    parser.add_argument('-memory_size', default=50, type=int, help='memory size for replay')
    parser.add_argument('-replay_epochs', default=1, type=int, help='epochs for replay')
    parser.add_argument('-replay_b', default=50, type=int, help='batch size per task for replay')
    parser.add_argument('-replay_lr', default=0.01, type=float, help='learning rate for replay')
    parser.add_argument('-replay_T_max', default=20, type=int, help='T_max for CosineAnnealingLR for replay')

    parser.add_argument('-gpu-id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')
    # SNN settings
    parser.add_argument('-timesteps', default=20, type=int)
    parser.add_argument('-Vth', default=0.3, type=float)
    parser.add_argument('-tau', default=1.0, type=float)
    parser.add_argument('-delta_t', default=0.05, type=float)
    parser.add_argument('-alpha', default=0.3, type=float)
    parser.add_argument('-train_Vth', default=1, type=int)
    parser.add_argument('-Vth_bound', default=0.0005, type=float)
    parser.add_argument('-rate_stat', default=0, type=int)

    parser.add_argument('-not_hlop_with_wfr', action='store_true', help='use spikes for hlop update')
    parser.add_argument('-hlop_spiking', action='store_true', help='use hlop with lateral spiking neurons')
    parser.add_argument('-hlop_spiking_scale', default=20., type=float)
    parser.add_argument('-hlop_spiking_timesteps', default=1000., type=float)

    args = parser.parse_args()
    main(args)
