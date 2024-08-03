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
from FLcore import models
from FLcore.aggregation import FedAvg
from FLcore.client import Client
from FLcore.server import Server
from FLcore.utils import Bar, accuracy
from FLcore.meter import AverageMeter
from FLcore.dataloader import miniimagenet as data_loader

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
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    dataloader = data_loader.DatasetGen(data_dir=args.data_dir, seed=_seed_)
    taskcla, inputsize = dataloader.taskcla, dataloader.inputsize

    task_names = {}  # 任务的名称
    xtrain, ytrain = {}, {}  # 训练集
    xtest, ytest = {}, {}  # 测试集

    for task_id, ncla in taskcla:
        data = dataloader.get(task_id)
        task_names[task_id] = data[task_id]['name']
        xtrain[task_id] = data[task_id]['train']['x']
        ytrain[task_id] = data[task_id]['train']['y']
        xtest[task_id] = data[task_id]['test']['x']
        ytest[task_id] = data[task_id]['test']['y']

    acc_matrix = np.zeros((20, 20))

    # resnet18 nf 20
    hlop_out_num = [24, [90, 90], [90, 90], [90, 180, 10], [180, 180], [180, 360, 20], [360, 360], [360, 720, 40],
                    [720, 720]]

    hlop_out_num_inc1 = [2, [6, 6], [6, 6], [6, 12, 1], [12, 12], [12, 24, 2], [24, 24], [24, 48, 4], [48, 48]]
    hlop_out_num_inc2 = [0, [2, 2], [2, 2], [2, 4, 0], [4, 4], [4, 8, 0], [8, 8], [8, 16, 0], [16, 16]]

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

    # 生成服务器
    server = Server()
    # 服务器配置测试数据
    server.configure_testset(xtest, ytest)
    server.configure_data_save_path(root_dir)

    # 生成客户端
    clients = []
    for i in range(args.num_clients):
        client = Client(i)
        client.configure_trainset(xtrain, ytrain)
        client.configure_data_save_path(root_dir, args)
        clients.append(client)

    task_count = 0
    task_list = []

    for task_id, ncla in taskcla:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(task_id, task_names[task_id]))
        print('*' * 100)
        task_list.append(task_id)

        writer = SummaryWriter(os.path.join(root_dir, 'logs_task{task_id}'.format(task_id=task_id)))

        if args.replay:
            for client in clients:
                client.replay_xtrain[task_id], client.replay_ytrain[task_id] = [], []
                for c in range(ncla):
                    num = args.memory_size
                    index = 0
                    while num > 0:
                        if ytrain[index] == c:
                            client.replay_xtrain[task_id].append(xtrain[index])
                            client.replay_ytrain[task_id].append(ytrain[index])
                            num -= 1
                        index += 1
                client.replay_xtrain[task_id] = torch.stack(client.replay_xtrain[task_id], dim=0)
                client.replay_ytrain[task_id] = torch.stack(client.replay_ytrain[task_id], dim=0)

        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False

        if task_count == 0:
            model = models.spiking_resnet18(snn_setting, num_classes=ncla, nf=20, ss=args.sign_symmetric,
                                            hlop_with_wfr=hlop_with_wfr, hlop_spiking=args.hlop_spiking,
                                            hlop_spiking_scale=args.hlop_spiking_scale,
                                            hlop_spiking_timesteps=args.hlop_spiking_timesteps,
                                            proj_type=args.hlop_proj_type, first_conv_stride2=True)
            model.add_hlop_subspace(hlop_out_num)
            model = model.cuda()
            # 客户端配置网络模型
            for client in clients:
                client.configure_local_model(model)
            # 服务器配置网络模型
            server.configure_global_model(model)
        else:
            # 客户端配置网络模型
            for client in clients:
                client.local_model.add_classifier(ncla)
            server.global_model.add_classifier(ncla)
            if task_count < 6:
                # 客户端配置网络模型
                for client in clients:
                    client.local_model.add_hlop_subspace(hlop_out_num_inc1)
                server.global_model.add_hlop_subspace(hlop_out_num_inc1)
            else:
                # 客户端配置网络模型
                for client in clients:
                    client.local_model.add_hlop_subspace(hlop_out_num_inc2)
                server.global_model.add_hlop_subspace(hlop_out_num_inc2)

        for client in clients:
            # 配置优化器 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            params = []
            for name, p in client.local_model.named_parameters():
                if 'hlop' not in name:
                    if task_id != 0:
                        if len(p.size()) != 1:
                            params.append(p)
                    else:
                        params.append(p)
            if args.opt == 'SGD':
                if task_id == 0:
                    client.optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)
                else:
                    client.optimizer = torch.optim.SGD(params, lr=args.lr_continual, momentum=args.momentum)
            elif args.opt == 'Adam':
                if task_id == 0:
                    client.optimizer = torch.optim.Adam(params, lr=args.lr)
                else:
                    client.optimizer = torch.optim.Adam(params, lr=args.lr_continual)
            else:
                raise NotImplementedError(args.opt)
            # 配置优化器 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # 配置学习率调节器 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            if args.lr_scheduler == 'StepLR':
                client.lr_scheduler = torch.optim.lr_scheduler.StepLR(client.optimizer, step_size=args.step_size, gamma=args.gamma)
            elif args.lr_scheduler == 'CosALR':
                # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
                lr_lambda = lambda cur_epoch: (cur_epoch + 1) / args.warmup if cur_epoch < args.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - args.warmup) / (args.T_max - args.warmup) * math.pi))
                client.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(client.optimizer, lr_lambda=lr_lambda)
            else:
                raise NotImplementedError(args.lr_scheduler)
            # 配置学习率调节器 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

        for global_epoch in range(args.global_epochs):
            for local_epoch in range(1, args.local_epochs + 1):
                start_time = time.time()

                train_losses, train_acces = [], []
                for client in clients:
                    train_loss, train_acc = client.train_epoch(task_id, local_epoch, args, False, False)
                    train_losses.append(train_loss)
                    train_acces.append(train_acc)
                    writer.add_scalar(f'client{client.id}-train_loss', train_loss, local_epoch)
                    writer.add_scalar(f'client{client.id}-train_acc', train_acc, local_epoch)

                # FedAvg算法聚合
                server.update_model_weight(FedAvg([client.commit_model_weight() for client in clients]))
                test_loss, test_acc = server.evaluate(task_id, args, False, False)
                writer.add_scalar(f'server-test_loss', test_loss, local_epoch)
                writer.add_scalar(f'server-test_acc', test_acc, local_epoch)
                total_time = time.time() - start_time
                print(
                    f'epoch={local_epoch}, train_loss={np.mean(train_losses)}, train_acc={np.mean(train_acces)}, test_loss={test_loss}, test_acc={test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now() + datetime.timedelta(seconds=total_time * (args.local_epochs - local_epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_count + 1]:
            _, acc_matrix[task_count, jj] = server.evaluate(ii, args, False, False)
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

        if args.replay and task_count >= 1:
            print('memory replay\n')
            for client in clients:
                # 配置重放优化器 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                params = []
                for name, p in client.local_model.named_parameters():
                    if 'hlop' not in name:
                        if task_id != 0:
                            if len(p.size()) != 1:
                                params.append(p)
                        else:
                            params.append(p)
                if args.opt == 'SGD':
                    client.optimizer = torch.optim.SGD(params, lr=args.replay_lr, momentum=args.momentum)
                elif args.opt == 'Adam':
                    client.optimizer = torch.optim.Adam(params, lr=args.replay_lr)
                else:
                    raise NotImplementedError(args.opt)
                # 配置优化器 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                # 配置重放学习率调节器 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
                if args.lr_scheduler == 'StepLR':
                    client.lr_scheduler = torch.optim.lr_scheduler.StepLR(client.optimizer, step_size=args.step_size,
                                                                   gamma=args.gamma)
                elif args.lr_scheduler == 'CosALR':
                    client.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(client.optimizer, T_max=args.replay_T_max)
                else:
                    raise NotImplementedError(args.lr_scheduler)
                # 配置重放学习率调节器 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            for replay_epoch in range(1, args.replay_epochs + 1):
                for client in clients:
                    client.replay_train_epoch(task_id, args)
                # FedAvg算法聚合
                server.update_model_weight(FedAvg([client.commit_model_weight() for client in clients]))

            # save accuracy
            jj = 0
            for ii in np.array(task_list)[0:task_count + 1]:
                _, acc_matrix[task_count, jj] = server.evaluate(ii, args, False, False)
                jj += 1
            print('Accuracies =')
            for i_a in range(task_count + 1):
                print('\t', end='')
                for j_a in range(acc_matrix.shape[1]):
                    print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                print()

        # update task id
        task_count += 1

    print('-' * 50)
    # Simulation Results
    print('Task Order : {}'.format(np.array(task_list)))
    print('Final Avg Accuracy: {:5.2f}%'.format(acc_matrix[-1].mean() * 100))
    bwt = np.mean((acc_matrix[-1] - np.diag(acc_matrix))[:-1])
    print('Backward transfer: {:5.2f}%'.format(bwt * 100))
    print('-' * 50)
    # Plots
    # array = acc_matrix
    # df_cm = pd.DataFrame(array, index = [i for i in ["T1","T2","T3","T4","T5"]],
    #                  columns = [i for i in ["T1","T2","T3","T4","T5"]])
    # sn.set(font_scale=1.4)
    # sn.heatmap(df_cm, annot=True, annot_kws={"size": 10})
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify miniimagenet')
    parser.add_argument('-b', default=64, type=int, help='batch size')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-lr_continual', default=0.01, type=float, help='learning rate for continual tasks')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=100, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-warmup', default=0, type=int, help='warmup epochs for learning rate')
    parser.add_argument('-cnf', type=str)

    parser.add_argument('-hlop_start_epochs', default=0, type=int, help='the start epoch to update hlop')
    parser.add_argument('-hlop_proj_type', type=str,
                        help='choice for projection type in bottom implementation, default is input, can choose weight for acceleration of convolutional operations',
                        default='input')
    parser.add_argument('-hlop_iteration', default=5, type=int, help='the number of iterations for hlop update')

    parser.add_argument('-replay', action='store_true', help='replay few-shot previous tasks')
    parser.add_argument('-memory_size', default=5, type=int, help='memory size for replay')
    parser.add_argument('-replay_epochs', default=20, type=int, help='epochs for replay')
    parser.add_argument('-replay_b', default=5, type=int, help='batch size per task for replay')
    parser.add_argument('-replay_lr', default=0.001, type=float, help='learning rate for replay')
    parser.add_argument('-replay_T_max', default=20, type=int, help='T_max for CosineAnnealingLR for replay')

    parser.add_argument('-sign_symmetric', action='store_true', help='use sign symmetric')
    parser.add_argument('-baseline', action='store_true', help='baseline')

    parser.add_argument('-gpu-id', default='0', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
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

    parser.add_argument('-num_clients', default=3, type=int, help='number of clients')
    parser.add_argument('-global_epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-local_epochs', default=10, type=int, metavar='N', help='number of local epochs to run')

    args = parser.parse_args()

    main(args)
