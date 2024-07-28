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
from FLcore.utils import Bar, accuracy
from FLcore.meter import AverageMeter
from FLcore.dataloader import pmnist as pmd

# 实验环境准备 --------------------------------------------------------------------------------------------------------- #
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


def test(args, model, x, y, task_id):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=((x.size(0) - 1) // args.b + 1))

    test_loss = 0
    test_acc = 0
    num_test_samples = 0

    batch_idx = 0

    r = np.arange(x.size(0))
    with torch.no_grad():
        for i in range(0, len(r), args.b):
            if i + args.b <= len(r):
                index = r[i: i + args.b]
            else:
                index = r[i:]
            batch_idx += 1
            input = x[index].float().cuda()

            # repeat for time steps
            input = input.unsqueeze(1)
            input = input.repeat(1, args.timesteps, 1, 1, 1)

            label = y[index].cuda()

            out = model(input, task_id, projection=False, update_hlop=False)
            loss = F.cross_entropy(out, label)

            num_test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (out.argmax(1) == label).float().sum().item()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
            losses.update(loss, input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=batch_idx,
                size=((x.size(0) - 1) // args.b + 1),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
    bar.finish()

    test_loss /= num_test_samples
    test_acc /= num_test_samples

    return test_loss, test_acc


def main(args):
    # 使用 CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    # 使用pmnist的dataloader
    data, task_classes, inputsize = pmd.get(data_dir=args.data_dir, seed=_seed_)
    print("data:", data)
    print("task_classes:", task_classes)
    print("inputsize:", inputsize)
    acc_matrix = np.zeros((10, 10))

    # 脉冲神经网络设置 -------------------------------------------------------------------------------------------------- #
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

    # 生成输出文件夹 ---------------------------------------------------------------------------------------------------- #
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(out_dir)

    # 生成输出文件夹下子文件夹 models ------------------------------------------------------------------------------------- #
    pt_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)
        print(f'Mkdir {pt_dir}.')

    # 写入实验参数
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    #
    task_id = 0
    task_list = []
    replay_data = {}
    hlop_out_num = [80, 200, 100]
    hlop_out_num_inc = [70, 70, 70]

    for key_, num_classes in task_classes:
        print('*' * 100)
        print('Task {:2d} ({:s})'.format(key_, data[key_]['name']))
        print('*' * 100)

        writer = SummaryWriter(os.path.join(out_dir, 'logs_task{task_id}'.format(task_id=task_id)))
        # 训练集
        xtrain = data[key_]['train']['x']
        ytrain = data[key_]['train']['y']
        # 测试集
        xtest = data[key_]['test']['x']
        ytest = data[key_]['test']['y']
        task_list.append(key_)

        # 如果需要重放 ------------------------------------------------------------------------------------------------- #
        if args.replay:
            # save samples for memory replay
            replay_data[task_id] = {'x': [], 'y': []}
            for class_ in range(num_classes):
                num = args.memory_size
                index = 0
                while num > 0:
                    if ytrain[index] == class_:
                        replay_data[task_id]['x'].append(xtrain[index])
                        replay_data[task_id]['y'].append(ytrain[index])
                        num -= 1
                    index += 1
            replay_data[task_id]['x'] = torch.stack(replay_data[task_id]['x'], dim=0)
            replay_data[task_id]['y'] = torch.stack(replay_data[task_id]['y'], dim=0)

        # 使用脉冲进行hlop更新
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False

        # 当任务id为0时候
        if task_id == 0:
            model = FLcore.models.spiking_MLP(snn_setting, num_classes=num_classes, n_hidden=800, ss=args.sign_symmetric,
                                       fa=args.feedback_alignment, hlop_with_wfr=hlop_with_wfr,
                                       hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale,
                                       hlop_spiking_timesteps=args.hlop_spiking_timesteps)
            model.add_hlop_subspace(hlop_out_num)
            model = model.cuda()
        else:
            if task_id % 3 == 0:
                hlop_out_num_inc[0] -= 20
                hlop_out_num_inc[1] -= 20
                hlop_out_num_inc[2] -= 20
            model.add_hlop_subspace(hlop_out_num_inc)

        params = []
        for name, param in model.named_parameters():
            if 'hlop' not in name:
                if task_id != 0:
                    if len(param.size()) != 1:
                        params.append(param)
                else:
                    params.append(param)

        # 设置优化器 --------------------------------------------------------------------------------------------------- #
        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(params, lr=args.lr)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(params, lr=args.lr)
        else:
            raise NotImplementedError(args.opt)

        # 设置学习率 --------------------------------------------------------------------------------------------------- #
        lr_scheduler = None
        if args.lr_scheduler == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'CosALR':
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
            lr_lambda = lambda cur_epoch: (cur_epoch + 1) / args.warmup if cur_epoch < args.warmup else 0.5 * (
                    1 + math.cos((cur_epoch - args.warmup) / (args.T_max - args.warmup) * math.pi))
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise NotImplementedError(args.lr_scheduler)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            if task_id != 0:
                model.fix_bn()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()

            bar = Bar('Processing', max=((xtrain.size(0) - 1) // args.b + 1))

            train_loss = 0
            train_acc = 0
            train_samples = 0
            batch_idx = 0

            r = np.arange(xtrain.size(0))
            np.random.shuffle(r)
            for i in range(0, len(r), args.b):
                if i + args.b <= len(r):
                    index = r[i: i + args.b]
                else:
                    index = r[i:]
                batch_idx += 1
                x = xtrain[index].float().cuda()

                # repeat for time steps
                x = x.unsqueeze(1)
                x = x.repeat(1, args.timesteps, 1, 1, 1)

                label = ytrain[index].cuda()

                optimizer.zero_grad()
                if task_id == 0:
                    if args.baseline:
                        out = model(x, task_id, projection=False, update_hlop=False)
                    else:
                        if epoch <= args.hlop_start_epochs:
                            out = model(x, task_id, projection=False, update_hlop=False)
                        else:
                            out = model(x, task_id, projection=False, update_hlop=True)
                else:
                    if args.baseline:
                        out = model(x, task_id, projection=False, proj_id_list=[0], update_hlop=False,
                                    fix_subspace_id_list=[0])
                    else:
                        if epoch <= args.hlop_start_epochs:
                            out = model(x, task_id, projection=True, proj_id_list=[0], update_hlop=False,
                                        fix_subspace_id_list=[0])
                        else:
                            out = model(x, task_id, projection=True, proj_id_list=[0], update_hlop=True,
                                        fix_subspace_id_list=[0])

                loss = F.cross_entropy(out, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * label.numel()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, x.size(0))
                top1.update(prec1.item(), x.size(0))
                top5.update(prec5.item(), x.size(0))

                train_samples += label.numel()
                train_acc += (out.argmax(1) == label).float().sum().item()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=((xtrain.size(0) - 1) // args.b + 1),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                )
                bar.next()
            bar.finish()

            train_loss /= train_samples
            train_acc /= train_samples

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            test_loss, test_acc = test(args, model, xtest, ytest, task_id)

            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

            total_time = time.time() - start_time
            print(
                f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now() + datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

        # save accuracy 
        jj = 0
        for ii in np.array(task_list)[0:task_id + 1]:
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id, jj] = test(args, model, xtest, ytest, ii)
            jj += 1
        print('Accuracies =')
        for i_a in range(task_id + 1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
            print()

        model.merge_hlop_subspace()

        # 如果重放并且起码参与了一个任务
        if args.replay and task_id >= 1:
            print('memory replay\n')
            params = []
            for name, param in model.named_parameters():
                if 'hlop' not in name:
                    if len(param.size()) != 1:
                        params.append(param)

            if args.opt == 'SGD':
                optimizer = torch.optim.SGD(params, lr=args.replay_lr)
            elif args.opt == 'Adam':
                optimizer = torch.optim.Adam(params, lr=args.replay_lr)
            else:
                raise NotImplementedError(args.opt)

            lr_scheduler = None
            if args.lr_scheduler == 'StepLR':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            elif args.lr_scheduler == 'CosALR':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.replay_T_max)
            else:
                raise NotImplementedError(args.lr_scheduler)

            for epoch in range(1, args.replay_epochs + 1):
                start_time = time.time()
                model.train()
                model.fix_bn()

                batch_per_task = args.replay_b
                task_data_num = replay_data[0]['x'].size(0)
                r = np.arange(task_data_num)
                np.random.shuffle(r)
                for i in range(0, task_data_num, batch_per_task):
                    optimizer.zero_grad()
                    for replay_task_id in range(task_id + 1):
                        xtrain = replay_data[replay_task_id]['x']
                        ytrain = replay_data[replay_task_id]['y']

                        if i + batch_per_task <= task_data_num:
                            index = r[i: i + batch_per_task]
                        else:
                            index = r[i:]

                        x = xtrain[index].float().cuda()

                        # repeat for time steps
                        x = x.unsqueeze(1)
                        x = x.repeat(1, args.timesteps, 1, 1, 1)

                        label = ytrain[index].cuda()

                        # out = model(x, replay_taskid, projection=False, update_hlop=True)
                        out = model(x, replay_task_id, projection=False, update_hlop=False)
                        loss = F.cross_entropy(out, label)
                        loss.backward()
                    optimizer.step()

                lr_scheduler.step()

            # 保存准确率
            jj = 0
            for ii in np.array(task_list)[0:task_id + 1]:
                xtest = data[ii]['test']['x']
                ytest = data[ii]['test']['y']
                _, acc_matrix[task_id, jj] = test(args, model, xtest, ytest, ii)
                jj += 1
            print('Accuracies =')
            for i_a in range(task_id + 1):
                print('\t', end='')
                for j_a in range(acc_matrix.shape[1]):
                    print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                print()

        # 保存模型
        torch.save(model.state_dict(), os.path.join(pt_dir, 'model_task{task_id}.pth'.format(task_id=task_id)))
        # 更新task的id
        task_id += 1

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
    parser.add_argument('-epochs', default=1, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, default='./data')
    parser.add_argument('-out_dir', type=str, help='root dir for saving logs and checkpoint', default='./logs')

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
