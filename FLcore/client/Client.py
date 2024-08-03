import copy
import datetime
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from progress.bar import Bar
from tensorboardX import SummaryWriter
from torch import optim, nn

from FLcore import models
from FLcore.meter import AverageMeter
from FLcore.utils import accuracy


class Client:
    def __init__(self, id):
        self.id = id

        self.xtrain = None  # 客户端数据集
        self.ytrain = None
        self.replay_xtrain = {}  # 客户端重放数据集
        self.replay_ytrain = {}
        self.tasks = []  # 客户端数据集允许进行的任务
        self.xtrain_epoch = None
        self.ytrain_epoch = None

        self.save_dir = None
        self.logs_dir = None  # 日志文件夹
        self.pts_dir = None  # 模型pt文件夹
        self.args_file = None  # 实验参数文件

        # 客户端网络模型
        self.local_model = None
        self.learning_rate = None
        self.optimizer = None
        self.local_epochs = None
        self.lr_scheduler = None

    def configure_trainset(self, xtrain, ytrain):
        """
        配置客户端的私有数据相关信息
        @param data:
        @return:
        """
        self.xtrain = xtrain  # 配置客户端的私有数据
        self.ytrain = ytrain
        self.tasks = list(self.xtrain.keys())  # 配置客户端的任务（私有数据允许进行的相关任务）

    def configure_data_save_path(self, root_dir, args):
        """
        配置客户端的数据存放路径
        @param root_dir: 存放一次task的根文件夹
        @return:
        """
        # 根文件夹
        self.save_dir = root_dir
        # 客户端自己的文件夹
        client_dir = os.path.join(root_dir, "client{:3d}".format(self.id))
        # 日志文件夹
        self.logs_dir = os.path.join(client_dir, "logs")
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
            print(f'Mkdir {self.logs_dir}.')
        # 模型pt文件夹
        self.pts_dir = os.path.join(client_dir, 'models')
        if not os.path.exists(self.pts_dir):
            os.makedirs(self.pts_dir)
            print(f'Mkdir {self.pts_dir}.')
        # 实验参数文件
        self.args_file = os.path.join(client_dir, 'args.txt')
        # 写入实验参数
        with open(self.args_file, 'w', encoding='utf-8') as file:
            file.write(str(args))

    def configure_local_model(self, model):
        """
        配置网络模型
        @param model:
        @return:
        """
        self.local_model = copy.deepcopy(model)

    def train(self, task_id, ncla, args, is_bptt, is_ottt):
        """
        获取task的id后进行训练
        """
        # 获取训练集
        xtrain = self.xtrain[task_id]
        ytrain = self.ytrain[task_id]

        writer = SummaryWriter(os.path.join(self.logs_dir, "logs_task{task_id}".format(task_id=task_id)))
        self.configure_opt(args.opt, args.lr)
        self.configure_lr_scheduler(args.lr_scheduler, args.step_size, args.gamma, args.warmup, args.T_max, False)

        # 如果需要重放
        if args.replay:
            # save samples for memory replay
            self.replay_xtrain[task_id], self.replay_ytrain[task_id] = [], []
            for c in range(ncla):
                num = args.memory_size
                index = 0
                while num > 0:
                    if ytrain[index] == c:
                        self.replay_xtrain[task_id].append(xtrain[index])
                        self.replay_ytrain[task_id].append(ytrain[index])
                        num -= 1
                    index += 1
            self.replay_xtrain[task_id] = torch.stack(self.replay_xtrain[task_id], dim=0)
            self.replay_ytrain[task_id] = torch.stack(self.replay_ytrain[task_id], dim=0)

    def train_epoch(self, task_id, local_epoch, args, is_bptt, is_ottt):
        self.xtrain_epoch = self.xtrain[task_id]
        self.ytrain_epoch = self.ytrain[task_id]

        self.local_model.train()  # 开启模型训练模式

        if task_id != 0:
            self.local_model.fix_bn()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Client {:3d}'.format(self.id), max=((self.xtrain_epoch.size(0) - 1) // args.b + 1))

        train_loss = 0
        train_acc = 0
        train_samples = 0
        batch_idx = 0

        r = np.arange(self.xtrain_epoch.size(0))
        np.random.shuffle(r)
        for i in range(0, len(r), args.b):
            if i + args.b <= len(r):
                index = r[i: i + args.b]
            else:
                index = r[i:]
            batch_idx += 1
            x = self.xtrain_epoch[index].float().cuda()
            label = self.ytrain_epoch[index].cuda()

            if is_ottt:
                total_loss = 0.
                if not args.online_update:
                    self.optimizer.zero_grad()
                for t in range(args.timesteps):
                    if args.online_update:
                        self.optimizer.zero_grad()
                    init = (t == 0)
                    if task_id == 0:
                        """原始代码
                        if args.baseline:
                            out_fr = self.local_model(x, task_id, projection=False, update_hlop=False, init=init)
                        else:
                            if local_epoch <= args.hlop_start_epochs:
                                out_fr = self.local_model(x, task_id, projection=False, update_hlop=False,
                                                          init=init)
                            else:
                                out_fr = self.local_model(x, task_id, projection=False, update_hlop=True, init=init)
                        """
                        flag = not (args.baseline and (local_epoch <= args.hlop_start_epochs))
                        out_fr = self.local_model(x, task_id, projection=False, update_hlop=flag, init=init)
                    else:
                        """原始代码
                        if args.baseline:
                            out_fr = self.local_model(x, task_id, projection=False, proj_id_list=[0],
                                                      update_hlop=False,
                                                      fix_subspace_id_list=[0], init=init)
                        else:
                            if local_epoch <= args.hlop_start_epochs:
                                out_fr = self.local_model(x, task_id, projection=True, proj_id_list=[0],
                                                          update_hlop=False,
                                                          fix_subspace_id_list=[0], init=init)
                            else:
                                out_fr = self.local_model(x, task_id, projection=True, proj_id_list=[0],
                                                          update_hlop=True,
                                                          fix_subspace_id_list=[0], init=init)
                        """
                        flag = not (args.baseline or (local_epoch <= args.hlop_start_epochs))
                        out_fr = self.local_model(x, task_id, projection=not args.baseline, proj_id_list=[0],
                                                  update_hlop=flag,
                                                  fix_subspace_id_list=[0], init=init)
                    if t == 0:
                        total_fr = out_fr.clone().detach()
                    else:
                        total_fr += out_fr.clone().detach()
                    loss = F.cross_entropy(out_fr, label) / args.timesteps
                    loss.backward()
                    total_loss += loss.detach()
                    if args.online_update:
                        self.optimizer.step()
                if not args.online_update:
                    self.optimizer.step()

                train_loss += total_loss.item() * label.numel()
                out = total_fr
            elif is_bptt:
                self.optimizer.zero_grad()
                if task_id == 0:
                    """原始代码
                    if args.baseline:
                        out = self.local_model(x, task_id, projection=False, update_hlop=False)
                    else:
                        if local_epoch <= args.hlop_start_epochs:
                            out = self.local_model(x, task_id, projection=False, update_hlop=False)
                        else:
                            out = self.local_model(x, task_id, projection=False, update_hlop=True)
                    """
                    flag = not (args.baseline and (local_epoch <= args.hlop_start_epochs))
                    out = self.local_model(x, task_id, projection=False, update_hlop=flag)
                else:
                    """原始代码
                    if args.baseline:
                        out = self.local_model(x, task_id, projection=False, proj_id_list=[0], update_hlop=False,
                                    fix_subspace_id_list=[0])
                    else:
                        if local_epoch <= args.hlop_start_epochs:
                            out = self.local_model(x, task_id, projection=True, proj_id_list=[0], update_hlop=False,
                                        fix_subspace_id_list=[0])
                        else:
                            out = self.local_model(x, task_id, projection=True, proj_id_list=[0], update_hlop=True,
                                        fix_subspace_id_list=[0])
                    """
                    flag = not (args.baseline or (local_epoch <= args.hlop_start_epochs))
                    out = self.local_model(x, task_id, projection=not args.baseline, proj_id_list=[0],
                                           update_hlop=flag, fix_subspace_id_list=[0])
                loss = F.cross_entropy(out, label)
                loss.backward()
                self.optimizer.step()
                self.reset_net(self.local_model)
                train_loss += loss.item() * label.numel()
            else:
                x = x.unsqueeze(1)
                x = x.repeat(1, args.timesteps, 1, 1, 1)
                self.optimizer.zero_grad()

                if task_id == 0:
                    """原始代码
                    if args.baseline:
                        out = model(x, task_id, projection=False, update_hlop=False)
                    else:
                        if epoch <= args.hlop_start_epochs:
                            out = model(x, task_id, projection=False, update_hlop=False)
                        else:
                            out = model(x, task_id, projection=False, update_hlop=True)
                    """
                    flag = not (args.baseline and (local_epoch <= args.hlop_start_epochs))
                    out = self.local_model(x, task_id, projection=False, update_hlop=flag)

                else:
                    """原始代码
                    if args.baseline:
                        out = self.local_model(x, task_id, projection=False, proj_id_list=[0], update_hlop=False,
                                               fix_subspace_id_list=[0])
                    else:
                        if local_epoch <= args.hlop_start_epochs:
                            out = self.local_model(x, task_id, projection=True, proj_id_list=[0], update_hlop=False,
                                                   fix_subspace_id_list=[0])
                        else:
                            out = self.local_model(x, task_id, projection=True, proj_id_list=[0], update_hlop=True,
                                                   fix_subspace_id_list=[0])
                    """
                    flag = not (args.baseline or (local_epoch <= args.hlop_start_epochs))
                    out = self.local_model(x, task_id, projection=not args.baseline, proj_id_list=[0], update_hlop=flag,
                                           fix_subspace_id_list=[0])
                loss = F.cross_entropy(out, label)
                loss.backward()
                self.optimizer.step()
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
                size=((self.xtrain_epoch.size(0) - 1) // args.b + 1),
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
        self.lr_scheduler.step()

        return train_loss, train_acc

    def replay_train_epoch(self, task_id, args):
        self.local_model.train()
        self.local_model.fix_bn()

        batch_per_task = args.replay_b
        task_data_num = self.replay_xtrain[0]['x'].size(0)
        r = np.arange(task_data_num)
        np.random.shuffle(r)
        for i in range(0, task_data_num, batch_per_task):
            self.optimizer.zero_grad()
            for replay_task_id in range(task_id + 1):
                xtrain = self.replay_xtrain[replay_task_id]['x']
                ytrain = self.replay_xtrain[replay_task_id]['y']

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
                out = self.local_model(x, replay_task_id, projection=False, update_hlop=False)
                loss = F.cross_entropy(out, label)
                loss.backward()
            self.optimizer.step()
        self.lr_scheduler.step()

    def commit_model_weight(self):
        return self.local_model.state_dict()

    def update_model_weight(self, weight):
        self.local_model.load_state_dict(weight)

    def save_pt(self, task_id):
        # 保存模型
        torch.save(self.local_model.state_dict(),
                   os.path.join(self.pts_dir, 'model_task{task_id}.pth'.format(task_id=task_id)))

    def reset_net(self, net: nn.Module):
        """
        将网络的状态重置。做法是遍历网络中的所有 ``Module``，若含有 ``reset()`` 函数，则调用。
        @param net: 任何属于 ``nn.Module`` 子类的网络
        @return:
        """
        for m in net.modules():
            if hasattr(m, 'reset'):
                m.reset()
