#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 联邦学习客户端的基础类
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from progress.bar import Bar

from ..meter.AverageMeter import AverageMeter
from ..utils.eval import accuracy

__all__ = ['Client']

from ..utils.model_utils import reset_net


class Client(object):
    def __init__(self, args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs):
        """

        @param args:
        @param id:
        @param xtrain:
        @param test_samples:
        @param kwargs:
        """
        # 设置CPU生成随机数的种子 ，方便下次复现实验结果。
        torch.manual_seed(0)
        self.args = args
        self.id = id  # id标识
        self.fed_algorithm = args.fed_algorithm  # 联邦算法

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 训练设备、数据集 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 设备
        # 训练集
        # 测试集
        # 重放数据集
        # 本地可执行任务
        self.device = args.device
        self.xtrain, self.ytrain = xtrain, ytrain
        self.xtest, self.ytest = xtest, ytest
        self.train_samples = xtrain[0].size(0)
        self.replay_xtrain, self.replay_ytrain = {}, {}
        self.local_tasks = list(self.xtrain.keys())

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 模型训练相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 本地模型
        # 损失
        # 优化器
        # 学习率
        # 学习率调节器
        self.local_model = copy.deepcopy(local_model)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = None
        self.learning_rate = args.client_learning_rate
        self.learning_rate_decay = args.learning_rate_decay
        self.learning_rate_scheduler = None

        # 本地轮次
        # 重放轮次
        # 批处理大小
        # 重放批处理大小
        self.local_epochs = args.local_epochs
        self.replay_epochs = args.replay_epochs
        self.batch_size = args.batch_size
        self.replay_batch_size = args.replay_batch_size
        # 记忆的大小
        # SNN的时间步
        self.memory_size = args.memory_size
        self.timesteps = args.timesteps

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.local_model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 保存文件相关路径 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 客户端i保存的根目录路径
        # 日志文件夹
        # 模型pt文件夹
        self.root_path = os.path.join(args.root_path, 'Client', f'id_{self.id}')
        self.logs_path = os.path.join(self.root_path, 'logs')
        self.models_path = os.path.join(self.root_path, 'models')

    def set_parameters(self, model):
        """
        根据接收到的模型参数设置本地模型参数
        :param model: 接收到的模型
        :return:
        """
        for new_param, old_param in zip(model.parameters(), self.local_model.parameters()):
            old_param.data = new_param.data.clone()

    def set_optimizer(self, task_id: int, experiment_name: str, replay: bool):
        """
        根据任务的id、实验的名称和是否重播来设置优化器
        @param task_id: 任务的id
        @param experiment_name: 实验的名称
        @param replay: 是否重播
        @return:
        """
        # 获取本地模型参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        params = []
        for name, p in self.local_model.named_parameters():
            if 'hlop' not in name:
                if task_id != 0:
                    if len(p.size()) != 1:
                        params.append(p)
                else:
                    params.append(p)
        # 获取本地模型参数 <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            # 如果实验的名称是pmnist，设置replay=True才能真正重放
            if replay and experiment_name == 'pmnist':
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.args.replay_lr)
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.args.replay_lr)
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.args.lr)
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
                else:
                    raise NotImplementedError(self.args.opt)
        elif experiment_name == 'cifar':  # cifar 实验
            if replay:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.args.replay_lr, momentum=self.args.momentum)
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.args.replay_lr)
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum)
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
                else:
                    raise NotImplementedError(self.args.opt)
        elif experiment_name == 'miniimagenet':  # miniimagenet 实验
            if replay:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.args.replay_lr, momentum=self.args.momentum)
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.args.replay_lr)
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum)
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.args.lr_continual, momentum=self.args.momentum)
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.args.lr_continual)
                else:
                    raise NotImplementedError(self.args.opt)
        elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if replay:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.args.replay_lr, momentum=self.args.momentum)
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.args.replay_lr)
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum)
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.args.lr_continual, momentum=self.args.momentum)
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.args.lr)
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.args.lr_continual)
                else:
                    raise NotImplementedError(self.args.opt)

    def set_learning_rate_scheduler(self, experiment_name: str, replay: bool):
        """
        根据实验的名称和是否重播来设置优化器
        @param experiment_name: 实验的名称
        @param replay: 是否重播
        @return:
        """
        if experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            # 如果实验的名称是pmnist，设置replay=True才能真正重放
            if replay and experiment_name == 'pmnist':
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.args.replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
                    lr_lambda = lambda cur_epoch: (
                                                          cur_epoch + 1) / self.args.warmup if cur_epoch < self.args.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.args.warmup) / (self.args.T_max - self.args.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif experiment_name == 'cifar':  # cifar 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.args.replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.learning_rate_scheduler == 'CosALR':
                    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
                    lr_lambda = lambda cur_epoch: (
                                                          cur_epoch + 1) / self.args.warmup if cur_epoch < self.args.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.args.warmup) / (self.args.T_max - self.args.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif experiment_name == 'miniimagenet':  # miniimagenet 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.args.replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
                    lr_lambda = lambda cur_epoch: (
                                                          cur_epoch + 1) / self.args.warmup if cur_epoch < self.args.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.args.warmup) / (self.args.T_max - self.args.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.args.replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.args.step_size,
                                                                                   gamma=self.args.gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.args.warmup if cur_epoch < self.args.warmup else 0.5 * (
                            1 + math.cos(
                        (cur_epoch - self.args.warmup) / (self.args.T_max - self.args.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)

    def set_replay_data(self, task_id, ncla):
        self.replay_xtrain[task_id], self.replay_ytrain[task_id] = [], []
        for class_name in range(ncla):
            num = self.memory_size
            index = 0
            while num > 0:
                if self.ytrain[task_id][index] == class_name:
                    self.replay_xtrain[task_id].append(self.xtrain[task_id][index])
                    self.replay_ytrain[task_id].append(self.ytrain[task_id][index])
                    num -= 1
                index += 1
        self.replay_xtrain[task_id] = torch.stack(self.replay_xtrain[task_id], dim=0)
        self.replay_ytrain[task_id] = torch.stack(self.replay_ytrain[task_id], dim=0)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 模型训练、重放、测试操作 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def train_metrics(self, task_id, bptt, ottt, **kwargs):
        """
        训练
        @param task_id: 任务的id
        @param bptt: 是否是bptt实验
        @param ottt: 是否是ottt实验
        @return: 
        """
        # 开启模型训练模式
        self.local_model.train()
        # 获取对应任务的训练集和测试集
        xtrain, ytrain = self.xtrain[task_id], self.ytrain[task_id]
        if task_id != 0:
            self.local_model.fix_bn()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Client {:3d} Training'.format(self.id), max=((xtrain.size(0) - 1) // self.batch_size + 1))

        train_num = 0
        train_acc = 0
        train_loss = 0
        batch_idx = 0

        samples_index = np.arange(xtrain.size(0))
        np.random.shuffle(samples_index)

        # 本地轮次的操作
        for local_epoch in range(1, self.local_epochs + 1):
            # 一个轮次中的批处理操作
            for i in range(0, len(samples_index), self.batch_size):
                # 如果可以获取完整的批次，那么就获取完整批次
                if i + self.batch_size <= len(samples_index):
                    index = samples_index[i: i + self.batch_size]
                # 如果没有完整的批次可供获取，那么获取所有剩下的
                else:
                    index = samples_index[i:]
                batch_idx += 1

                # 获取一个批次的数据和标签
                x, y = xtrain[index].float().to(self.device), ytrain[index].to(self.device)

                if ottt:
                    total_loss = 0.
                    if not self.args.online_update:
                        self.optimizer.zero_grad()
                    for t in range(self.timesteps):
                        if self.args.online_update:
                            self.optimizer.zero_grad()
                        init = (t == 0)
                        if task_id == 0:
                            flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                            out_fr = self.local_model(x, task_id, projection=False, update_hlop=flag, init=init)
                        else:
                            flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                            out_fr = self.local_model(x, task_id, projection=not self.args.baseline, proj_id_list=[0],
                                                      update_hlop=flag, fix_subspace_id_list=[0], init=init)
                        if t == 0:
                            total_fr = out_fr.clone().detach()
                        else:
                            total_fr += out_fr.clone().detach()
                        loss = self.loss(out_fr, y) / self.timesteps
                        loss.backward()
                        total_loss += loss.detach()
                        if self.args.online_update:
                            if self.fed_algorithm == 'SCAFFOLD':
                                # self.optimizer.step(self.global_controls, self.local_controls)
                                self.optimizer.step(kwargs['global_controls'], kwargs['local_controls'])

                            else:
                                self.optimizer.step()
                    if not self.args.online_update:
                        if self.fed_algorithm == 'SCAFFOLD':
                            self.optimizer.step(kwargs['global_controls'], kwargs['local_controls'])
                        else:
                            self.optimizer.step()
                    train_loss += total_loss.item() * y.numel()
                    out = total_fr
                elif bptt:
                    self.optimizer.zero_grad()
                    if task_id == 0:
                        flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                        out = self.local_model(x, task_id, projection=False, update_hlop=flag)
                    else:
                        flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                        out = self.local_model(x, task_id, projection=not self.args.baseline, proj_id_list=[0],
                                               update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, y)
                    loss.backward()
                    if self.fed_algorithm == 'SCAFFOLD':
                        self.optimizer.step(kwargs['global_controls'], kwargs['local_controls'])
                    else:
                        self.optimizer.step()
                    reset_net(self.local_model)
                    train_loss += loss.item() * y.numel()
                else:
                    x = x.unsqueeze(1)
                    x = x.repeat(1, self.timesteps, 1, 1, 1)
                    self.optimizer.zero_grad()

                    if task_id == 0:
                        flag = not (self.args.baseline and (local_epoch <= self.args.hlop_start_epochs))
                        out = self.local_model(x, task_id, projection=False, update_hlop=flag)

                    else:
                        flag = not (self.args.baseline or (local_epoch <= self.args.hlop_start_epochs))
                        out = self.local_model(x, task_id, projection=not self.args.baseline, proj_id_list=[0],
                                               update_hlop=flag, fix_subspace_id_list=[0])
                    loss = self.loss(out, y)
                    loss.backward()
                    if self.fed_algorithm == 'SCAFFOLD':
                        self.optimizer.step(kwargs['global_controls'], kwargs['local_controls'])
                    else:
                        self.optimizer.step()
                    train_loss += loss.item() * y.numel()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, y.data, topk=(1, 5))
                losses.update(loss, x.size(0))
                top1.update(prec1.item(), x.size(0))
                top5.update(prec5.item(), x.size(0))

                train_num += y.numel()
                train_acc += (out.argmax(1) == y).float().sum().item()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=((xtrain.size(0) - 1) // self.batch_size + 1),
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

        train_loss /= train_num
        train_acc /= train_num
        self.learning_rate_scheduler.step()

        return train_loss, train_acc, train_num

    def replay_metrics(self, tasks_learned, **kwargs):
        """
        重放
        @param tasks_learned: 学习完的任务
        @return:
        """

        self.local_model.train()
        self.local_model.fix_bn()

        for replay_task in tasks_learned:
            xtrain, ytrain = self.replay_xtrain[replay_task], self.replay_ytrain[replay_task]
            task_data_num = self.replay_xtrain[replay_task].size(0)
            r = np.arange(task_data_num)
            np.random.shuffle(r)
            for epoch in range(1, self.replay_epochs + 1):
                for i in range(0, task_data_num, self.replay_batch_size):
                    if i + self.replay_batch_size <= task_data_num:
                        index = r[i: i + self.replay_batch_size]
                    else:
                        index = r[i:]
                    x = xtrain[index].float().to(self.device)
                    x = x.unsqueeze(1)
                    x = x.repeat(1, self.timesteps, 1, 1, 1)
                    y = ytrain[index].to(self.device)

                    self.optimizer.zero_grad()
                    out = self.local_model(x, replay_task, projection=False, update_hlop=False)
                    loss = self.loss(out, y)
                    loss.backward()
                if self.fed_algorithm == 'SCAFFOLD':
                    self.optimizer.step(kwargs['global_controls'], kwargs['local_controls'])
                else:
                    self.optimizer.step()
            self.learning_rate_scheduler.step()

    def test_metrics(self, task_id, bptt, ottt):
        """
        测试模型
        :return: 模型测试准确率、测试数据的数量和模型的AUC（Area Under the Curve）指标
        """
        self.local_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Client {:3d} Testing'.format(self.id),
                  max=((self.xtest[task_id].size(0) - 1) // self.batch_size + 1))

        test_acc = 0
        test_loss = 0
        test_num = 0
        batch_idx = 0

        r = np.arange(self.xtest[task_id].size(0))
        with torch.no_grad():
            for i in range(0, len(r), self.batch_size):
                if i + self.batch_size <= len(r):
                    index = r[i: i + self.batch_size]
                else:
                    index = r[i:]
                batch_idx += 1
                input = self.xtest[task_id][index].float().to(self.device)
                label = self.ytest[task_id][index].to(self.device)

                if bptt:
                    out = self.local_model(input, task_id, projection=False, update_hlop=False)
                    loss = self.loss(out, label)
                    reset_net(self.local_model)
                elif ottt:
                    loss = 0.
                    for t in range(self.timesteps):
                        if t == 0:
                            out_fr = self.local_model(input, task_id, projection=False, update_hlop=False, init=True)
                            total_fr = out_fr.clone().detach()
                        else:
                            out_fr = self.local_model(input, task_id, projection=False, update_hlop=False)
                            total_fr += out_fr.clone().detach()
                        loss += self.loss(out_fr, label).detach() / self.timesteps
                    out = total_fr
                else:
                    # repeat for time steps
                    input = input.unsqueeze(1)
                    input = input.repeat(1, self.timesteps, 1, 1, 1)
                    out = self.local_model(input, task_id, projection=False, update_hlop=False)
                    loss = self.loss(out, label)

                test_num += label.numel()
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
                    size=((self.xtest[task_id].size(0) - 1) // self.batch_size + 1),
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

        test_acc /= test_num
        test_loss /= test_num

        return test_loss, test_acc, test_num

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 数据保存、加载操作 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    def save_local_model(self, model_name):
        """
        保存本地模型
        :param model_name: 模型名称（不需要绝对/相对路径）
        :return:
        """
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        torch.save(self.local_model, os.path.join(self.models_path, f'{model_name}.pt'))

    def load_local_model(self, model_name):
        """
        加载本地模型
        :param model_name: 模型名称（不需要绝对/相对路径）
        :return:
        """
        model_abs_path = os.path.join(self.models_path, model_name)
        assert os.path.exists(model_abs_path)
        self.local_model = torch.load(model_abs_path)
