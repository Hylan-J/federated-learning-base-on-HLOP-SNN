#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 联邦学习客户端的基础类
import os
import copy
import math

import torch
import torch.nn as nn

__all__ = ['Client']


class Client(object):
    def __init__(self, args, id, xtrain, ytrain, local_model, **kwargs):
        """
        @param args:
        @param id:
        @param xtrain:
        @param test_samples:
        @param kwargs:
        """
        self.args = args
        self.id = id  # id标识
        self.experiment_name = args.experiment_name
        self.fed_algorithm = args.fed_algorithm  # 联邦算法
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 训练设备、数据集 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 设备
        # 训练集
        # 测试集
        # 重放数据集
        # 本地可执行任务
        self.device = args.device
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.train_samples = len(xtrain[0])
        self.replay_xtrain = {}
        self.replay_ytrain = {}
        self.local_tasks = list(self.xtrain.keys())

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 模型训练相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 本地模型
        # 损失
        # 优化器
        # 学习率
        # 学习率调节器
        self.local_model = copy.deepcopy(local_model)

        self.loss = nn.CrossEntropyLoss()
        # 本地轮次
        # 重放轮次
        # 批处理大小
        # 重放批处理大小
        self.local_epochs = args.local_epochs
        self.replay_local_epochs = args.replay_local_epochs
        self.batch_size = args.batch_size
        self.replay_batch_size = args.replay_batch_size

        self.optimizer = None
        self.momentum = args.momentum
        self.learning_rate = args.learning_rate
        self.continual_learning_rate = args.continual_learning_rate
        self.replay_learning_rate = args.replay_learning_rate
        self.current_learning_rate = self.learning_rate

        self.learning_rate_scheduler = None
        self.warmup = args.warmup
        self.StepLR_step_size = args.step_size
        self.StepLR_gamma = args.gamma
        self.CosineAnnealingLR_T_max = args.T_max
        self.CosineAnnealingLR_replay_T_max = args.replay_T_max

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
        pass

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
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
                    self.current_learning_rate = self.learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                    self.current_learning_rate = self.learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
        elif experiment_name == 'cifar':  # cifar 实验
            if replay:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                    self.current_learning_rate = self.learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
        elif experiment_name == 'miniimagenet':  # miniimagenet 实验
            if replay:
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_learning_rate,
                                                         momentum=self.momentum)
                        self.current_learning_rate = self.continual_learning_rate
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_learning_rate)
                        self.current_learning_rate = self.continual_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
        elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if replay:  # 如果重放
                if self.args.opt == 'SGD':
                    self.optimizer = torch.optim.SGD(params, lr=self.replay_learning_rate, momentum=self.momentum)
                    self.current_learning_rate = self.replay_learning_rate
                elif self.args.opt == 'Adam':
                    self.optimizer = torch.optim.Adam(params, lr=self.replay_learning_rate)
                    self.current_learning_rate = self.replay_learning_rate
                else:
                    raise NotImplementedError(self.args.opt)
            else:  # 如果不重放
                if self.args.opt == 'SGD':
                    if task_id == 0:
                        self.optimizer = torch.optim.SGD(params, lr=self.learning_rate, momentum=self.momentum)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.SGD(params, lr=self.continual_learning_rate,
                                                         momentum=self.momentum)
                        self.current_learning_rate = self.continual_learning_rate
                elif self.args.opt == 'Adam':
                    if task_id == 0:
                        self.optimizer = torch.optim.Adam(params, lr=self.learning_rate)
                        self.current_learning_rate = self.learning_rate
                    else:
                        self.optimizer = torch.optim.Adam(params, lr=self.continual_learning_rate)
                        self.current_learning_rate = self.continual_learning_rate
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
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif experiment_name == 'cifar':  # cifar 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.learning_rate_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif experiment_name == 'miniimagenet':  # miniimagenet 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
        elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if replay:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                                              T_max=self.CosineAnnealingLR_replay_T_max)
                else:
                    raise NotImplementedError(self.args.lr_scheduler)
            else:
                if self.args.lr_scheduler == 'StepLR':
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                                                   step_size=self.StepLR_step_size,
                                                                                   gamma=self.StepLR_gamma)
                elif self.args.lr_scheduler == 'CosALR':
                    lr_lambda = lambda cur_epoch: (cur_epoch + 1) / self.warmup if cur_epoch < self.warmup else 0.5 * (
                            1 + math.cos((cur_epoch - self.warmup) / (self.CosineAnnealingLR_T_max - self.warmup) * math.pi))
                    self.learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
                                                                                     lr_lambda=lr_lambda)
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
    def train(self, task_id, HLOP_SNN):
        """
        模型训练的主要部分
        @param task_id: 任务的id
        @param HLOP_SNN: 是否是HLOP_SNN相关实验
        @return: 
        """
        pass

    def replay(self, tasks_learned, HLOP_SNN):
        """
        重放
        @param tasks_learned: 学习完的任务
        @return:
        """
        pass

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
