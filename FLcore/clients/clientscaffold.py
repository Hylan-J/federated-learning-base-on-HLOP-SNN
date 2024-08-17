#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 联邦学习联邦（clientscaffold）客户端类
import copy
import time

import numpy as np
import torch

from ..clients.clientbase import Client
from ..optimizers.SCAFFOLDOptimizer import SCAFFOLDOptimizer

__all__ = ['clientSCAFFOLD']


class clientSCAFFOLD(Client):
    def __init__(self, args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs):
        super().__init__(args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs)
        # 优化器
        self.optimizer = SCAFFOLDOptimizer(self.local_model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                   gamma=self.args.learning_rate_decay_gamma)
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< SCAFFOLD 算法相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 本地控制参数
        self.local_controls = []

        # 全局控制参数
        self.global_controls = None
        # 全局模型
        self.global_model = None

        self.num_batches = None
        self.max_local_epochs = None

    def set_optimizer(self):
        self.optimizer = SCAFFOLDOptimizer(self.local_model.parameters(), lr=self.learning_rate)

    def set_learning_rate_scheduler(self):
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer,
                                                                   gamma=self.args.learning_rate_decay_gamma)

    def set_parameters(self, global_model, global_controls):
        """
        从服务器接收到全局的模型和控制参数更新本地的模型和控制参数
        @param global_model: 全局的模型
        @param global_controls: 全局的控制参数
        @return:
        """
        # 利用服务器接收到的全局模型更新本地模型
        """for global_param, local_param in zip(global_model.parameters(), self.local_model.parameters()):
            local_param.data = global_param.data.clone()"""
        self.local_model = copy.deepcopy(global_model)
        # 获取全局的控制参数
        self.global_controls = copy.deepcopy(global_controls)
        # 获取全局的模型
        self.global_model = copy.deepcopy(global_model)

        self.local_controls = [torch.zeros_like(param) for param in self.local_model.parameters()]

        """for param, local_control in zip(self.local_model.parameters(), self.local_controls):
            print(param.shape)
            print(local_control.shape)"""


    def train(self, task_id, bptt, ottt):
        # 本地模型开启训练模式
        self.local_model.train()

        self.max_local_epochs = self.local_epochs
        if self.train_slow:
            self.max_local_epochs = np.random.randint(1, self.max_local_epochs // 2)

        start_time = time.time()

        super().train_metrics(task_id, bptt, ottt, global_controls=self.global_controls,
                              local_controls=self.local_controls)

        self.update_model_and_control_param(task_id)
        # self.delta_c, self.delta_y = self.delta_yc(max_local_epochs)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def update_model_and_control_param(self, task_id):
        num_batches = np.ceil(len(self.xtrain[task_id]) // self.batch_size)
        for local_param, local_control, global_param, global_control in zip(self.local_model.parameters(),
                                                                            self.local_controls,
                                                                            self.global_model.parameters(),
                                                                            self.global_controls):
            print('local_model形状:', local_param.shape)
            print('local_control形状:', local_control.shape)
            print('global_model形状:', global_param.shape)
            print('global_control形状:', global_control.shape)
            local_control.data = local_control - global_control + (global_param - local_param) / num_batches / self.max_local_epochs / self.learning_rate

    def calculate_delta_model_and_control_param(self, task_id):
        """
        计算
        @param task_id:
        @return:
        """
        delta_model, delta_control = [], []

        num_batches = np.ceil(self.xtrain[task_id].shape[-1] // self.batch_size)
        print(self.xtrain[task_id].shape)
        print(self.xtrain[task_id].shape[-1])
        print(self.xtrain[task_id].shape[-2])
        for global_control, global_model_param, local_model_param in zip(self.global_controls,
                                                                         self.global_model.parameters(),
                                                                         self.local_model.parameters()):
            delta_model.append(local_model_param - global_model_param)
            delta_control.append(- global_control + 1 / num_batches / self.max_local_epochs / self.learning_rate *
                                 (global_model_param - local_model_param))

        return delta_model, delta_control
