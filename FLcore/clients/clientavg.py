#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 联邦学习联邦平均（fedavg）客户端类

import time

import numpy as np

from ..clients.clientbase import Client

__all__ = ['clientAVG']


class clientAVG(Client):
    def __init__(self, args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs):
        super().__init__(args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs)

    def train(self, task_id, bptt, ottt):
        self.local_model.train()
        if self.train_slow:
            self.local_epochs = np.random.randint(1, self.local_epochs // 2)

        start_time = time.time()

        super().train_metrics(task_id, bptt, ottt)

        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def replay(self, tasks_learned):
        super().replay_metrics(tasks_learned)
