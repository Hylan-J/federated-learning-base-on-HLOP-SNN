#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : SCAFFOLD算法的服务器类
import copy
import os
import random
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from ..clients.clientscaffold import clientSCAFFOLD
from ..servers.serverbase import Server
from ..utils.prepare_utils import prepare_bptt_ottt, prepare_hlop_out

__all__ = ['SCAFFOLD']


class SCAFFOLD(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model, times):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model, times)
        self.set_slow_clients()
        self.set_clients(clientSCAFFOLD, self.xtrain, self.ytrain, model)
        # 全局控制变量
        self.global_controls = []
        self.learning_rate = args.server_learning_rate
        self.time_cost = []

    def send_models(self):
        """
        向客户端发送全局模型
        @return:
        """
        # 断言服务器的客户端数不为零
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            client.set_parameters(self.global_model, self.global_controls)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def aggregate_parameters(self, task_id):
        """
        SCAFFOLD聚合参数
        @return:
        """
        # 全局模型的深拷贝
        global_model = copy.deepcopy(self.global_model)
        # 全局控制参数的深拷贝
        global_controls = copy.deepcopy(self.global_controls)
        # 计算聚合后的全局模型和控制参数
        for idx in self.received_info['client_ids']:
            delta_model, delta_control = self.clients[idx].delta_yc(task_id)
            for global_model_param, local_model_param in zip(global_model.parameters(), delta_model):
                global_model_param.data += local_model_param.data.clone() / self.num_join_clients * self.learning_rate
            for global_control_param, local_control_param in zip(global_controls, delta_control):
                global_control_param.data += local_control_param.data.clone() / self.num_clients
        self.global_model = global_model
        self.global_controls = global_controls

    def execute(self, experiment_name: str, replay: bool, HLOP_SNN: bool):
        bptt, ottt = prepare_bptt_ottt(experiment_name)
        if bptt or ottt:
            replay = False

        hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1 = prepare_hlop_out(experiment_name)

        task_learned = []
        task_count = 0

        tasks = [task_id for task_id, ncla in self.taskcla]
        total_task_num = len(tasks)

        acc_matrix = np.zeros((total_task_num, total_task_num))

        for task_id, ncla in self.taskcla:
            task_learned.append(task_id)
            writer = SummaryWriter(os.path.join(self.args.root_path, 'task{task_id}'.format(task_id=task_id)))

            # 如果使用HLOP-SNN方法，那么就需要根据相关参数进行调整
            if HLOP_SNN:
                self.adjust_to_HLOP_SNN_before_train_task(experiment_name, ncla, task_count,
                                                          hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1)
                self.global_controls = [torch.zeros_like(param) for param in self.global_model.parameters()]

            for client in self.clients:
                if replay:
                    client.set_replay_data(task_id, ncla)
                client.set_optimizer(task_id, experiment_name, False)
                client.set_learning_rate_scheduler(experiment_name, False)

            # 对于任务task_id，进行联邦训练
            for global_round in range(1, self.global_rounds + 1):
                start_time = time.time()
                # ①挑选合适客户端
                self.select_clients(task_id)
                # ②服务器向选中的客户端发放全局模型
                self.send_models()
                # ③选中的客户端进行训练
                for client in self.selected_clients:
                    client.train(task_id, self.HLOP_SNN)
                # ④服务器接收训练后的客户端模型
                self.receive_models()
                # ⑤服务器聚合全局模型
                self.aggregate_parameters(task_id)

                self.time_cost.append(time.time() - start_time)
                print('-' * 25, 'Task', task_id, 'Time Cost', self.time_cost[-1], '-' * 25)
                # 当前轮次达到评估轮次
                if global_round % self.eval_gap == 0:
                    print(f"\n-------------Round number: {global_round}-------------")
                    print("\nEvaluate global model")
                    test_loss, test_acc = self.evaluate(task_id, self.HLOP_SNN)
                    writer.add_scalar('test_loss', test_loss, global_round)
                    writer.add_scalar('test_acc', test_acc, global_round)
                """if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                        break"""

            jj = 0
            for ii in np.array(task_learned)[0:task_count + 1]:
                _, acc_matrix[task_count, jj] = self.evaluate(ii, self.HLOP_SNN)
                jj += 1
            print('Accuracies =')
            for i_a in range(task_count + 1):
                print('\t', end='')
                for j_a in range(acc_matrix.shape[1]):
                    print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                print()

            if HLOP_SNN:
                self.adjust_to_HLOP_SNN_after_train_task()

            # 如果重放并且起码参与了一个任务
            if replay and task_count >= 1:
                print('memory replay\n')

                for replay_global_round in range(1, self.replay_global_rounds + 1):
                    self.select_clients(task_id)
                    self.send_models()

                    for client in self.clients:
                        client.replay(task_learned, self.HLOP_SNN)
                    self.receive_models()
                    self.aggregate_parameters(task_id)

                # 保存准确率
                jj = 0
                for ii in np.array(task_learned)[0:task_count + 1]:
                    _, acc_matrix[task_count, jj] = self.evaluate(ii, self.HLOP_SNN)
                    jj += 1
                print('Accuracies =')
                for i_a in range(task_count + 1):
                    print('\t', end='')
                    for j_a in range(acc_matrix.shape[1]):
                        print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                    print()

            if self.num_new_clients > 0:
                self.eval_new_clients = True
                self.set_new_clients(clientSCAFFOLD, self.xtrain, self.ytrain, self.xtest, self.ytest)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(task_id, self.HLOP_SNN)

            task_count += 1

