#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description : 服务器的基础类
import os
import copy
import time
import random

import numpy as np
import torch
from progress.bar import Bar
from torch import nn

from ..meter.AverageMeter import AverageMeter
from ..utils.eval import accuracy


__all__ = ['Server']

from ..utils.model_utils import reset_net


class Server(object):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model, times):
        self.learning_rate = args.learning_rate

        self.args = args
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 训练设备、数据集 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 设备
        # 数据集
        # 数据集描述
        # 全局可执行任务
        self.device = args.device
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.taskcla = taskcla
        self.global_tasks = []

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 模型训练相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 全局模型
        # 全局训练轮数
        # 重放轮数
        self.global_model = copy.deepcopy(model)
        self.loss = nn.CrossEntropyLoss()
        self.batch_size = args.batch_size
        self.global_rounds = args.global_rounds
        self.replay_global_rounds = args.replay_global_rounds
        self.timesteps = args.timesteps

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 客户端相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 客户端数量
        # 参与比例
        # 随机参与比例
        # 参与的客户端数量
        # 当前参与的客户端数量
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.top_cnt = 100
        self.auto_break = args.auto_break

        self.received_info = {
            'client_ids': [],
            'client_train_samples_weights': [],
            'client_models': []
        }

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 保存文件相关路径 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # 服务器保存的根目录路径
        # 日志文件夹
        # 模型pt文件夹
        self.root_path = os.path.join(args.root_path, 'Server')
        self.logs_path = os.path.join(self.root_path, 'logs')
        self.models_path = os.path.join(self.root_path, 'models')

        self.experiment_name = args.experiment_name
        self.HLOP_SNN = args.HLOP_SNN
        self.replay = args.replay

    # ------------------------------------------------------------------------------------------------------------------
    # 设置相关客户端操作
    # ------------------------------------------------------------------------------------------------------------------

    # 生成现有的客户端
    def set_clients(self, clientObj, xtrain, ytrain, model):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            client = clientObj(args=self.args, id=i,
                               xtrain=xtrain, ytrain=ytrain,
                               local_model=copy.deepcopy(model),
                               train_slow=train_slow, send_slow=send_slow)
            self.clients.append(client)

    # 生成新增的客户端
    def set_new_clients(self, clientObj, xtrain, ytrain, xtest, ytest):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            client = clientObj(args=self.args, id=i,
                               xtrain=xtrain, ytrain=ytrain,
                               xtest=xtest, ytest=ytest,
                               local_model=self.global_model,
                               train_slow=False, send_slow=False)
            self.new_clients.append(client)

    # 设置train_slow和send_slow的客户端
    def set_slow_clients(self):
        indexes = [i for i in range(self.num_clients)]  # 获取所有客户端的索引

        # 计算train_slow客户端的数量
        train_slow_num = int(self.train_slow_rate * self.num_clients)
        # 从所有客户端的索引中随机挑选满足train_slow客户端的数量的索引
        train_slow_indexes = np.random.choice(indexes, train_slow_num)
        # 将挑出的索引标记为train_slow客户端
        self.train_slow_clients = [False for _ in range(self.num_clients)]
        for i in train_slow_indexes:
            self.train_slow_clients[i] = True

        # 计算send_slow客户端的数量
        send_slow_num = int(self.send_slow_rate * self.num_clients)
        # 从所有客户端的索引中随机挑选满足send_slow客户端的数量的索引
        send_slow_indexes = np.random.choice(indexes, send_slow_num)
        # 将挑出的索引标记为send_slow客户端
        self.send_slow_clients = [False for _ in range(self.num_clients)]
        for i in send_slow_indexes:
            self.send_slow_clients[i] = True

    # ------------------------------------------------------------------------------------------------------------------
    # 联邦主要操作
    # ------------------------------------------------------------------------------------------------------------------

    # 挑选客户端
    def select_clients(self, task_id):
        # 如果随机加入客户端的比例大于0
        if self.random_join_ratio:
            # 从[self.num_join_clients，self.num_clients]中选中一个数值作为当前随机加入的客户端数量
            self.current_num_join_clients = \
                np.random.choice(range(self.num_join_clients, self.num_clients + 1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients

        # 除上述要求外，可挑选的客户端还需要其数据可进行task_id任务
        selective_clients = [client for client in self.clients if task_id in client.local_tasks]
        self.selected_clients = list(np.random.choice(selective_clients, self.current_num_join_clients, replace=False))

    # 向客户端发送全局模型
    def send_models(self):
        """
        向客户端发送全局模型
        @return:
        """
        # 断言服务器的客户端数不为零
        assert (len(self.selected_clients) > 0)
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters(self.global_model)
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            client.send_time_cost['num_rounds'] += 1

    # 从客户端接收训练后的本地模型
    def receive_models(self):
        """
        从选中训练的客户端接收其本地模型
        @return:
        """
        # 断言被选中的客户端数不为零
        assert (len(self.selected_clients) > 0)
        # 计算选中的客户端中的活跃客户端数量
        activate_client_num = int((1 - self.client_drop_rate) * self.current_num_join_clients)
        # 随机采样
        activate_clients = random.sample(self.selected_clients, activate_client_num)

        self.received_info = {
            'client_ids': [],
            'client_weights': [],
            'client_models': []
        }

        for client in activate_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                                   client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                self.received_info['client_ids'].append(client.id)
                self.received_info['client_weights'].append(client.train_samples)
                self.received_info['client_models'].append(client.local_model)

        total_client_train_samples = sum(self.received_info['client_weights'])
        for idx, train_samples in enumerate(self.received_info['client_weights']):
            self.received_info['client_weights'][idx] = train_samples / total_client_train_samples

    # 根据本地模型聚合全局模型
    def aggregate_parameters(self):
        """
        根据本地模型聚合全局模型
        @return:
        """
        # 断言客户端上传的模型数量不为零
        assert (len(self.received_info['client_models']) > 0)
        self.global_model = copy.deepcopy(self.received_info['client_models'][0])
        # 将全局模型的参数值清空
        for param in self.global_model.parameters():
            param.data.zero_()
        # 获取全局模型的参数值
        for weight, model in zip(self.received_info['client_weights'], self.received_info['client_models']):
            for server_param, client_param in zip(self.global_model.parameters(), model.parameters()):
                server_param.data += client_param.data.clone() * weight

    def evaluate(self, task_id: int, HLOP_SNN: bool):
        # --------------------------------------------------------------------------------------------------------------
        # 获取实验相关参数（是否是HLOP_SNN相关实验，如果是的话，是否是bptt/ottt相关设置）
        # --------------------------------------------------------------------------------------------------------------
        bptt = False
        ottt = False
        if HLOP_SNN:
            if self.experiment_name.endswith('bptt'):
                bptt = True
            elif self.experiment_name.endswith('ottt'):
                ottt = True

        # 全局模型开启评估模式
        self.global_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('Server Testing', max=((self.xtest[task_id].size(0) - 1) // self.batch_size + 1))

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

                data = self.xtest[task_id][index].float().to(self.device)
                label = self.ytest[task_id][index].to(self.device)

                if HLOP_SNN:
                    if bptt:
                        out_, out = self.global_model(data, task_id, projection=False, update_hlop=False)
                        loss = self.loss(out, label)
                        reset_net(self.global_model)
                    elif ottt:
                        loss = 0.
                        for t in range(self.timesteps):
                            if t == 0:
                                out_fr = self.global_model(data, task_id, projection=False, update_hlop=False, init=True)
                                total_fr = out_fr.clone().detach()
                            else:
                                out_fr = self.global_model(data, task_id, projection=False, update_hlop=False)
                                total_fr += out_fr.clone().detach()
                            loss += self.loss(out_fr, label).detach() / self.timesteps
                        out_, out = total_fr
                    else:
                        data = data.unsqueeze(1)
                        data = data.repeat(1, self.timesteps, 1, 1, 1)
                        out_, out = self.global_model(data, task_id, projection=False, update_hlop=False)
                        loss = self.loss(out, label)
                else:
                    out_, out = self.global_model(data)
                    loss = self.loss(out, label)

                test_num += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out.argmax(1) == label).float().sum().item()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

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

        print("Test Accuracy: {:.4f}".format(test_acc))
        print("Test Loss: {:.4f}".format(test_loss))
        return test_loss, test_acc

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.local_model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.local_model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.local_model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc

    # ------------------------------------------------------------------------------------------------------------------
    # HLOP_SNN相关操作
    # ------------------------------------------------------------------------------------------------------------------
    def adjust_to_HLOP_SNN_before_train_task(self, experiment_name, ncla, task_count,
                                             hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1):
        """
        训练task之前调整HLOP-SNN
        @param experiment_name:
        @param task_count:
        @param hlop_out_num:
        @param hlop_out_num_inc:
        @param hlop_out_num_inc1:
        @param ncla:
        @return:
        """
        if experiment_name.startswith('pmnist'):  # pmnist/pmnist_bptt/pmnist_ottt 实验
            if task_count == 0:
                self.global_model.add_hlop_subspace(hlop_out_num)
                self.global_model.to(self.device)
                for client in self.clients:
                    client.local_model.add_hlop_subspace(hlop_out_num)
                    client.local_model.to(self.device)
            else:
                if task_count % 3 == 0:
                    hlop_out_num_inc[0] -= 20
                    hlop_out_num_inc[1] -= 20
                    hlop_out_num_inc[2] -= 20
                self.global_model.add_hlop_subspace(hlop_out_num_inc)
                for client in self.clients:
                    client.local_model.add_hlop_subspace(hlop_out_num_inc)
        elif experiment_name == 'cifar':  # cifar 实验
            if task_count == 0:
                self.global_model.add_hlop_subspace(hlop_out_num)
                self.global_model.to(self.device)
                for client in self.clients:
                    client.local_model.add_hlop_subspace(hlop_out_num)
                    client.local_model.to(self.device)
            else:
                self.global_model.add_classifier(ncla)
                self.global_model.add_hlop_subspace(hlop_out_num_inc)
                for client in self.clients:
                    client.local_model.add_classifier(ncla)
                    client.local_model.add_hlop_subspace(hlop_out_num_inc)
        elif experiment_name == 'miniimagenet':  # miniimagenet 实验
            if task_count == 0:
                self.global_model.add_hlop_subspace(hlop_out_num)
                self.global_model.to(self.device)
                for client in self.clients:
                    client.local_model.add_hlop_subspace(hlop_out_num)
                    client.local_model.to(self.device)
            else:
                self.global_model.add_classifier(ncla)
                for client in self.clients:
                    client.local_model.add_classifier(ncla)
                if task_count < 6:
                    self.global_model.add_hlop_subspace(hlop_out_num_inc)
                    for client in self.clients:
                        client.local_model.add_hlop_subspace(hlop_out_num_inc)
                else:
                    self.global_model.add_hlop_subspace(hlop_out_num_inc1)
                    for client in self.clients:
                        client.local_model.add_hlop_subspace(hlop_out_num_inc1)
        elif experiment_name.startswith('fivedataset'):  # fivedataset/fivedataset_domain 实验
            if task_count == 0:
                self.global_model.add_hlop_subspace(hlop_out_num)
                self.global_model.to(self.device)
                for client in self.clients:
                    client.local_model.add_hlop_subspace(hlop_out_num)
                    client.local_model.to(self.device)
            else:
                self.global_model.add_classifier(ncla)
                self.global_model.add_hlop_subspace(hlop_out_num_inc)
                for client in self.clients:
                    client.local_model.add_classifier(ncla)
                    client.local_model.add_hlop_subspace(hlop_out_num_inc)

    def adjust_to_HLOP_SNN_after_train_task(self):
        """
        训练task之后调整HLOP-SNN
        @return:
        """
        self.global_model.to('cpu')
        self.global_model.merge_hlop_subspace()
        self.global_model.to(self.device)
        for client in self.clients:
            client.local_model.to('cpu')
            client.local_model.merge_hlop_subspace()
            client.local_model.to(self.device)

    # ------------------------------------------------------------------------------------------------------------------
    # 数据保存、加载操作
    # ------------------------------------------------------------------------------------------------------------------
    def save_global_model(self, model_name):
        """
        保存全局模型
        @param model_name: 模型名称（不需要绝对/相对路径）
        @return:
        """
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        torch.save(self.global_model, os.path.join(self.models_path, f'{model_name}.pt'))

    def load_global_model(self, model_name):
        """
        加载全局模型
        @param model_name: 模型名称（不需要绝对/相对路径）
        @return:
        """
        model_abs_path = os.path.join(self.models_path, model_name)
        assert os.path.exists(model_abs_path)
        self.global_model = torch.load(model_abs_path)
