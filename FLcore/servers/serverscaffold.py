import copy
import os
import random
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter

from ..clients.clientscaffold import clientSCAFFOLD
from ..servers.serverbase import Server
from ..utils.prepare_utils import prepare_bptt_ottt, prepare_hlop_out_XXX

__all__ = ['SCAFFOLD']


class SCAFFOLD(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model, times):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model, times)
        self.set_slow_clients()
        self.set_clients(clientSCAFFOLD, self.xtrain, self.ytrain, self.xtest, self.ytest, model)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # 全局控制变量
        self.global_controls = [torch.zeros_like(param) for param in self.global_model.parameters()]
        """for param in self.global_model.parameters():
            self.global_controls.append(torch.zeros_like(param))"""

        self.learning_rate = args.server_learning_rate
        self.time_cost = []

    def train(self, experiment_name: str, replay: bool):
        bptt, ottt = prepare_bptt_ottt(experiment_name)
        hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1 = prepare_hlop_out_XXX(experiment_name)

        if bptt or ottt:
            replay = False

        task_learned = []
        task_count = 0

        tasks = [task_id for task_id, ncla in self.taskcla]
        total_task_num = len(tasks)

        acc_matrix = np.zeros((total_task_num, total_task_num))

        for task_id, ncla in self.taskcla:
            task_learned.append(task_id)
            writer = SummaryWriter(os.path.join(self.args.root_path, 'task{task_id}'.format(task_id=task_id)))

            if replay:
                for client in self.clients:
                    client.set_replay_data(task_id, ncla)

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

            # 对于任务task_id，进行联邦训练
            for global_round in range(1, self.global_rounds + 1):
                start_time = time.time()
                self.select_clients(task_id)
                self.send_models()

                for client in self.selected_clients:
                    client.train(task_id, bptt, ottt)

                self.receive_models()
                """if self.dlg_eval and global_round % self.dlg_gap == 0:
                        self.call_dlg(global_round)"""
                self.aggregate_parameters(task_id)
                self.time_cost.append(time.time() - start_time)
                print('-' * 25, 'Task', task_id, 'Time Cost', '-' * 25, self.time_cost[-1])
                # 当前轮次达到评估轮次
                if global_round % self.eval_gap == 0:
                    print(f"\n-------------Round number: {global_round}-------------")
                    print("\nEvaluate global model")
                    test_loss, test_acc = self.evaluate(task_id, bptt, ottt)
                    writer.add_scalar('test_loss', test_loss, global_round)
                    writer.add_scalar('test_acc', test_acc, global_round)
                """if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                        break"""

            jj = 0
            for ii in np.array(task_learned)[0:task_count + 1]:
                _, acc_matrix[task_count, jj] = self.evaluate(ii, bptt, ottt)
                jj += 1
            print('Accuracies =')
            for i_a in range(task_count + 1):
                print('\t', end='')
                for j_a in range(acc_matrix.shape[1]):
                    print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                print()

            self.global_model.merge_hlop_subspace()
            self.send_models()

            # 如果重放并且起码参与了一个任务
            if replay and task_count >= 1:
                print('memory replay\n')

                for replay_global_round in range(1, self.replay_global_rounds + 1):
                    self.select_clients(task_id)
                    self.send_models()

                    for client in self.clients:
                        client.replay(task_learned)
                    self.receive_models()
                    self.aggregate_parameters(task_id)

                # 保存准确率
                jj = 0
                for ii in np.array(task_learned)[0:task_count + 1]:
                    _, acc_matrix[task_count, jj] = self.evaluate(ii, bptt, ottt)
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
                self.evaluate(task_id, bptt, ottt)

            task_count += 1

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

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 重写的方法 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
            delta_model, delta_control = self.clients[idx].calculate_delta_model_and_control_param(task_id)
            for global_model_param, local_model_param in zip(global_model.parameters(), delta_model):
                global_model_param.data += local_model_param.data.clone() / self.num_join_clients * self.learning_rate
            for global_control_param, local_control_param in zip(global_controls, delta_control):
                global_control_param.data += local_control_param.data.clone() / self.num_clients
        self.global_model = global_model
        self.global_controls = global_controls

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model, self.global_controls)
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
