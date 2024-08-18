import os
import time

import numpy as np
from tensorboardX import SummaryWriter

from ..clients.clientprox import clientProx
from ..servers.serverbase import Server
from ..utils.prepare_utils import prepare_bptt_ottt, prepare_hlop_out


class FedProx(Server):
    def __init__(self, args, xtrain, ytrain, xtest, ytest, taskcla, model, times):
        super().__init__(args, xtrain, ytrain, xtest, ytest, taskcla, model, times)
        self.set_slow_clients()
        self.set_clients(clientProx, self.xtrain, self.ytrain, self.xtest, self.ytest, model)
        self.time_cost = []

    def train(self, experiment_name: str, replay: bool, HLOP_SNN: bool):
        bptt, ottt = prepare_bptt_ottt(experiment_name)
        hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1 = prepare_hlop_out(experiment_name)

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

            # 如果使用HLOP-SNN方法，那么就需要根据相关参数进行调整
            if HLOP_SNN:
                self.adjust_to_HLOP_SNN_before_train_task(experiment_name, ncla, task_count,
                                                          hlop_out_num, hlop_out_num_inc, hlop_out_num_inc1)

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
                    client.train(task_id, bptt, ottt)
                # ④服务器接收训练后的客户端模型
                self.receive_models()
                # ⑤服务器聚合全局模型
                self.aggregate_parameters()

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

            if HLOP_SNN:
                self.adjust_to_HLOP_SNN_after_train_task()

            # 如果重放并且起码参与了一个任务
            if replay and task_count >= 1:
                print('memory replay\n')
                for client in self.clients:
                    client.set_optimizer(task_id, experiment_name, True)
                    client.set_learning_rate_scheduler(experiment_name, True)

                for replay_global_round in range(1, self.replay_global_rounds + 1):
                    self.select_clients(task_id)
                    self.send_models()

                    for client in self.clients:
                        client.replay(task_learned)
                    self.receive_models()
                    self.aggregate_parameters()

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
                self.set_new_clients(clientProx, self.xtrain, self.ytrain, self.xtest, self.ytest)
                print(f"\n-------------Fine tuning round-------------")
                print("\nEvaluate new clients")
                self.evaluate(task_id, bptt, ottt)

            task_count += 1
