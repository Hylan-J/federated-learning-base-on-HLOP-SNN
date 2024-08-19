import time
import copy

import numpy as np
from progress.bar import Bar

from ..meter import AverageMeter
from ..clients.clientbase import Client
from ..utils import accuracy
from ..utils.model_utils import reset_net


class clientProx(Client):
    def __init__(self, args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs):
        super().__init__(args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs)
        self.mu = args.FedProx_mu
        self.global_model = None

    def train(self, task_id, bptt, ottt):
        if self.train_slow:
            self.local_epochs = np.random.randint(1, self.local_epochs // 2)
        start_time = time.time()
        self.train_model(task_id, bptt, ottt)
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def set_parameters(self, model):
        """
        从服务器接收到全局的模型更新本地的模型
        @param model: 全局的模型
        @return:
        """
        self.global_model = copy.deepcopy(self.local_model)
        for p, gp, lp in zip(model.parameters(), self.global_model.parameters(), self.local_model.parameters()):
            gp.data = p.data.clone()
            lp.data = p.data.clone()

    def train_model(self, task_id, bptt, ottt, **kwargs):
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

        for local_epoch in range(1, self.local_epochs + 1):
            for i in range(0, len(samples_index), self.batch_size):
                # 如果可以获取完整的批次，那么就获取完整批次
                if i + self.batch_size <= len(samples_index):
                    index = samples_index[i: i + self.batch_size]
                # 如果没有完整的批次可供获取，那么获取所有剩下的
                else:
                    index = samples_index[i:]
                batch_idx += 1

                # 获取一个批次的数据和标签
                x = xtrain[index].float().to(self.device)
                y = ytrain[index].to(self.device)

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
                        # 根据FedProx论文计算loss
                        proximal_term = 0.0
                        for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                            proximal_term += (lp - gp).norm(2)
                        loss = self.loss(out_fr, y) / self.timesteps + (self.mu / 2) * proximal_term

                        loss.backward()
                        total_loss += loss.detach()
                        if self.args.online_update:
                            self.optimizer.step()
                    if not self.args.online_update:
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
                    # 根据FedProx论文计算loss
                    proximal_term = 0.0
                    for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                        proximal_term += (lp - gp).norm(2)
                    loss = self.loss(out, y) + (self.mu / 2) * proximal_term
                    # loss反向传播
                    loss.backward()
                    # 参数更新
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
                    # 根据FedProx论文计算loss
                    proximal_term = 0.0
                    for lp, gp in zip(self.local_model.parameters(), self.global_model.parameters()):
                        proximal_term += (lp - gp).norm(2)
                    loss = self.loss(out, y) + (self.mu / 2) * proximal_term
                    # loss反向传播
                    loss.backward()
                    # 参数更新
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
