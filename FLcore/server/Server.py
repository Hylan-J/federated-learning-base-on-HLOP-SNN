import copy
import os
import time

import numpy as np
from progress.bar import Bar
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import nn

from FLcore.meter import AverageMeter
from FLcore.utils import accuracy


class Server:
    def __init__(self):
        self.global_model = None  # 网络模型
        self.xtest = None  # 测试集图像
        self.ytest = None  # 测试集标签
        self.tasks = None  # 数据集中拥有的任务
        self.save_dir = None  # 存储的目录

    def configure_testset_info(self, xtest, ytest):
        """
        配置服务器的测试数据集信息
        @param xtest:
        @param ytest:
        @return:
        """
        self.xtest = xtest
        self.ytest = ytest
        self.tasks = list(self.xtest.keys())

    def configure_global_model(self, global_model):
        self.global_model = global_model

    def configure_data_save_path(self, save_dir):
        self.save_dir = save_dir

    def evaluate(self, task_id, global_epoch, args, is_bptt, is_ottt):
        writer = SummaryWriter(os.path.join(self.save_dir, "logs_task{task_id}".format(task_id=task_id)))
        self.global_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('服务器测试', max=((self.xtest[task_id].size(0) - 1) // args.b + 1))

        test_acc = 0
        test_loss = 0
        test_samples = 0

        batch_idx = 0

        r = np.arange(self.xtest[task_id].size(0))
        with torch.no_grad():
            for i in range(0, len(r), args.b):
                if i + args.b <= len(r):
                    index = r[i: i + args.b]
                else:
                    index = r[i:]
                batch_idx += 1

                if not is_ottt:
                    input = self.xtest[task_id][index].float().cuda()

                    # repeat for time steps
                    input = input.unsqueeze(1)
                    input = input.repeat(1, args.timesteps, 1, 1, 1)

                    label = self.ytest[task_id][index].cuda()

                    out = self.global_model(input, task_id, projection=False, update_hlop=False)
                    loss = F.cross_entropy(out, label)

                    if is_bptt:
                        self.reset_net(self.global_model)
                else:
                    input = self.xtest[index].float().cuda()

                    label = self.ytest[index].cuda()

                    loss = 0.
                    for t in range(args.timesteps):
                        if t == 0:
                            out_fr = self.global_model(input, task_id, projection=False, update_hlop=False, init=True)
                            total_fr = out_fr.clone().detach()
                        else:
                            out_fr = self.global_model(input, task_id, projection=False, update_hlop=False)
                            total_fr += out_fr.clone().detach()
                        loss += F.cross_entropy(out_fr, label).detach() / args.timesteps
                    out = total_fr

                test_samples += label.numel()
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
                    size=((self.xtest[task_id].size(0) - 1) // args.b + 1),
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

        test_acc /= test_samples
        test_loss /= test_samples
        writer.add_scalar('test_loss', test_loss, global_epoch)
        writer.add_scalar('test_acc', test_acc, global_epoch)
        return test_acc, test_loss

    def update_model_weight(self, weight):
        self.global_model.load_state_dict(weight)

    def issue_model_weight(self):
        return self.global_model.state_dict()

    def reset_net(self, net: nn.Module):
        """
        将网络的状态重置。做法是遍历网络中的所有 ``Module``，若含有 ``reset()`` 函数，则调用。
        @param net: 任何属于 ``nn.Module`` 子类的网络
        @return:
        """
        for m in net.modules():
            if hasattr(m, 'reset'):
                m.reset()
