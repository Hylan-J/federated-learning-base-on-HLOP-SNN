import time

import numpy as np
from progress.bar import Bar
import torch
import torch.nn.functional as F

from FLcore.meter import AverageMeter
from FLcore.utils import accuracy


class Server:
    def __init__(self, args):
        self._args = args
        # 测试集
        self._Xtest = None
        self._Ytest = None
        # 批处理大小
        self._batch_size = args.b
        # 网络模型
        self._global_model = None

    def load_testset(self, Xtest, Ytest):
        self._Xtest = Xtest
        self._Ytest = Ytest

    def load_model(self, global_model):
        self._global_model = global_model

    def update_model_weight(self, weight):
        self._global_model.load_state_dict(weight)

    def issue_model_weight(self):
        return self._global_model.state_dict()

    def evaluate(self, task_id):
        self._global_model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        end = time.time()

        bar = Bar('客户端测试', max=((self._Xtest.size(0) - 1) // self._batch_size + 1))

        test_acc = 0
        test_loss = 0
        num_samples = 0

        batch_idx = 0

        r = np.arange(self._Xtest.size(0))
        with torch.no_grad():
            for i in range(0, len(r), self._batch_size):
                if i + self._batch_size <= len(r):
                    index = r[i: i + self._batch_size]
                else:
                    index = r[i:]
                batch_idx += 1
                input = self._Xtest[index].float().cuda()

                # repeat for time steps
                input = input.unsqueeze(1)
                input = input.repeat(1, self._args.timesteps, 1, 1, 1)

                label = self._Ytest[index].cuda()

                out = self._global_model(input, task_id, projection=False, update_hlop=False)
                loss = F.cross_entropy(out, label)

                num_samples += label.numel()
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
                    size=((self._Xtest.size(0) - 1) // self._batch_size + 1),
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

        test_acc /= num_samples
        test_loss /= num_samples

        return test_acc, test_loss
