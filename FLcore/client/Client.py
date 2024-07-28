import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Client:
    def __init__(self, id, args):
        self.id = id
        self.args = args

        # 数据集相关质量参数
        self.distribution = None

        # 客户端数据集
        self._xtrain = None
        self._ytrain = None
        self._batch_size = None
        self._train_dataloader = None

        # 客户端网络模型
        self._local_model = None
        self._learning_rate = None
        self._optimizer = None
        self._local_epochs = None

    def config(self, distribution):
        self.distribution = distribution

    def load_trainset(self, xtrain, ytrain, batch_size):
        self._xtrain = xtrain
        self._ytrain = ytrain
        self._batch_size = batch_size
        self._train_dataloader = DataLoader(dataset=self._xtrain, batch_size=self._batch_size, shuffle=True)

    def load_model(self, local_model, learning_rate, local_epochs):
        self._local_model = local_model
        self._learning_rate = learning_rate
        self._optimizer = optim.SGD(self._local_model.parameters(), lr=self._learning_rate, momentum=0.9)
        self._local_epochs = local_epochs

    def commit_model_weight(self):
        return self._local_model.state_dict()

    def update_model_weight(self, weight):
        self._local_model.load_state_dict(weight)

    def train(self, task_id, data):
        # 生成输出文件夹 ---------------------------------------------------------------------------------------------------- #
        out_dir = self.args.out_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            print(f'Mkdir {out_dir}.')
        else:
            print(out_dir)

        # 生成输出文件夹下子文件夹 models ------------------------------------------------------------------------------------- #
        pt_dir = os.path.join(out_dir, 'models')
        if not os.path.exists(pt_dir):
            os.makedirs(pt_dir)
            print(f'Mkdir {pt_dir}.')

        # 写入实验参数
        with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
            args_txt.write(str(args))

        print('*' * 100)
        print('Task {:2d} ({:s})'.format(key_, data[key_]['name']))
        print('*' * 100)

        writer = SummaryWriter(os.path.join(out_dir, 'logs_task{task_id}'.format(task_id=task_id)))
        # 训练集
        xtrain = self._xtrain
        ytrain = self._ytrain

        task_list.append(key_)

        # 如果需要重放 ------------------------------------------------------------------------------------------------- #
        if args.replay:
            # save samples for memory replay
            replay_data[task_id] = {'x': [], 'y': []}
            for class_ in range(num_classes):
                num = args.memory_size
                index = 0
                while num > 0:
                    if ytrain[index] == class_:
                        replay_data[task_id]['x'].append(xtrain[index])
                        replay_data[task_id]['y'].append(ytrain[index])
                        num -= 1
                    index += 1
            replay_data[task_id]['x'] = torch.stack(replay_data[task_id]['x'], dim=0)
            replay_data[task_id]['y'] = torch.stack(replay_data[task_id]['y'], dim=0)

        # 使用脉冲进行hlop更新
        hlop_with_wfr = True
        if args.not_hlop_with_wfr:
            hlop_with_wfr = False

        # 当任务id为0时候
        if task_id == 0:
            model = models.spiking_MLP(snn_setting, num_classes=num_classes, n_hidden=800, ss=args.sign_symmetric,
                                       fa=args.feedback_alignment, hlop_with_wfr=hlop_with_wfr,
                                       hlop_spiking=args.hlop_spiking, hlop_spiking_scale=args.hlop_spiking_scale,
                                       hlop_spiking_timesteps=args.hlop_spiking_timesteps)
            model.add_hlop_subspace(hlop_out_num)
            model = model.cuda()
        else:
            if task_id % 3 == 0:
                hlop_out_num_inc[0] -= 20
                hlop_out_num_inc[1] -= 20
                hlop_out_num_inc[2] -= 20
            model.add_hlop_subspace(hlop_out_num_inc)

        params = []
        for name, param in model.named_parameters():
            if 'hlop' not in name:
                if task_id != 0:
                    if len(param.size()) != 1:
                        params.append(param)
                else:
                    params.append(param)

        # 设置优化器 --------------------------------------------------------------------------------------------------- #
        if args.opt == 'SGD':
            optimizer = torch.optim.SGD(params, lr=args.lr)
        elif args.opt == 'Adam':
            optimizer = torch.optim.Adam(params, lr=args.lr)
        else:
            raise NotImplementedError(args.opt)

        # 设置学习率 --------------------------------------------------------------------------------------------------- #
        lr_scheduler = None
        if args.lr_scheduler == 'StepLR':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.lr_scheduler == 'CosALR':
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
            lr_lambda = lambda cur_epoch: (cur_epoch + 1) / args.warmup if cur_epoch < args.warmup else 0.5 * (
                    1 + math.cos((cur_epoch - args.warmup) / (args.T_max - args.warmup) * math.pi))
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            raise NotImplementedError(args.lr_scheduler)

        for epoch in range(1, args.epochs + 1):
            start_time = time.time()
            model.train()
            if task_id != 0:
                model.fix_bn()

            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            end = time.time()

            bar = Bar('Processing', max=((xtrain.size(0) - 1) // args.b + 1))

            train_loss = 0
            train_acc = 0
            train_samples = 0
            batch_idx = 0

            r = np.arange(xtrain.size(0))
            np.random.shuffle(r)
            for i in range(0, len(r), args.b):
                if i + args.b <= len(r):
                    index = r[i: i + args.b]
                else:
                    index = r[i:]
                batch_idx += 1
                x = xtrain[index].float().cuda()

                # repeat for time steps
                x = x.unsqueeze(1)
                x = x.repeat(1, args.timesteps, 1, 1, 1)

                label = ytrain[index].cuda()

                optimizer.zero_grad()
                if task_id == 0:
                    if args.baseline:
                        out = model(x, task_id, projection=False, update_hlop=False)
                    else:
                        if epoch <= args.hlop_start_epochs:
                            out = model(x, task_id, projection=False, update_hlop=False)
                        else:
                            out = model(x, task_id, projection=False, update_hlop=True)
                else:
                    if args.baseline:
                        out = model(x, task_id, projection=False, proj_id_list=[0], update_hlop=False,
                                    fix_subspace_id_list=[0])
                    else:
                        if epoch <= args.hlop_start_epochs:
                            out = model(x, task_id, projection=True, proj_id_list=[0], update_hlop=False,
                                        fix_subspace_id_list=[0])
                        else:
                            out = model(x, task_id, projection=True, proj_id_list=[0], update_hlop=True,
                                        fix_subspace_id_list=[0])

                loss = F.cross_entropy(out, label)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * label.numel()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(out.data, label.data, topk=(1, 5))
                losses.update(loss, x.size(0))
                top1.update(prec1.item(), x.size(0))
                top5.update(prec5.item(), x.size(0))

                train_samples += label.numel()
                train_acc += (out.argmax(1) == label).float().sum().item()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx,
                    size=((xtrain.size(0) - 1) // args.b + 1),
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

            train_loss /= train_samples
            train_acc /= train_samples

            writer.add_scalar('train_loss', train_loss, epoch)
            writer.add_scalar('train_acc', train_acc, epoch)
            lr_scheduler.step()

            test_loss, test_acc = test(args, model, xtest, ytest, task_id)

            writer.add_scalar('test_loss', test_loss, epoch)
            writer.add_scalar('test_acc', test_acc, epoch)

            total_time = time.time() - start_time
            print(
                f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, total_time={total_time}, escape_time={(datetime.datetime.now() + datetime.timedelta(seconds=total_time * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}')

        # save accuracy
        jj = 0
        for ii in np.array(task_list)[0:task_id + 1]:
            xtest = data[ii]['test']['x']
            ytest = data[ii]['test']['y']
            _, acc_matrix[task_id, jj] = test(args, model, xtest, ytest, ii)
            jj += 1
        print('Accuracies =')
        for i_a in range(task_id + 1):
            print('\t', end='')
            for j_a in range(acc_matrix.shape[1]):
                print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
            print()

        model.merge_hlop_subspace()

        # 如果重放并且起码参与了一个任务
        if args.replay and task_id >= 1:
            print('memory replay\n')
            params = []
            for name, param in model.named_parameters():
                if 'hlop' not in name:
                    if len(param.size()) != 1:
                        params.append(param)

            if args.opt == 'SGD':
                optimizer = torch.optim.SGD(params, lr=args.replay_lr)
            elif args.opt == 'Adam':
                optimizer = torch.optim.Adam(params, lr=args.replay_lr)
            else:
                raise NotImplementedError(args.opt)

            lr_scheduler = None
            if args.lr_scheduler == 'StepLR':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
            elif args.lr_scheduler == 'CosALR':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.replay_T_max)
            else:
                raise NotImplementedError(args.lr_scheduler)

            for epoch in range(1, args.replay_epochs + 1):
                start_time = time.time()
                model.train()
                model.fix_bn()

                batch_per_task = args.replay_b
                task_data_num = replay_data[0]['x'].size(0)
                r = np.arange(task_data_num)
                np.random.shuffle(r)
                for i in range(0, task_data_num, batch_per_task):
                    optimizer.zero_grad()
                    for replay_task_id in range(task_id + 1):
                        xtrain = replay_data[replay_task_id]['x']
                        ytrain = replay_data[replay_task_id]['y']

                        if i + batch_per_task <= task_data_num:
                            index = r[i: i + batch_per_task]
                        else:
                            index = r[i:]

                        x = xtrain[index].float().cuda()

                        # repeat for time steps
                        x = x.unsqueeze(1)
                        x = x.repeat(1, args.timesteps, 1, 1, 1)

                        label = ytrain[index].cuda()

                        # out = model(x, replay_taskid, projection=False, update_hlop=True)
                        out = model(x, replay_task_id, projection=False, update_hlop=False)
                        loss = F.cross_entropy(out, label)
                        loss.backward()
                    optimizer.step()

                lr_scheduler.step()

            # 保存准确率
            jj = 0
            for ii in np.array(task_list)[0:task_id + 1]:
                xtest = data[ii]['test']['x']
                ytest = data[ii]['test']['y']
                _, acc_matrix[task_id, jj] = test(args, model, xtest, ytest, ii)
                jj += 1
            print('Accuracies =')
            for i_a in range(task_id + 1):
                print('\t', end='')
                for j_a in range(acc_matrix.shape[1]):
                    print('{:5.1f}% '.format(acc_matrix[i_a, j_a] * 100), end='')
                print()

        # 保存模型
        torch.save(model.state_dict(), os.path.join(pt_dir, 'model_task{task_id}.pth'.format(task_id=task_id)))