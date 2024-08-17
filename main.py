import copy
import argparse
import os
import time
import logging
import warnings

import numpy as np
import torch
import torchvision

from FLcore.servers.serveravg import FedAvg
from FLcore.servers.serverprox import FedProx
from FLcore.servers.serverscaffold import SCAFFOLD

from FLcore.utils.prepare_utils import prepare_dataset, prepare_model

# 创建日志记录器对象
logger = logging.getLogger()
# 设置日志级别
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")

# 设置torch的随机种子
torch.manual_seed(0)


def run(args):
    xtrain, ytrain, xtest, ytest, taskcla = prepare_dataset(args.experiment_name, args.dataset_path, 0)
    model = prepare_model(args.experiment_name, args, 10)
    time_list = []
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        if args.fed_algorithm == 'FedAvg':
            server = FedAvg(args, xtrain, ytrain, xtest, ytest, taskcla, model.to(args.device), i)
        elif args.fed_algorithm == 'SCAFFOLD':
            server = SCAFFOLD(args, xtrain, ytrain, xtest, ytest, taskcla, model.to(args.device), i)
        elif args.fed_algorithm == 'FedProx':
            server = FedProx(args, xtrain, ytrain, xtest, ytest, taskcla, model.to(args.device), i)
        server.train(args.experiment_name, True)

        time_list.append(time.time() - start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    print("All done!")


if __name__ == "__main__":
    # 获取命令行参数解析对象
    parser = argparse.ArgumentParser()

    parser.add_argument('-opt', type=str, help='use which optimizer. SGD or Adam', default='SGD')
    parser.add_argument('-lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=100, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=200, type=int, help='T_max for CosineAnnealingLR')
    parser.add_argument('-warmup', default=5, type=int, help='warmup epochs for learning rate')
    parser.add_argument('-cnf', type=str)

    parser.add_argument('-hlop_start_epochs', default=0, type=int, help='the start epoch to update hlop')
    parser.add_argument('-hlop_proj_type', type=str,
                        help='choice for projection type in bottom implementation, default is input, can choose weight for acceleration of convolutional operations',
                        default='input')

    parser.add_argument('-memory_size', default=50, type=int, help='memory size for replay')
    parser.add_argument('-replay_lr', default=0.001, type=float, help='learning rate for replay')
    parser.add_argument('-replay_T_max', default=20, type=int, help='T_max for CosineAnnealingLR for replay')
    parser.add_argument('-feedback_alignment', action='store_true', help='feedback alignment')
    parser.add_argument('-sign_symmetric', action='store_true', help='use sign symmetric')
    parser.add_argument('-baseline', action='store_true', help='baseline')

    # SNN settings
    parser.add_argument('-timesteps', default=20, type=int)
    parser.add_argument('-Vth', default=0.3, type=float)
    parser.add_argument('-tau', default=1.0, type=float)
    parser.add_argument('-delta_t', default=0.05, type=float)
    parser.add_argument('-alpha', default=0.3, type=float)
    parser.add_argument('-train_Vth', default=1, type=int)
    parser.add_argument('-Vth_bound', default=0.0005, type=float)
    parser.add_argument('-rate_stat', default=0, type=int)

    parser.add_argument('-not_hlop_with_wfr', action='store_true', help='use spikes for hlop update')
    parser.add_argument('-hlop_spiking', action='store_true', help='use hlop with lateral spiking neurons')
    parser.add_argument('-hlop_spiking_scale', default=20., type=float)
    parser.add_argument('-hlop_spiking_timesteps', default=1000., type=float)

    # 普遍参数
    parser.add_argument('-go', "--goal", type=str, default="test", help="The goal for this experiment")

    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")

    parser.add_argument('-pv', "--prev", type=int, default=0, help="之前的运行次数")
    parser.add_argument('-t', "--times", type=int, default=1, help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1, help="Rounds gap for evaluation")
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)

    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< 学习率相关参数 >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser.add_argument("--server_learning_rate", type=float, default=1.0, help="服务器学习率")
    parser.add_argument("--client_learning_rate", type=float, default=0.005, help="客户端学习率")
    parser.add_argument("--learning_rate_decay", type=bool, default=False)
    parser.add_argument("--learning_rate_decay_gamma", type=float, default=0.99)

    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    # 实际参数？
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0, help="参与训练但中途退出的客户端比例")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0, help="本地训练时，速度慢的客户端比例")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0, help="发送全局模型时，速度慢的客户端比例")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
                        help="是否根据时间成本对每轮客户进行分组和选择")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000, help="丢弃慢客户端的阈值")

    parser.add_argument("--experiment_name", type=str, default="pmnist", help="实验名称")
    parser.add_argument('--fed_algorithm', type=str, default='FedAvg', help='联邦算法')
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="实验设备")
    parser.add_argument("--device_id", type=str, default="0", help="实验设备的id")
    parser.add_argument("--num_clients", type=int, default=3, help="客户端数量")
    parser.add_argument("--batch_size", type=int, default=64, help="训练数据批处理大小")
    parser.add_argument("--replay_batch_size", default=64, type=int, help="回放数据批处理大小")
    parser.add_argument("--global_rounds", type=int, default=10, help="全局通信轮次")
    parser.add_argument("--replay_global_rounds", type=int, default=10, help="全局重放通信轮次")
    parser.add_argument("--local_epochs", type=int, default=1, help="本地训练轮次")
    parser.add_argument("--replay_epochs", type=int, default=20, help="本地回放轮次")
    parser.add_argument("--dataset_path", type=str, default='./dataset', help="数据集的根路径")
    parser.add_argument("--root_path", type=str, default='./logs', help="文件保存文件夹的根路径")
    # 解析命令行参数
    args = parser.parse_args()

    args.root_path = os.path.join('logs', args.experiment_name+time.strftime(" %Y-%m-%d %H：%M：%S"))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.fed_algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local epochs: {}".format(args.local_epochs))
    print("Total number of clients: {}".format(args.num_clients))
    print("Clients join in each round: {}".format(args.join_ratio))
    print("Clients randomly join: {}".format(args.random_join_ratio))
    print("Client drop rate: {}".format(args.client_drop_rate))
    print("Client select regarding time: {}".format(args.time_select))
    if args.time_select:
        print("Time threthold: {}".format(args.time_threthold))
    print("Running times: {}".format(args.times))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))

    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    if args.device == "cuda":
        print("Cuda device id: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    print("DLG attack: {}".format(args.dlg_eval))
    if args.dlg_eval:
        print("DLG attack round gap: {}".format(args.dlg_gap))
    print("Total number of new clients: {}".format(args.num_new_clients))
    print("Fine tuning epoches on new clients: {}".format(args.fine_tuning_epoch_new))
    print("=" * 50)

    run(args)
