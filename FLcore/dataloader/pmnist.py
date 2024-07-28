import os
import sys

import numpy as np
import torch
from sklearn.utils import shuffle
from torchvision import datasets, transforms


def get(data_dir='./data/', seed=0, fixed_order=False):
    data = {}
    taskcla = []

    # MNIST图像size
    size = [1, 28, 28]

    mnist_dir = data_dir
    pmnist_dir = os.path.join(mnist_dir, 'binary_pmnist')

    # 10个任务
    nperm = 10
    # 生成任务id的种子
    seeds = np.array(list(range(nperm)), dtype=int)
    # 如果次序不固定
    if not fixed_order:
        # 打乱种子
        seeds = shuffle(seeds, random_state=seed)

    # 如果没有之前处理过数据集
    if not os.path.isdir(pmnist_dir):
        os.makedirs(pmnist_dir)
        # 预加载
        # MNIST
        mean = (0.1307,)
        std = (0.3081,)
        mnist_tensor_data = {}
        mnist_tensor_data['train'] = datasets.MNIST(mnist_dir, train=True, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))
        mnist_tensor_data['test'] = datasets.MNIST(mnist_dir, train=False, download=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]))

        for index_, class_ in enumerate(seeds):
            print(index_, end=',')
            sys.stdout.flush()
            data[index_] = {}
            data[index_]['name'] = 'pmnist-{:d}'.format(index_)
            data[index_]['num_classes'] = 10

            for type_ in ['train', 'test']:
                # 每次加载一份数据的数据加载器
                loader = torch.utils.data.DataLoader(mnist_tensor_data[type_], batch_size=1, shuffle=False)
                # 创建一个空的字典，用于存储数据集中每个样本的处理结果
                data[index_][type_] = {'x': [], 'y': []}

                # 遍历数据加载器，逐个处理图像和标签
                for image, target in loader:
                    # 将图像展平成一维张量并转换为numpy数组
                    aux = image.view(-1).numpy()
                    # 根据指定的随机状态（class_ * 100 + index_）对数据进行随机打乱
                    aux = shuffle(aux, random_state=class_ * 100 + index_)
                    # 将打乱后的数据重新转换为PyTorch的FloatTensor，并恢复原始形状size
                    image = torch.FloatTensor(aux).view(size)
                    # 将处理后的图像数据添加到数据字典中的'x'列表中
                    data[index_][type_]['x'].append(image)
                    # 将对应的标签（转换为numpy数组后的第一个元素）添加到数据字典中的'y'列表中
                    data[index_][type_]['y'].append(target.numpy()[0])

            # "Unify" and save
            for type_ in ['train', 'test']:
                # 将图像数据拼接起来并reshape成多个1*28*28维度的张量
                data[index_][type_]['x'] = torch.stack(data[index_][type_]['x']).view(-1, size[0], size[1], size[2])
                # 将标签转化为PyTorch的FloatTensor并展平成一维张量
                data[index_][type_]['y'] = torch.LongTensor(np.array(data[index_][type_]['y'], dtype=int)).view(-1)
                # 保存数据集为二进制文件
                torch.save(data[index_][type_]['x'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(class_) + type_ + 'x.bin'))
                torch.save(data[index_][type_]['y'],os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(class_) + type_ + 'y.bin'))
        print()

    # 如果之前处理过数据集 ----------------------------------------------------------------------------------------------- #
    else:
        # 加载数据集
        for index_, class_ in enumerate(seeds):
            # 创建具有'name', 'ncla', 'train', 'test'键的字典，不设置值
            data[index_] = dict.fromkeys(['name', 'num_classes', 'train', 'test'])
            data[index_]['num_classes'] = 10
            data[index_]['name'] = 'pmnist-{:d}'.format(index_)

            for type_ in ['train', 'test']:
                data[index_][type_] = {'x': [], 'y': []}
                data[index_][type_]['x'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(class_) + type_ + 'x.bin'))
                data[index_][type_]['y'] = torch.load(os.path.join(os.path.expanduser(pmnist_dir), 'data' + str(class_) + type_ + 'y.bin'))

    # Validation
    #for t in data.keys():
    #    r=np.arange(data[t]['train']['x'].size(0))
    #    # r=np.array(shuffle(r,random_state=seed),dtype=int)
    #    r=np.array(r,dtype=int)
    #    nvalid=int(pc_valid*len(r))
    #    ivalid=torch.LongTensor(r[:nvalid])
    #    itrain=torch.LongTensor(r[nvalid:])
    #    data[t]['valid'] = {}
    #    data[t]['valid']['x']=data[t]['train']['x'][ivalid].clone()
    #    data[t]['valid']['y']=data[t]['train']['y'][ivalid].clone()
    #    data[t]['train']['x']=data[t]['train']['x'][itrain].clone()
    #    data[t]['train']['y']=data[t]['train']['y'][itrain].clone()

    # Others
    n = 0
    for key_ in data.keys():
        taskcla.append((key_, data[key_]['num_classes']))
        n += data[key_]['num_classes']
    data['num_classes'] = n

    return data, taskcla, size

########################################################################################################################
