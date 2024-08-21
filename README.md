# FL-HLOP-SNN
This is the PyTorch implementation of the federated continual learning algorithm which based on paper: Hebbian Learning based Orthogonal Projection for Continual Learning of Spiking Neural Networks **(ICLR 2024)**. \[[openreview](https://openreview.net/forum?id=MeB86edZ1P)\] \[[arxiv](https://arxiv.org/abs/2402.11984)\]

## Dependencies and Installation
- Python 3 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch, torchvision](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Python packages: `pip install numpy opencv-python`

## Implemented Federated Learning Algorithm
- ### FedAvg
《Communication-Efficient Learning of Deep Networks from Decentralized Data》
[[arxiv](https://arxiv.org/abs/1602.05629)]

- ### FedProx
《Federated Optimization in Heterogeneous Networks》
[[arxiv](https://arxiv.org/abs/1812.06127)]

- ### FedDyn
《Federated Learning Based on Dynamic Regularization》
[[arxiv](https://arxiv.org/abs/2111.04263)]

- ### SCAFFOLD
《SCAFFOLD: Stochastic Controlled Averaging for Federated Learning》
[[arxiv](https://arxiv.org/abs/1910.06378)]

- ### MOON
《Model-Contrastive Federated Learning》
[[arxiv](https://arxiv.org/abs/2103.16257)]

## Training
Run as following examples:

    # pmnist实验，使用FedAvg联邦算法，设置3个客户端，全局通信轮次为10，客户端本地epoch为1，数据集所在文件夹为dataset，实验过程相关日志文件会保存在logs文件夹下
	python main.py --experiment_name pmnist --fed_algorithm FedAvg --num_clients 3 --global_rounds 10 --local_epochs 1 --dataset_path ./dataset --root_path ./logs
	
    # --experiment_name可选：【pmnist、pmnist_bptt、pmnist_ottt、cifar、miniimagenet、fivedataset、fivedataset_domain】
    # --fed_algorithm可选：【FedAvg、SCAFFOLD（暂时没处理好）】

The default hyperparameters are the same as the paper.

## Acknowledgement

Some codes are adpated from [DSR](https://github.com/qymeng94/DSR), [OTTT](https://github.com/pkuxmq/OTTT-SNN), and [spikingjelly](https://github.com/fangwei123456/spikingjelly). Some codes for data processing are adapted from [GPM](https://github.com/sahagobinda/GPM).
