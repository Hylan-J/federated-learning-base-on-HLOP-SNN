from copy import deepcopy

import torch


def FedAvg(local_weights):
    global_weight = deepcopy(local_weights[0])
    for key in global_weight.keys():
        for i in range(1, len(local_weights)):
            global_weight[key] += local_weights[i][key]
        if 'num_batches_tracked' in key:
            global_weight[key] = global_weight[key].true_divide(len(local_weights))
        else:
            global_weight[key] = torch.div(global_weight[key], len(local_weights))
    return global_weight
