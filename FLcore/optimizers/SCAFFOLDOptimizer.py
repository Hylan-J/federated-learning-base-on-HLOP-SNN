from torch.optim import Optimizer


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)

    def step(self, server_controls, client_controls):
        for param_group in self.param_groups:
            for param, server_control, client_control in zip(param_group['params'], server_controls, client_controls):
                param.data.add_(other=(param.grad.data + server_control - client_control), alpha=-param_group['lr'])
