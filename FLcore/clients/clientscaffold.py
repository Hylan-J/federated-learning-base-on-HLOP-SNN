import time

import numpy as np
import torch

from ..clients.clientbase import Client
from ..optimizers.SCAFFOLDOptimizer import SCAFFOLDOptimizer

__all__ = ['clientSCAFFOLD']


class clientSCAFFOLD(Client):
    def __init__(self, args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs):
        super().__init__(args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs)

        self.num_batches = None
        self.optimizer = SCAFFOLDOptimizer(self.local_model.parameters(), lr=self.learning_rate)
        """
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        """
        self.client_c = []
        for param in self.local_model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        self.global_model = None

    def train(self, task_id, bptt, ottt):
        self.local_model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        start_time = time.time()

        SCAFFOLD = {'global_c': self.global_c, 'client_c': self.client_c}
        super().train_metrics(task_id, bptt, ottt, SCAFFOLD=SCAFFOLD)

        # self.model.cpu()
        self.num_batches = np.ceil(self.xtrain[task_id].size(0) // self.batch_size)
        self.update_yc(max_local_epochs)
        # self.delta_c, self.delta_y = self.delta_yc(max_local_epochs)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model, global_c):
        for new_param, old_param in zip(model.parameters(), self.local_model.parameters()):
            old_param.data = new_param.data.clone()

        self.global_c = global_c
        self.global_model = model

    def update_yc(self, max_local_epochs):
        for ci, c, x, yi in zip(self.client_c, self.global_c, self.global_model.parameters(),
                                self.local_model.parameters()):
            ci.data = ci - c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi)

    def delta_yc(self, max_local_epochs):
        delta_y = []
        delta_c = []
        for c, x, yi in zip(self.global_c, self.global_model.parameters(), self.local_model.parameters()):
            delta_y.append(yi - x)
            delta_c.append(- c + 1 / self.num_batches / max_local_epochs / self.learning_rate * (x - yi))

        return delta_y, delta_c
