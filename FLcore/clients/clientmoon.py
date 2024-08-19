import copy
import time

import torch
import numpy as np
import torch.nn.functional as F

from ..clients.clientbase import Client


class clientMOON(Client):
    def __init__(self, args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs):
        super().__init__(args, id, xtrain, ytrain, xtest, ytest, local_model, **kwargs)
        self.tau = args.MOON_tau
        self.mu = args.MOON_mu
        self.global_model = None
        self.old_local_model = copy.deepcopy(self.local_model)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        self.max_local_epochs = self.local_epochs
        if self.train_slow:
            self.max_local_epochs = np.random.randint(1, self.max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                rep_old = self.old_local_model.base(x).detach()
                rep_global = self.global_model.base(x).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (
                        torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(
                    F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.old_local_model = copy.deepcopy(self.local_model)
        self.train_time_cost['total_cost'] += time.time() - start_time
        self.train_time_cost['num_rounds'] += 1

    def set_parameters(self, model):
        for gp, lp in zip(model.parameters(), self.local_model.parameters()):
            lp.data = gp.data.clone()
        self.global_model = copy.deepcopy(model)

    def train_model(self, task_id, bptt, ottt, **kwargs):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                rep_old = self.old_local_model.base(x).detach()
                rep_global = self.global_model.base(x).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) / (
                        torch.exp(F.cosine_similarity(rep, rep_global) / self.tau) + torch.exp(
                    F.cosine_similarity(rep, rep_old) / self.tau)))
                loss += self.mu * torch.mean(loss_con)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
